################################################################################
# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################

import json
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from benchgc.tuner.config_filter import BloomFilter, HashSetFilter
from benchgc.tuner.op_config import *
from benchgc.tuner.utils import attach_configs_to_ir, gen_configs_from_ir
from gc_mlir import ir


class TuningSpace:
    """
    The class works as a bridge between the tuner and the configs in MLIR module.
    """

    DEFAULT_SPACE_PERCENT = 1.0

    def __init__(
        self, ir_module: ir.Module, space_percent: float = DEFAULT_SPACE_PERCENT
    ):
        self.initial_ir = ir_module
        self.graph_config = gen_configs_from_ir(ir_module)
        self.space_size = 1
        self.flatten_candidates = []
        self.flatten_field_name = []
        self.flatten_constraints = []
        self.ind_candidate_to_config = {}
        candidate_ind = 0
        for config_ind, config in enumerate(self.graph_config):
            for field_name, candidates in config.field_candidates.items():
                self.space_size = self.space_size * len(candidates)
                self.flatten_candidates.append(candidates)
                self.flatten_field_name.append(field_name)
                self.flatten_constraints.append(config.field_constraints[field_name])
                self.ind_candidate_to_config[candidate_ind] = config_ind
                candidate_ind += 1
        self.space_size = int(self.space_size * space_percent)

    def make_config_from_indexes(self, indexes: List[int]):
        """
        Make a config from a list of indexes of candidates.
        """
        graph_config = deepcopy(self.graph_config)
        for cid, candidate in enumerate(self.flatten_candidates):
            val = candidate[indexes[cid]]
            config = graph_config[self.ind_candidate_to_config[cid]]
            field_name = self.flatten_field_name[cid]
            setattr(config, field_name, val)
        return graph_config

    def get_cur_config(self, candidate_ind: int):
        """
        Get the current config with a incoming candidate index
        """
        return self.graph_config[self.ind_candidate_to_config[candidate_ind]]

    def verify_config(self, candidate_idx, val) -> bool:
        """
        Verify the config with constraints
        """
        config = self.get_cur_config(candidate_idx)
        field_name = self.flatten_field_name[candidate_idx]
        constraint = self.flatten_constraints[candidate_idx]
        val = self.flatten_candidates[candidate_idx][val]
        setattr(config, field_name, val)
        if constraint and (not constraint(config, val)):
            return False
        # verify the config when it has all fields
        if (candidate_idx + 1) == len(
            self.flatten_candidates
        ) or self.ind_candidate_to_config[
            candidate_idx + 1
        ] != self.ind_candidate_to_config[
            candidate_idx
        ]:
            return config.verify()
        return True

    def filter_next_candidates(self, candidate_idx, val) -> List[int]:
        """
        Get the next candidates with the incoming candidate index and value
        """
        field_name = self.flatten_field_name[candidate_idx]
        config = self.get_cur_config(candidate_idx)
        setattr(
            config,
            field_name,
            self.flatten_candidates[candidate_idx][val],
        )
        if (candidate_idx + 1) >= len(self.flatten_candidates):
            return []
        constraint = self.flatten_constraints[candidate_idx + 1]
        if constraint:
            next_candidates = self.flatten_candidates[candidate_idx + 1]
            return [
                index
                for index, value in enumerate(next_candidates)
                if constraint(config, value)
            ]
        else:
            return list(range(len(self.flatten_candidates[candidate_idx + 1])))


class Tuner(ABC):
    """
    Class for creating different configs and choose the config with best perf
    """

    DEFAULT_BATCH_SIZE = 50
    DEFAULT_EARLY_STOP = -1
    DEFAULT_TIMEOUT = -1
    DEFAULT_MAX_ITERS = sys.maxsize

    def __init__(
        self,
        batch_executor,
        tunning_space: TuningSpace,
        batch_size=DEFAULT_BATCH_SIZE,
        early_stop=DEFAULT_EARLY_STOP,
        checkpoint="",
        tuner_verbose=False,
    ):
        self.batch_executor = batch_executor
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.best_cost = sys.float_info.max
        self.best = []
        self.iter = 0
        self.last_update_iter = 0
        self.skipped_num = 0
        self.tunning_space = tunning_space
        self.checkpoint = checkpoint
        if self.checkpoint:
            os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)
        self.tuner_verbose = tuner_verbose
        assert len(tunning_space.graph_config), "There are no tunable ops"

    def tuner_update(self, config_indices_batch: List[List[int]], costs: List[float]):
        """
        Update after each batch of configs was executed
        """
        if min(costs) < self.best_cost:
            self.best_cost = min(costs)
            self.best = config_indices_batch[costs.index(min(costs))]
        if self.checkpoint:
            self.save_status()

    @abstractmethod
    def get_next_config_indices_batch(self) -> List[List[int]]:
        """
        Get the next batch of config indices
        """
        pass

    @abstractmethod
    def load_status(self):
        """
        Load the Tuner status from the checkpoint
        """
        pass

    @abstractmethod
    def save_status(self):
        """
        Save the Tuner status to the checkpoint
        """
        pass

    def tuner_finish(self, tuning_time):
        """
        Execute when tuning is finished
        """
        print("Tuning ends in", tuning_time, "s")
        best_config = self.tunning_space.make_config_from_indexes(self.best)
        print("Best cost:", self.best_cost, "ms")
        print("Best config:", [str(single_cfg) for single_cfg in best_config])
        attach_configs_to_ir(self.tunning_space.initial_ir, best_config)
        print(
            "mlir:\n",
            self.tunning_space.initial_ir,
        )

    def run(self, max_iter: int = DEFAULT_MAX_ITERS, timeout: int = DEFAULT_TIMEOUT):
        """
        Start of tuning process
        """
        if self.early_stop > 0 and self.iter - self.last_update_iter > self.early_stop:
            # in case of resuming from a saved state and it has already
            # early-stopped
            print("Early stop now")
            return
        start_time = time.time()
        spaces_size = self.tunning_space.space_size
        while self.iter < max_iter and self.iter < spaces_size:
            config_indices_batch = self.get_next_config_indices_batch()
            if not config_indices_batch:
                print("Tuner returns empty batch, early stop now")
                break
            if len(config_indices_batch) > min(
                max_iter - self.iter, spaces_size - self.iter
            ):
                config_indices_batch = config_indices_batch[
                    : min(max_iter - self.iter, spaces_size - self.iter)
                ]

            old_iter = self.iter
            self.iter += len(config_indices_batch)
            perf_result = []
            ir_modules = []
            for config_indexes in config_indices_batch:
                real_config = self.tunning_space.make_config_from_indexes(
                    config_indexes
                )
                # todo : ir.Module can not support deepcopy
                new_ir = ir.Module.parse(
                    str(self.tunning_space.initial_ir),
                    self.tunning_space.initial_ir.context,
                )
                attach_configs_to_ir(new_ir, real_config)
                ir_modules.append(new_ir)
            if self.tuner_verbose:
                print("start to execute the batch of configs ...")
            res = self.batch_executor(ir_modules)
            perf_result = [item[1] for item in res]
            # print the perf result of each config
            if self.tuner_verbose:
                for i, config_indexes in enumerate(config_indices_batch):
                    real_config = self.tunning_space.make_config_from_indexes(
                        config_indexes
                    )
                    perf_to_cfg = {"cost": perf_result[i], "cfg": repr(real_config)}
                    print(json.dumps(perf_to_cfg))

            old_best = self.best_cost
            self.tuner_update(config_indices_batch, perf_result)
            print(
                "[",
                self.iter,
                "/",
                min(max_iter, spaces_size),
                "] skipped:",
                self.skipped_num,
                "best:",
                self.best_cost,
                "ms",
            )
            if self.best_cost != old_best:
                self.last_update_iter = old_iter
            else:
                if (
                    self.early_stop > 0
                    and old_iter - self.last_update_iter > self.early_stop
                ):
                    print("Early stop now")
                    break
            if timeout >= 0 and time.time() - start_time > timeout:
                print("Tuning timeout...")
                break
        self.tuner_finish(time.time() - start_time)


class GridTuner(Tuner):
    """
    Tuner with grid serach
    """

    def __init__(
        self,
        batch_executor,
        tunning_space: TuningSpace,
        batch_size=Tuner.DEFAULT_BATCH_SIZE,
        early_stop=Tuner.DEFAULT_EARLY_STOP,
        checkpoint="",
        tuner_verbose=False,
    ):
        super().__init__(
            batch_executor,
            tunning_space,
            batch_size,
            early_stop,
            checkpoint,
            tuner_verbose,
        )
        self.current_idx = 0
        self.cumulative_size = [1] * len(self.tunning_space.flatten_candidates)
        self.cumulative_size[-1] = 1
        for i in range(len(self.cumulative_size) - 2, -1, -1):
            self.cumulative_size[i] = self.cumulative_size[i + 1] * len(
                self.tunning_space.flatten_candidates[i + 1]
            )
        if self.checkpoint:
            self.load_status()

    def get_next_config_indices_batch(self) -> list:
        config_indices_batch = []
        while len(config_indices_batch) < self.batch_size:
            if self.current_idx >= self.tunning_space.space_size:
                break
            config_ids = [-1] * len(self.tunning_space.flatten_candidates)
            remain = self.current_idx
            valid_config_idx = True
            for j in range(len(config_ids)):
                config_ids[j] = remain // self.cumulative_size[j]
                valid_config_idx = self.tunning_space.verify_config(j, config_ids[j])
                if not valid_config_idx:
                    break
                remain = remain % self.cumulative_size[j]
            self.current_idx = self.current_idx + 1
            if valid_config_idx:
                config_indices_batch.append(config_ids)
                if self.tuner_verbose:
                    print(
                        "find valid config",
                        self.tunning_space.make_config_from_indexes(config_ids),
                    )
            else:
                self.skipped_num += 1
                if self.tuner_verbose:
                    print("bad config, skip...")
        return config_indices_batch

    def save_status(self):
        save_dict = {
            "iter": self.iter,
            "last_update_iter": self.last_update_iter,
            "best": self.best,
            "best_cost": self.best_cost,
            "current_idx": self.current_idx,
            "skipped_num": self.skipped_num,
        }
        with open(self.checkpoint, "w") as file:
            json.dump(save_dict, file, indent=4)

    def load_status(self):
        print("continue tuning from checkpoint...")
        with open(
            self.checkpoint,
            "r",
        ) as file:
            try:
                data = json.load(file)
                assert set(
                    [
                        "iter",
                        "last_update_iter",
                        "best",
                        "best_cost",
                        "current_idx",
                        "skipped_num",
                    ]
                ) == set(data.keys())
                self.iter = data["iter"]
                self.last_update_iter = data["last_update_iter"]
                self.best = data["best"]
                self.best_cost = data["best_cost"]
                self.current_idx = data["current_idx"]
                self.skipped_num = data["skipped_num"]
            except Exception as e:
                print("load checkpoint failed", e)


class GATuner(Tuner):
    """Tuner with Genetic Algorithm"""

    DEFAULT_ELITE_NUM = 9
    DEFAULT_MUTATION_PROB = 0.1
    DEFAULT_RANDOM_SEED = 0
    DEFAULT_EXPECTED_TUNE_NUM = 0

    def __init__(
        self,
        batch_executor,
        tuning_space,
        pop_size=Tuner.DEFAULT_BATCH_SIZE,
        early_stop=Tuner.DEFAULT_EARLY_STOP,
        checkpoint="",
        tuner_verbose=False,
        elite_num: int = DEFAULT_ELITE_NUM,
        mutation_prob: float = DEFAULT_MUTATION_PROB,
        random_seed: int = DEFAULT_RANDOM_SEED,
        expected_tune_num: int = DEFAULT_EXPECTED_TUNE_NUM,
    ):
        super().__init__(
            batch_executor,
            tuning_space,
            pop_size,
            early_stop,
            checkpoint,
            tuner_verbose,
        )
        self.elite_num = min(elite_num, pop_size)
        self.mutation_prob = mutation_prob
        self.pop_size = pop_size
        self.cur_mutation_prob = mutation_prob
        self.prev_results = []
        self.elites = []
        random.seed(random_seed)
        if expected_tune_num == 0:
            self.filter = HashSetFilter()
        else:
            self.filter = BloomFilter(expected_tune_num, err_rate=0.01)

        self.candidate_indices = [[]] * len(self.tunning_space.flatten_candidates)
        self.candidate_indices[0] = list(
            range(len(self.tunning_space.flatten_candidates[0]))
        )
        if self.checkpoint:
            self.load_status()

    def save_status(self):
        save_dict = {
            "iter": self.iter,
            "last_update_iter": self.last_update_iter,
            "best": self.best,
            "best_cost": self.best_cost,
            "skipped_num": self.skipped_num,
            "cur_mutation_prob": self.cur_mutation_prob,
            "prev_results": self.prev_results,
            "elites": self.elites,
            "tuned": list(self.filter.save()),
        }
        with open(self.checkpoint, "w") as file:
            json.dump(save_dict, file, indent=4)

    def load_status(self):
        print("continue tuning from checkpoint...")
        with open(
            self.checkpoint,
            "r",
        ) as file:
            try:
                data = json.load(file)
                assert set(
                    [
                        "iter",
                        "last_update_iter",
                        "best",
                        "best_cost",
                        "skipped_num",
                        "cur_mutation_prob",
                        "prev_results",
                        "elites",
                        "tuned",
                    ]
                ) == set(data.keys())
                self.iter = data["iter"]
                self.last_update_iter = data["last_update_iter"]
                self.best = data["best"]
                self.best_cost = data["best_cost"]
                self.skipped_num = data["skipped_num"]
                self.cur_mutation_prob = data["cur_mutation_prob"]
                self.prev_results = data["prev_results"]
                self.elites = data["elites"]
                self.filter.load(data["tuned"])
            except Exception as e:
                print("load checkpoint failed", e)

    def set_field(self, gene, idx, val):
        gene[idx] = val
        self.update_candidate_indices(idx, val)

    def update_candidate_indices(self, idx, val):
        next_candidates = self.tunning_space.filter_next_candidates(idx, val)
        if idx + 1 < len(self.candidate_indices):
            self.candidate_indices[idx + 1] = next_candidates

    @staticmethod
    def update_mutation_prob(prob, lower_bound, move_up):
        if move_up:
            prob = min(prob * 1.01, 0.5)
        else:
            prob = max(prob * 0.98, lower_bound)
        return prob

    @staticmethod
    def random_choice(prob_range) -> int:
        random_val = random.randint(0, sys.maxsize) / sys.maxsize
        for i in range(len(prob_range)):
            if random_val <= prob_range[i]:
                return i
        return -1

    def push_to_tune(self, to_tune, gene) -> bool:
        if self.filter.already_met(gene):
            self.cur_mutation_prob = GATuner.update_mutation_prob(
                self.cur_mutation_prob, self.mutation_prob, True
            )
            return False
        if gene in to_tune:
            self.cur_mutation_prob = GATuner.update_mutation_prob(
                self.cur_mutation_prob, self.mutation_prob, True
            )
            return False

        graph_cfg = self.tunning_space.make_config_from_indexes(gene)
        for cfg in graph_cfg:
            if not cfg.verify():
                return False

        to_tune.append(gene)
        self.cur_mutation_prob = GATuner.update_mutation_prob(
            self.cur_mutation_prob, self.mutation_prob, False
        )
        return True

    def get_next_config_indices_batch(self) -> list:
        prob_range = [0.0] * len(self.prev_results)
        total_score = 0
        for i, prev_result in enumerate(self.prev_results):
            total_score += prev_result[1]
            prob_range[i] = total_score
        prob_range = [x / total_score for x in prob_range]
        to_tune = []
        for i in range(self.pop_size):
            self.get_next_config(prob_range, to_tune)

        if self.tuner_verbose:
            print("to_tune list:")
            for to_tune_config in to_tune:
                print(self.tunning_space.make_config_from_indexes(to_tune_config))

        if len(to_tune) < self.pop_size:
            print(
                f"GA Cannot generate enough unmet genes in this batch (batch_size={self.pop_size})"
            )
        return to_tune

    def get_next_config(self, prob_range, to_tune):
        max_tries = 20
        try_cnt = 0
        while try_cnt < max_tries:
            try_cnt += 1
            if not self.elites:
                gene = [-1] * len(self.tunning_space.flatten_candidates)
                need_repo = True
                redo_cnt = 0
                while redo_cnt < 50 and need_repo:
                    need_repo = False
                    for j in range(len(gene)):
                        # try to randomly pick one candidate
                        data, success = GATuner.random_item_from(
                            self.candidate_indices[j]
                        )
                        if not success:
                            need_repo = True
                            break
                        else:
                            self.set_field(gene, j, data)
                    redo_cnt += 1
                if need_repo:
                    print("Cannot create a valid random gene")
                if self.push_to_tune(to_tune, gene):
                    return
            else:
                assert len(self.prev_results) > 0
                # print("len(prob_range) = ", len(prob_range))
                if len(prob_range) == 1:
                    return
                gene_size = len(self.tunning_space.flatten_candidates)
                first_gene = GATuner.random_choice(prob_range)
                second_gene = GATuner.random_choice(prob_range)
                while second_gene == first_gene:
                    second_gene = GATuner.random_choice(prob_range)

                joint_point = random.randint(0, gene_size)

                new_gene = [-1] * gene_size
                need_redo = False
                for j in range(gene_size):
                    candidates = self.candidate_indices[j]
                    if not candidates:
                        need_redo = True
                        continue
                    if (
                        random.randint(0, sys.maxsize) / sys.maxsize
                    ) < self.cur_mutation_prob:
                        self.set_field(
                            new_gene, j, GATuner.random_item_from(candidates)[0]
                        )
                    else:
                        #  inherit from parents
                        left_gene = self.prev_results[first_gene][0][j]
                        right_gene = self.prev_results[second_gene][0][j]
                        if j < joint_point:
                            prefered_gene = left_gene
                            unprefered_gene = right_gene
                        else:
                            prefered_gene = right_gene
                            unprefered_gene = left_gene

                        if prefered_gene in candidates:
                            self.set_field(new_gene, j, prefered_gene)
                        elif unprefered_gene in candidates:
                            self.set_field(new_gene, j, unprefered_gene)
                        else:
                            self.set_field(
                                new_gene, j, GATuner.random_item_from(candidates)[0]
                            )
                if need_redo:
                    print("need_redo")
                    continue

                if self.push_to_tune(to_tune, new_gene):
                    return

    def tuner_update(
        self, config_indices_batch: List[List[int]], perf_result: List[float]
    ):
        self.prev_results.clear()
        for i, config_indices in enumerate(config_indices_batch):
            self.filter.add(config_indices)
            self.prev_results.append((config_indices, 1 / perf_result[i]))

        for elite in self.elites:
            self.prev_results.append(elite)
        self.elites = sorted(self.prev_results, key=lambda x: x[1], reverse=True)[
            : self.elite_num
        ]
        super().tuner_update(config_indices_batch, perf_result)

    @staticmethod
    def random_item_from(v: List[int]):
        if not v:
            return 0, False
        return v[random.randint(0, len(v) - 1)], True
