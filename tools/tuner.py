from copy import deepcopy
import os
import sys
import random
from functools import reduce
import time
from config_filter import *
from op_config import *
from gc_mlir import ir
import utils
import json
from abc import ABC, abstractmethod
from typing import List

need_print = False


class TuningSpace:
    def __init__(self, ir_module: ir.Module):
        self.initial_ir = ir_module
        self.graph_config = utils.gen_configs_from_ir(ir_module)
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

    def make_config_from_indexes(self, indexes: List[int]):
        graph_config = deepcopy(self.graph_config)
        for cid, candidate in enumerate(self.flatten_candidates):
            val = candidate[indexes[cid]]
            config = graph_config[self.ind_candidate_to_config[cid]]
            field_name = self.flatten_field_name[cid]
            setattr(config, field_name, val)
        return graph_config

    def get_cur_config(self, candidate_ind):
        return self.graph_config[self.ind_candidate_to_config[candidate_ind]]

    def verify_config(self, candidate_idx, val) -> bool:
        config = self.get_cur_config(candidate_idx)
        field_name = self.flatten_field_name[candidate_idx]
        constraint = self.flatten_constraints[candidate_idx]
        val = self.flatten_candidates[candidate_idx][val]
        setattr(config, field_name, val)
        if constraint:
            return constraint(config, val)
        return True

    def filter_next_candidates(self, candidate_idx, val) -> List[int]:
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
    def __init__(
        self,
        executor,
        tunning_space: TuningSpace,
        batch_size=50,
        early_stop=-1,
        checkpoint="",
    ):
        self.executor = executor
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

    def tuner_update(self, config_indices_batch: List[List[int]], costs: List[float]):
        if min(costs) < self.best_cost:
            self.best_cost = min(costs)
            self.best = config_indices_batch[costs.index(min(costs))]
        if self.checkpoint:
            self.save_status()

    @abstractmethod
    def get_next_config_indices_batch(self) -> List[List[int]]:
        pass

    @abstractmethod
    def load_status(self):
        pass

    @abstractmethod
    def save_status(self):
        pass

    def tuner_finish(self, tuning_time):
        print("Tuning ends in", tuning_time, "s")
        best_config = self.tunning_space.make_config_from_indexes(self.best)
        print("Best cost:", self.best_cost, "ms")
        print("Best config:", best_config)
        utils.attach_configs_to_ir(self.tunning_space.initial_ir, best_config),
        print(
            "mlir:\n",
            self.tunning_space.initial_ir,
        )

    def run(self, max_iter: int, timeout: int = -1):
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
            if need_print:
                print("config_indices_batch:", config_indices_batch)
            perf_result = []
            for config_indexes in config_indices_batch:
                real_config = self.tunning_space.make_config_from_indexes(
                    config_indexes
                )
                # todo : ir.Module can not support deepcopy
                new_ir = ir.Module.parse(
                    str(self.tunning_space.initial_ir),
                    self.tunning_space.initial_ir.context,
                )
                utils.attach_configs_to_ir(new_ir, real_config)
                _, cost = self.executor(new_ir)
                perf_result.append(cost)

            print(
                "[",
                self.iter,
                "/",
                max_iter,
                "] skipped:",
                self.skipped_num,
                "best:",
                self.best_cost,
                "ms",
            )
            old_best = self.best_cost
            self.tuner_update(config_indices_batch, perf_result)
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
    def __init__(
        self,
        executor,
        tunning_space: TuningSpace,
        batch_size,
        early_stop,
        checkpoint="",
    ):
        super().__init__(executor, tunning_space, batch_size, early_stop, checkpoint)
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
                if need_print:
                    print(self.tunning_space.make_config_from_indexes(config_ids))
            else:
                self.skipped_num += 1
                print("bad config, skip")
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
    def __init__(
        self,
        executor,
        tuning_space,
        pop_size=100,
        early_stop=-1,
        checkpoint="",
        elite_num: int = 9,
        mutation_prob: float = 0.1,
        random_seed: int = 0,
        expected_tune_num: int = 0,
    ):
        super().__init__(executor, tuning_space, pop_size, early_stop, checkpoint)
        self.elite_num = min(elite_num, pop_size)
        self.mutation_prob = mutation_prob
        self.pop_size = pop_size
        self.cur_mutation_prob = mutation_prob
        self.prev_result = []
        self.elites = []
        if expected_tune_num == 0:
            self.filter = HashSetFilter()
        else:
            self.filter = BloomFilter(expected_tune_num)

        self.candidate_indices = [[]] * len(self.tunning_space.flatten_candidates)
        self.candidate_indices[0] = list(
            range(len(self.tunning_space.flatten_candidates[0]))
        )

    def save_status(self):
        save_dict = {
            "iter": self.iter,
            "last_update_iter": self.last_update_iter,
            "best": self.best,
            "best_cost": self.best_cost,
            "skipped_num": self.skipped_num,
            "cur_mutation_prob": self.cur_mutation_prob,
            "prev_result": self.prev_result,
            "elites": self.elites,
            "tuned": self.filter.save(),
        }
        return super().save_status()

    def load_status(self):
        return super().load_status()

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

        to_tune.append(gene)
        self.cur_mutation_prob = GATuner.update_mutation_prob(
            self.cur_mutation_prob, self.mutation_prob, False
        )
        return True

    def get_next_config_indices_batch(self) -> list:
        prob_range = [0.0] * len(self.prev_result)
        total_score = 0
        for i in range(len(self.prev_result)):
            total_score += self.prev_result[i][1]
            prob_range[i] = total_score
        prob_range = [x / total_score for x in prob_range]
        to_tune = []
        for i in range(self.pop_size):
            self.get_next_config(prob_range, to_tune)

        if need_print:
            print("to_tune", to_tune)
            for i in range(len(to_tune)):
                print(self.tunning_space.make_config_from_indexes(to_tune[i]))

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
                assert len(self.prev_result) > 0
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
                        left_gene = self.prev_result[first_gene][0][j]
                        right_gene = self.prev_result[second_gene][0][j]
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
        super().tuner_update(config_indices_batch, perf_result)
        self.prev_result.clear()
        for i in range(len(config_indices_batch)):
            self.filter.add(config_indices_batch[i])
            self.prev_result.append((config_indices_batch[i], 1 / perf_result[i]))

        for elite in self.elites:
            self.prev_result.append(elite)
        self.elites = sorted(self.prev_result, key=lambda x: x[1], reverse=True)[
            : self.elite_num
        ]

    @staticmethod
    def random_item_from(v: List[int]):
        if not v:
            return 0, False
        return v[random.randint(0, len(v) - 1)], True
