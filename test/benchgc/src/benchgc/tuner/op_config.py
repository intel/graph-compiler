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
import math
import os

from gc_mlir.extras import types as T
from gc_mlir.ir import IntegerAttr, OpView


class Config:
    def __init__(self):
        self.field_candidates = {}
        self.field_constraints = {}
        self.init_candidates()
        self.init_constraints()

    def init_candidates(self):
        pass

    def init_constraints(self):
        pass

    def attach_to_ir(self, op: OpView):
        pass


def find_factors(num):
    factors = set()
    for i in range(1, int(math.sqrt(num)) + 1):
        if num % i == 0:
            factors.add(i)
            factors.add(num // i)
    return sorted(factors)


class MatMulConfig(Config):
    def __init__(
        self,
        op: OpView = None,
        M_threads: int = 1,
        K_threads: int = 1,
        N_threads: int = 1,
        M_block: int = 1,
        K_block: int = 1,
        N_block: int = 1,
        innermostM_block: int = 1,
        innermostK_block: int = 1,
        innermostN_block: int = 1,
    ):
        # you can set the default value and candidates by info from matmul_op
        self.M = op.inputs[0].type.shape[0]
        self.K = op.inputs[0].type.shape[1]
        self.N = op.inputs[1].type.shape[1]
        # self.input_a_dtype = str(op.inputs[0].type.element_type)
        self.num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.M_threads = M_threads
        self.K_threads = K_threads
        self.N_threads = N_threads
        self.M_block = M_block
        self.K_block = K_block
        self.N_block = N_block
        self.innermostM_block = innermostM_block
        self.innermostK_block = innermostK_block
        self.innermostN_block = innermostN_block
        super().__init__()

    def init_candidates(self):
        default_blocks = [16, 32, 64]
        self.field_candidates["M_threads"] = find_factors(self.num_threads)
        self.field_candidates["K_threads"] = find_factors(self.num_threads)
        self.field_candidates["N_threads"] = find_factors(self.num_threads)
        self.field_candidates["M_block"] = [
            block for block in default_blocks if self.M >= block
        ]
        self.field_candidates["K_block"] = [
            block for block in default_blocks if self.K >= block
        ]
        self.field_candidates["N_block"] = [
            block for block in default_blocks if self.N >= block
        ]
        self.field_candidates["innermostM_block"] = [
            block for block in default_blocks if self.M >= block
        ]
        self.field_candidates["innermostK_block"] = [
            block for block in default_blocks if self.K >= block
        ]
        self.field_candidates["innermostN_block"] = [
            block for block in default_blocks if self.N >= block
        ]

    def init_constraints(self):
        # example: using lambda to add constraints, adding constraints by the order of the fields
        self.field_constraints["M_threads"] = None
        self.field_constraints["K_threads"] = (
            lambda MatMulConfig, K_threads: self.num_threads
            % (MatMulConfig.M_threads * K_threads)
            == 0
        )
        self.field_constraints["N_threads"] = (
            lambda MatMulConfig, N_threads: self.num_threads
            % (MatMulConfig.M_threads * MatMulConfig.K_threads * N_threads)
            == 0
        )
        self.field_constraints["M_block"] = None
        self.field_constraints["K_block"] = None
        self.field_constraints["N_block"] = None
        self.field_constraints["innermostM_block"] = (
            lambda MatMulConfig, innermostM_block: MatMulConfig.M_block
            % innermostM_block
            == 0
        )
        self.field_constraints["innermostK_block"] = (
            lambda MatMulConfig, innermostK_block: MatMulConfig.K_block
            % innermostK_block
            == 0
        )
        self.field_constraints["innermostN_block"] = (
            lambda MatMulConfig, innermostN_block: MatMulConfig.N_block
            % innermostN_block
            == 0
        )

    def attach_to_ir(self, op: OpView):
        attr_to_field = {
            "Mthreads": self.M_threads,
            "Kthreads": self.K_threads,
            "Nthreads": self.N_threads,
            "MBlock": self.M_block,
            "KBlock": self.K_block,
            "NBlock": self.N_block,
            "innermostMBlock": self.innermostM_block,
            "innermostKBlock": self.innermostK_block,
            "innermostNBlock": self.innermostN_block,
        }
        for name, value in attr_to_field.items():
            op.attributes[name] = IntegerAttr.get(T.i32(), value)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        obj_dict = {
            "MatMulConfig": {
                "M_threads": self.M_threads,
                "K_threads": self.K_threads,
                "N_threads": self.N_threads,
                "M_block": self.M_block,
                "K_block": self.K_block,
                "N_block": self.N_block,
                "innermostM_block": self.innermostM_block,
                "innermostK_block": self.innermostK_block,
                "innermostN_block": self.innermostN_block,
            }
        }
        return json.dumps(obj_dict, indent=4)


OP_TO_CONFIG = {"linalg.matmul": MatMulConfig, "onednn_graph.matmul": MatMulConfig}
