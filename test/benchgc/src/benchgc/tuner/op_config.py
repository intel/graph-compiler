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
from gc_mlir.tools import validate_matmul_config


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

    def verify(self) -> bool:
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
        op: OpView,
        MThreads: int = 1,
        KThreads: int = 1,
        NThreads: int = 1,
        MBlock: int = 1,
        KBlock: int = 1,
        NBlock: int = 1,
        innerMostMBlock: int = 1,
        innerMostKBlock: int = 1,
        innerMostNBlock: int = 1,
    ):
        # you can set the default value and candidates by info from matmul_op
        self.m = op.inputs[0].type.shape[0]
        self.k = op.inputs[0].type.shape[1]
        self.n = op.inputs[1].type.shape[1]
        self.input_a_dtype = str(op.inputs[0].type.element_type)
        self.num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.m_threads = MThreads
        self.k_threads = KThreads
        self.n_threads = NThreads
        self.m_block = MBlock
        self.k_block = KBlock
        self.n_block = NBlock
        self.innermost_m_block = innerMostMBlock
        self.innermost_k_block = innerMostKBlock
        self.innermost_n_block = innerMostNBlock
        super().__init__()

    def init_candidates(self):
        default_blocks = [16, 32, 64, 128, 256, 512]
        default_innermost_blocks = [16, 32]
        self.field_candidates["m_threads"] = find_factors(self.num_threads)
        self.field_candidates["k_threads"] = find_factors(self.num_threads)
        self.field_candidates["n_threads"] = find_factors(self.num_threads)
        self.field_candidates["m_block"] = [
            block for block in default_blocks if self.m >= block
        ]
        self.field_candidates["k_block"] = [
            block for block in default_blocks if self.k >= block
        ]
        self.field_candidates["n_block"] = [
            block for block in default_blocks if self.n >= block
        ]
        self.field_candidates["innermost_m_block"] = [
            block for block in default_innermost_blocks if self.m >= block
        ]
        self.field_candidates["innermost_k_block"] = [
            block for block in default_innermost_blocks if self.k >= block
        ]
        self.field_candidates["innermost_n_block"] = [
            block for block in default_innermost_blocks if self.n >= block
        ]

    def init_constraints(self):
        # example: using lambda to add constraints, adding constraints by the order of the fields
        self.field_constraints["m_threads"] = None
        self.field_constraints["k_threads"] = (
            lambda MatMulConfig, k_threads: self.num_threads
            % (MatMulConfig.m_threads * k_threads)
            == 0
        )
        self.field_constraints["n_threads"] = (
            lambda MatMulConfig, n_threads: self.num_threads
            % (MatMulConfig.m_threads * MatMulConfig.k_threads * n_threads)
            == 0
        )
        self.field_constraints["m_block"] = None
        self.field_constraints["k_block"] = None
        self.field_constraints["n_block"] = None
        self.field_constraints["innermost_m_block"] = (
            lambda MatMulConfig, innermost_m_block: MatMulConfig.m_block
            % innermost_m_block
            == 0
        )
        self.field_constraints["innermost_k_block"] = (
            lambda MatMulConfig, innermost_k_block: MatMulConfig.k_block
            % innermost_k_block
            == 0
        )
        self.field_constraints["innermost_n_block"] = (
            lambda MatMulConfig, innermost_n_block: MatMulConfig.n_block
            % innermost_n_block
            == 0
        )

    def verify(self):
        allow_indivisible_innerblock = False
        is_vnni_mm2d = True if self.input_a_dtype == "bf16" else False
        return validate_matmul_config(
            [
                self.m_threads,
                self.n_threads,
                self.k_threads,
                self.m_block,
                self.n_block,
                self.k_block,
                self.innermost_m_block,
                self.innermost_n_block,
                self.innermost_k_block,
            ],
            [self.m, self.n, self.k],
            allow_indivisible_innerblock,
            is_vnni_mm2d,
        )

    def attach_to_ir(self, op: OpView):
        attr_to_field = {
            "MThreads": self.m_threads,
            "KThreads": self.k_threads,
            "NThreads": self.n_threads,
            "MBlock": self.m_block,
            "KBlock": self.k_block,
            "NBlock": self.n_block,
            "innerMostMBlock": self.innermost_m_block,
            "innerMostKBlock": self.innermost_k_block,
            "innerMostNBlock": self.innermost_n_block,
        }
        for name, value in attr_to_field.items():
            op.attributes[name] = IntegerAttr.get(T.i32(), value)

    def __repr__(self) -> str:
        return str(
            [
                self.m_threads,
                self.k_threads,
                self.n_threads,
                self.m_block,
                self.k_block,
                self.n_block,
                self.innermost_m_block,
                self.innermost_k_block,
                self.innermost_n_block,
            ]
        )

    def __str__(self) -> str:
        obj_dict = {
            "MatMulConfig": {
                "MThreads": self.m_threads,
                "KThreads": self.k_threads,
                "NThreads": self.n_threads,
                "MBlock": self.m_block,
                "KBlock": self.k_block,
                "NBlock": self.n_block,
                "innerMostMBlock": self.innermost_m_block,
                "innerMostKBlock": self.innermost_k_block,
                "innerMostNBlock": self.innermost_n_block,
            }
        }
        return json.dumps(obj_dict, indent=4)


OP_TO_CONFIG = {"linalg.matmul": MatMulConfig, "onednn_graph.matmul": MatMulConfig}
