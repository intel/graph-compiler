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

from gc_mlir.dialects import onednn_graph
from gc_mlir.dialects._ods_common import OpView
from gc_mlir.extras import types as T
from gc_mlir.ir import IntegerAttr


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


class MatMulConfig(Config):
    def __init__(
        self,
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
        super().__init__()
        self.M_threads = M_threads
        self.K_threads = K_threads
        self.N_threads = N_threads
        self.M_block = M_block
        self.K_block = K_block
        self.N_block = N_block
        self.innermostM_block = innermostM_block
        self.innermostK_block = innermostK_block
        self.innermostN_block = innermostN_block

    def __init__(self, op: OpView):
        super().__init__()
        # you can set the default value by matmul_op
        # cpu_counts = os.cpu_count()
        # self.input_a_shape = op.input_a.type.shape
        # self.input_b_shape = op.input_b.type.shape
        # self.input_a_dtype = op.input_a.type.element_type
        # print(self.input_a_shape, self.input_a_dtype)
        self.M_threads = 1
        self.K_threads = 1
        self.N_threads = 1
        self.M_block = 1
        self.K_block = 1
        self.N_block = 1
        self.innermostM_block = 1
        self.innermostK_block = 1
        self.innermostN_block = 1

    def init_candidates(self):
        # you can set the candidates by info form matmul op
        self.field_candidates["M_block"] = [16, 32]
        self.field_candidates["K_block"] = [16, 32, 64]
        self.field_candidates["N_block"] = [16]

    def init_constraints(self):
        # example: using lambda to add constraints
        # self.field_constraints["K_block"] = (
        #     lambda MatMulConfig, K_block: MatMulConfig.M_block <= K_block
        # )
        self.field_constraints["M_block"] = None
        self.field_constraints["K_block"] = None
        self.field_constraints["N_block"] = None

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
