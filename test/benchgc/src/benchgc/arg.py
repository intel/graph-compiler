################################################################################
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import gc_mlir.ir
from gc_mlir.dialects import tensor
from typing import List


class Arg:
    name: str
    dtype: str | None
    shape: List[int] | None
    fill_type: str | None
    fill_param: List[str]

    def __init__(self, cfg: str):
        cfgs = cfg.split(":")
        self.name = cfgs[0]
        self.dtype = None if cfgs[1] == "" else cfgs[1]

        if cfgs[2] == "":
            self.shape = None
        else:
            self.shape = []
            for dim in cfgs[2].split("x"):
                self.shape.append(int(dim))

        self.fill_type = None if cfgs[3] == "" else cfgs[3]
        self.fill_param = cfgs[4:]

    def get_mlir_dtype(self, ctx: gc_mlir.ir.Context) -> gc_mlir.ir.Type:
        if self.dtype == "f32":
            return gc_mlir.ir.F32Type.get(ctx)
        elif self.dtype == "f64":
            return gc_mlir.ir.F64Type.get(ctx)
        elif self.dtype == "f16":
            return gc_mlir.ir.F16Type.get(ctx)
        elif self.dtype == "bf16":
            return gc_mlir.ir.BF16Type.get(ctx)
        elif self.dtype == "u8":
            return gc_mlir.ir.IntegerType.get_unsigned(8, ctx)
        elif self.dtype == "s8":
            return gc_mlir.ir.IntegerType.get_signed(8, ctx)
        elif self.dtype == "boolean":
            return gc_mlir.ir.IntegerType.get_unsigned(1, ctx)
        elif self.dtype == "f8_e4m3":
            return gc_mlir.ir.Float8E4M3FNType.get(ctx)
        elif self.dtype == "f8_e5m2":
            return gc_mlir.ir.Float8E5M2Type.get(ctx)
        elif self.dtype == "s32":
            return gc_mlir.ir.IntegerType.get_signed(32, ctx)
        else:
            raise Exception("data type not support: %s" % self.dtype)

    def get_ranked_tensor_type(
        self, ctx: gc_mlir.ir.Context
    ) -> gc_mlir.ir.RankedTensorType:
        if self.shape is None:
            raise Exception("shape is unknown")
        return gc_mlir.ir.RankedTensorType.get(self.shape, self.get_mlir_dtype(ctx))

    def get_empty_op(self, ctx: gc_mlir.ir.Context) -> tensor.EmptyOp:
        if self.shape is None:
            raise Exception("shape is unknown")
        return tensor.EmptyOp(self.shape, self.get_mlir_dtype(ctx))
