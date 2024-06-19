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

from . import util
import torch
import gc_mlir.ir
import importlib
from gc_mlir.dialects import tensor
from typing import List

class Arg:
    name: str
    dtype: str | None
    shape: List[int] | None
    fill_type: str | None
    fill_param: List[str] | None

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
        self.fill_param = None if cfgs[4] == "" else cfgs[4:]
    
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

    def get_ranked_tensor_type(self, ctx: gc_mlir.ir.Context) -> gc_mlir.ir.RankedTensorType:
        if self.shape is None:
            raise Exception("shape is unknown")
        return gc_mlir.ir.RankedTensorType.get(self.shape, self.get_mlir_dtype(ctx))
    
    def get_empty_op(self, ctx: gc_mlir.ir.Context) -> tensor.EmptyOp:
        if self.shape is None:
            raise Exception("shape is unknown")
        return tensor.EmptyOp(self.shape, self.get_mlir_dtype(ctx))

    def get_filled_tensor(self, verbose: int) -> torch.Tensor | None:
        if self.shape is None or self.dtype is None or self.fill_type is None or self.fill_param is None:
            if verbose >= util.INPUT_VERBOSE:
                print("skip arg %s filling: shape/dtype/fill_type/fill_param is not set" % self.name)
            return None

        if self.fill_type == "N" and len(self.fill_param) == 2:
            # Normal distribution
            mean = float(self.fill_param[0])
            std = float(self.fill_param[1])
            tensor = torch.normal(mean = mean, std = std, size = self.shape)

        elif self.fill_type == "P" and len(self.fill_param) == 1:
            # Poisson distribution
            _lambda = float(self.fill_param[0])
            lambda_tensor = torch.full(self.shape, _lambda)
            tensor = torch.poisson(lambda_tensor)
        elif self.fill_type == "B" and len(self.fill_param) == 2:
            # Binomial distribution
            n = int(self.fill_param[0])
            p = float(self.fill_param[1])
            bdist = torch.distributions.binomial.Binomial(total_count=n, probs=p)
            tensor = bdist.sample(torch.Size(self.shape))
        elif self.fill_type == "U" and len(self.fill_param) == 2:
            # Uniform distribution
            a = float(self.fill_param[0])
            b = float(self.fill_param[1])
            tensor = torch.distributions.uniform.Uniform(a, b).sample(torch.Size(self.shape))
        elif self.fill_type == "F" and len(self.fill_param) == 1:
            # read from pytorch tensor dump file
            filename = self.fill_param[0]
            tensor = torch.load(f = filename)
            if not isinstance(tensor, torch.Tensor):
                raise Exception("torch object from file %s is not a tensor object" % filename)
            if tensor.shape != torch.Size(self.shape):
                raise Exception("tensor object from file %s does not match shape" % filename)
            if tensor.dtype != util.get_dtype(self.dtype):
                raise Exception("tensor object from file %s does not match dtype" % filename)
        elif self.fill_type == "D" and len(self.fill_param) == 2:
            # Driver fill
            driver: str = self.fill_param[0]
            arg: str = self.fill_param[1]

            driver_module = importlib.import_module("fill.%s" % driver)
            tensor = driver_module.fill(self.shape, util.get_dtype(self.dtype), arg)
        else:
            raise Exception("invalid fill type or fill parameter")

        tensor = tensor.to(util.get_dtype(self.dtype))
        if verbose >= util.INPUT_VERBOSE:
            print("fill arg: " + self.name)
            print(tensor)
        return tensor
        



