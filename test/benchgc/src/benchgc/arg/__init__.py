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

import argparse
import torch
import numpy
from typing import List, Self, Tuple, Callable
import benchgc.util
import importlib
import gc_mlir.ir
import gc_mlir.dialects.tensor

class Arg:
    dtype: str
    shape: List[int]

    # filling type if arg is an input arg
    # compare type if arg is an output arg
    type: str

    param: List[str]
    index: int

    def __init__(self, cfg: str, index: int):
        cfgs = cfg.split(":")
        tensor_type = cfgs[0].split("x")
        self.dtype = tensor_type[-1]

        self.shape = []
        for dim in tensor_type[:-1]:
            self.shape.append(int(dim))

        self.type = cfgs[1]
        self.param = cfgs[2:]
        self.index = index

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

    def get_mlir_type(self, ctx: gc_mlir.ir.Context) -> gc_mlir.ir.Type:
        if self.shape == []:
            return self.get_mlir_dtype(ctx)
        else:
            return gc_mlir.ir.RankedTensorType.get(self.shape, self.get_mlir_dtype(ctx))

    def get_ranked_tensor_type(
        self, ctx: gc_mlir.ir.Context
    ) -> gc_mlir.ir.RankedTensorType:
        return gc_mlir.ir.RankedTensorType.get(self.shape, self.get_mlir_dtype(ctx))

    def get_empty_op(self, ctx: gc_mlir.ir.Context) -> gc_mlir.dialects.tensor.EmptyOp:
        if self.shape == []:
            raise Exception("shape is unknown")
        return gc_mlir.dialects.tensor.EmptyOp(self.shape, self.get_mlir_dtype(ctx))

    def set_default_fill_param(self, flags: argparse.Namespace, argin: List[Self], argout: List[Self]):

        if self.shape == [] or self.dtype == "" or self.type == "":
            raise Exception("arg%d filling: shape/dtype/fill_type is not set" % self.index)
        if self.type == "D" and len(self.param) == 0:
            # need to generate a default param for driver filling here
            if flags.driver not in ["linalg"]:
                raise Exception("unsupported driver %s for default filling" % flags.driver)

            if flags.case in ["add", "div", "mul"]:
                self.param = [
                    "binary",
                    "src0" if self.index == 0 else "src1",
                    argin[0].dtype,
                    argin[1].dtype,
                    argout[0].dtype,
                ]
            elif flags.case in [
                "batch_matmul",
                "batch_matmul_transpose_a",
                "batch_matmul_transpose_b",
                "batch_matvec",
                "batch_mmt4d",
                "batch_reduce_matmul",
                "batch_vecmat",
                "matmul_transpose_b",
            ]:
                self.param = [
                    "matmul",
                    "src" if self.index == 0 else "wei",
                    argin[0].dtype,
                    argin[1].dtype,
                    argout[0].dtype,
                ]

                if (
                    flags.case == "batch_matmul_transpose_a"
                    and self.index == 0
                    or flags.case == "batch_matmul_transpose_b"
                    or flags.case == "matmul_transpose_b"
                    and self.index == 1
                    or flags.case == "batch_matmul"
                    and self.index == 0
                    or flags.case == "batch_matvec"
                    or flags.case == "batch_vecmat" and self.index == 0
                ):
                    self.param.append(str(self.shape[-1]))

                elif flags.case == "batch_reduce_matmul" and self.index == 0:
                    self.param.append(str(self.shape[0] * self.shape[-1]))
                elif flags.case == "batch_reduce_matmul" and self.index == 1:
                    self.param.append(str(self.shape[0] * self.shape[-2]))
                elif flags.case == "batch_mmt4d":
                    self.param.append(str(self.shape[-1] * self.shape[-3]))
                else:
                    self.param.append(str(self.shape[-2]))
            elif flags.case in ["abs", "negf", "exp"]:
                self.param = ["eltwise", flags.case]
                if flags.case in ["abs", "exp"]:
                    self.param.extend(["", ""])
                elif flags.case == "negf":
                    self.param.extend(["-1", "0"])

    def set_default_compare_param(self, flags: argparse.Namespace, argin: List[Self], argout: List[Self]):
        if self.shape == [] or self.dtype == "" or self.type == "":
            raise Exception("arg%d filling: shape/dtype/fill_type is not set" % self.index)
        if self.type == "D" and len(self.param) == 0:
            # need to generate a default param for driver filling here
            if flags.driver not in ["linalg"]:
                raise Exception("unsupported driver %s for default compare strategy" % flags.driver)

            if flags.case in ["add", "div", "mul"]:
                self.param = ["binary",]
            elif flags.case in [
                "batch_matmul",
                "batch_matmul_transpose_a",
                "batch_matmul_transpose_b",
                "batch_matvec",
                "batch_mmt4d",
                "batch_reduce_matmul",
                "batch_vecmat",
                "matmul_transpose_b",
            ]:
                self.param = ["matmul"]
            elif flags.case in ["abs", "negf", "exp"]:
                self.param = ["eltwise"]

def fill_tensor(flags: argparse.Namespace, arg: Arg, idx: int) -> torch.Tensor:
    if arg.dtype == "" or arg.type == "":
        raise Exception("arg%d filling: dtype/fill_type is not set" % idx)

    if arg.type == "N" and len(arg.param) == 2:
        # Normal distribution
        mean = float(arg.param[0])
        std = float(arg.param[1])
        tensor = torch.normal(mean=mean, std=std, size=arg.shape)

    elif arg.type == "P" and len(arg.param) == 1:
        # Poisson distribution
        _lambda = float(arg.param[0])
        lambda_tensor = torch.full(arg.shape, _lambda)
        tensor = torch.poisson(lambda_tensor)
    elif arg.type == "B" and len(arg.param) == 2:
        # Binomial distribution
        n = int(arg.param[0])
        p = float(arg.param[1])
        bdist = torch.distributions.binomial.Binomial(total_count=n, probs=p)
        tensor = bdist.sample(torch.Size(arg.shape))
    elif arg.type == "U" and len(arg.param) == 2:
        # Uniform distribution
        a = float(arg.param[0])
        b = float(arg.param[1])
        tensor = torch.distributions.uniform.Uniform(a, b).sample(torch.Size(arg.shape))
    elif arg.type == "F" and len(arg.param) == 1:
        # read from pytorch tensor dump file
        filename = arg.param[0]
        tensor = torch.load(f=filename)
        if not isinstance(tensor, torch.Tensor):
            raise Exception(
                "torch object from file %s is not a tensor object" % filename
            )
        if tensor.shape != torch.Size(arg.shape):
            raise Exception(
                "tensor object from file %s does not match shape" % filename
            )
        if tensor.dtype != benchgc.util.get_dtype(arg.dtype):
            raise Exception(
                "tensor object from file %s does not match dtype" % filename
            )
    elif arg.type == "D" and len(arg.param) > 0:
        # Driver fill
        driver: str = arg.param[0]
        driver_module = importlib.import_module("benchgc.arg.%s" % driver)
        tensor = driver_module.fill(
            arg.shape, benchgc.util.get_dtype(arg.dtype), arg.param[1:]
        )
    else:
        raise Exception("invalid fill type or fill parameter")

    tensor = tensor.to(benchgc.util.get_dtype(arg.dtype))
    if flags.verbose >= benchgc.util.INPUT_VERBOSE:
        print("fill arg%d: " % idx)
        print(tensor)
    return tensor


def compare_tensor(arg: Arg, ref: torch.Tensor, res: torch.Tensor, verbose: int) -> Tuple[bool, bool | None]:
    if arg.type == "P": # p2p check
        threshold = float(arg.param[0])
        zero_percent = float(arg.param[1])
        return p2p(threshold, zero_percent, ref, res, verbose)
    if arg.type == "N": # norm check
        threshold = float(arg.param[0])
        return norm(threshold, ref, res, verbose)
    elif arg.type == "D" and len(arg.param) > 0: # driver check
        driver: str = arg.param[0]
        driver_module = importlib.import_module("benchgc.arg.%s" % driver)
        return driver_module.compare(ref, res, verbose)
    else:
        raise Exception("invalid compare type or compare parameter")

def iterate_tensor(tensor: torch.Tensor, fn: Callable[[Tuple[int, ...]], None]):
    index: List[int] = [0] * tensor.ndim

    def dfs(depth: int):
        if depth == tensor.ndim:
            fn(tuple(index))
        else:
            for i in range(tensor.shape[depth]):
                index[depth] = i
                dfs(depth + 1)
    dfs(0)

def norm(threshold: float, ref: torch.Tensor, res: torch.Tensor, verbose: int) -> Tuple[bool, bool | None]:

    f32_ref = ref.to(torch.float32)
    f32_res = res.to(torch.float32)
    if f32_ref.nelement() == 0:
        return (True, None)

    diff_square_sum = torch.square(torch.subtract(f32_ref, f32_res)).sum()
    square_sum = torch.square(f32_ref).sum()

    l2_diff_norm = torch.sqrt(diff_square_sum / square_sum).item()
    if verbose >= benchgc.util.COMPARE_VERBOSE:
        print(
            "norm check: %.10f / threshold: %.10f" % (l2_diff_norm, threshold)
        )

    return (l2_diff_norm < threshold, None)


def p2p(threshold: float, zero_percent: float, ref: torch.Tensor, res: torch.Tensor, verbose: int) -> Tuple[bool, bool | None]:

    if verbose >= benchgc.util.COMPARE_VERBOSE:
        print("p2p check: threshold: %.7f" % threshold)
    f32_ref = ref.to(torch.float32)
    f32_res = res.to(torch.float32)

    check = torch.BoolTensor([False])

    check = check.bitwise_or(torch.bitwise_and(f32_ref.isnan(), f32_res.isnan()))
    check = check.bitwise_or(
        torch.bitwise_and(f32_ref.isneginf(), f32_res.isneginf())
    )
    check = check.bitwise_or(
        torch.bitwise_and(f32_ref.isposinf(), f32_res.isposinf())
    )

    # choose diff/rel_diff based on value
    abs_diff = (f32_ref - f32_res).abs()
    rel_diff = abs_diff / torch.where(
        f32_ref.abs() > numpy.finfo(numpy.float32).smallest_subnormal,
        f32_ref.abs(),
        1,
    )
    # pick a diff for comparison
    diff = torch.where(f32_ref.abs() > 1e-5, rel_diff, abs_diff)

    check = check.bitwise_or(diff <= threshold)

    if verbose >= benchgc.util.OUTPUT_VERBOSE:
        iterate_tensor(
            check,
            lambda idx: print(
                "%20s: ref: %12.7f res: %12.7f abs_diff: %12.7f rel_diff: %12.7f"
                % (
                    idx,
                    f32_ref[idx].item(),
                    f32_res[idx].item(),
                    abs_diff[idx].item(),
                    rel_diff[idx].item(),
                )
            ),
        )
    if check.all():
        # check mistrusted
        zero = res.nelement() - res.count_nonzero().item()
        if res.nelement() < 10:
            mistrust = False
        else:
            mistrust = zero * 100.0 / res.nelement() > zero_percent
        return (True, mistrust)
    else:
        if verbose < benchgc.util.OUTPUT_VERBOSE:  # skip verbose print if full output tensor is alrady printed
            fail = torch.argwhere(torch.where(check, 0, 1))
            if verbose < benchgc.util.ERROR_OUTPUT_VERBOSE:
                # only print top 10 failed data points if verbose level does not satisfied
                fail = fail[:10]  
            for idx in fail:
                index: Tuple[int, ...] = tuple(idx.tolist())
                print(
                    "%20s: ref: %12.7f res: %12.7f abs_diff: %12.7f rel_diff: %12.7f"
                    % (
                        index,
                        f32_ref[index].item(),
                        f32_res[index].item(),
                        abs_diff[index].item(),
                        rel_diff[index].item(),
                    )
                )
        return (False, None)