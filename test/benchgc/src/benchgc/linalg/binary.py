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

import torch
import argparse

from benchgc.arg import Arg
from benchgc.mlir import MLIRCache, init_i2o1_module
from typing import Dict, List

from gc_mlir.dialects import linalg
import gc_mlir.ir


def ref_add(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.add(var[cache.opr[0]], var[cache.opr[1]])


def mlir_add(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i2o1_module(
        ins[0],
        ins[1],
        outs[0],
        lambda ctx, arg0, arg1: linalg.add(
            arg0, arg1, outs=[outs[0].get_empty_op(ctx)]
        ),
    )


def ref_powf(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.pow(var[cache.opr[0]], var[cache.opr[1]])


def mlir_powf(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    # This is a workaround fix.
    # see https://github.com/llvm/llvm-project/blob/a355c2d07464f020c9a66cbd6189c22a42c2be2e/mlir/python/mlir/dialects/linalg/opdsl/lang/dsl.py#L140-L142
    # python binding will deduce cpp_class_name based on the function
    # powf will be translated into PowfOp, which is not equal to PowFOp.
    linalg.powf.op_def.metadata.cpp_class_name = "PowFOp"
    return init_i2o1_module(
        ins[0],
        ins[1],
        outs[0],
        lambda ctx, arg0, arg1: linalg.powf(
            arg0, arg1, outs=[outs[0].get_empty_op(ctx)]
        ),
    )


def ref_div(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.div(var[cache.opr[0]], var[cache.opr[1]])


def mlir_div(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i2o1_module(
        ins[0],
        ins[1],
        outs[0],
        lambda ctx, arg0, arg1: linalg.div(
            arg0, arg1, outs=[outs[0].get_empty_op(ctx)]
        ),
    )


def ref_mul(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.mul(var[cache.opr[0]], var[cache.opr[1]])


def mlir_mul(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i2o1_module(
        ins[0],
        ins[1],
        outs[0],
        lambda ctx, arg0, arg1: linalg.mul(
            arg0, arg1, outs=[outs[0].get_empty_op(ctx)]
        ),
    )
