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
import copy
from typing import Dict, List, Tuple

import benchgc.util
import torch
from benchgc.arg import Arg
from benchgc.mlir.module import init_module
from benchgc.mlir.util import MLIRCache
from gc_mlir import ir
from gc_mlir.dialects import linalg
from gc_mlir.dialects.linalg.opdsl.lang.comprehension import TypeFnType


# 1. use to reshape to match ndim
# 2. perform broadcast
def ref_broadcast(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    dst_shape: List[int] = op.results[0].type.shape
    tmp_shape = copy.copy(dst_shape)
    dimensions: ir.DenseI64ArrayAttr = op.attributes["dimensions"]
    for d in dimensions:
        tmp_shape[d] = 1

    return (var[cache.opr[0]].reshape(tmp_shape).broadcast_to(dst_shape).contiguous(),)


def mlir_broadcast(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:

    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [
            linalg.broadcast(
                arg0, outs=[args[1].get_zero_op(ctx)], dimensions=flags.dimensions
            )
        ],
    )


def ref_fill(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.full(tuple(op.results[0].type.shape), var[cache.opr[0]]),)


def mlir_fill(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [
            linalg.fill(
                arg0, outs=[args[1].get_zero_op(ctx)], dimensions=flags.dimensions
            )
        ],
    )


def ref_copy(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        var[cache.opr[0]]
        .to(benchgc.util.get_dtype(str(op.result.type.element_type)))
        .clone(),
    )


def mlir_copy(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:

    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [
            linalg.copy(
                arg0, outs=[args[1].get_zero_op(ctx)], cast=TypeFnType(flags.cast)
            )
        ],
    )
