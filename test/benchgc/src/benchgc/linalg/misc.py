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
import gc_mlir.ir
import copy

import benchgc.util
from benchgc.mlir import MLIRCache, init_i1o1_module
from gc_mlir._mlir_libs._mlir.ir import DenseI64ArrayAttr
from gc_mlir.dialects import linalg
from gc_mlir.dialects.linalg.opdsl.lang.comprehension import TypeFnType

from benchgc.arg import Arg
from typing import Dict, List

# 1. use to reshape to match ndim
# 2. perform broadcast
def ref_broadcast(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    dst_shape: List[int] = op.results[0].type.shape
    tmp_shape = copy.copy(dst_shape)
    dimensions: DenseI64ArrayAttr = op.attributes["dimensions"]
    for d in dimensions:
        tmp_shape[d] = 1

    var[cache.res[0]] = var[cache.opr[0]].reshape(tmp_shape).broadcast_to(dst_shape)

def mlir_broadcast(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:

    return init_i1o1_module(ins[0], outs[0], lambda ctx, arg0: linalg.broadcast(arg0, outs=[outs[0].get_empty_op(ctx)], dimensions= flags.dimensions))

def ref_fill(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.full(tuple(op.results[0].type.shape), var[cache.opr[0]])

def mlir_fill(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i1o1_module(ins[0], outs[0], lambda ctx, arg0: linalg.fill(arg0, outs=[outs[0].get_empty_op(ctx)], dimensions= flags.dimensions))

def ref_copy(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] =  var[cache.opr[0]].to(benchgc.util.get_dtype(str(op.result.type.element_type))).clone()

def mlir_copy(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:

    return init_i1o1_module(ins[0], outs[0], lambda ctx, arg0: linalg.copy(arg0, outs=[outs[0].get_empty_op(ctx)], cast = TypeFnType(flags.cast)))


