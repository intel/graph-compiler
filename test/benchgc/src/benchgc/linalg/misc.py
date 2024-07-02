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

from benchgc.mlir import init_i1o1_module, escape_var
from gc_mlir._mlir_libs._mlir.ir import DenseI64ArrayAttr
from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, Tuple, List

def __ref_init(
    op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, str]:

    src = var[escape_var(op.operands[0].get_name())]
    dst_var: str = escape_var(op.results[0].get_name())
    return (src, dst_var)

def map_misc_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src", "1": "dst"}.items():
        args[k] = args[v]
        del args[v]


# 1. use to reshape to match ndim
# 2. perform broadcast
def ref_broadcast(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, dst_var = __ref_init(op, var)

    dst_shape: List[int] = op.results[0].type.shape
    tmp_shape = copy.copy(dst_shape)
    dimensions: DenseI64ArrayAttr = op.attributes["dimensions"]
    for d in dimensions:
        tmp_shape[d] = 1

    var[dst_var] = src.reshape(tmp_shape).broadcast_to(dst_shape)

def mlir_broadcast(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:

    for k, v in {"arg0": "src", "broadcasted": "dst"}.items():
        args[k] = args[v]
        del args[v]

    return init_i1o1_module(args["arg0"], args["broadcasted"], lambda ctx, arg0: linalg.broadcast(arg0, outs=[args["broadcasted"].get_empty_op(ctx)], dimensions= flags.dimensions))


