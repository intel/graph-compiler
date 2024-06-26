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

from benchgc.linalg.mlir import init_i1o1_module, escape_var

from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, Tuple

def __ref_init(
    op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, str]:

    src = var[escape_var(op.operands[0].get_name())]
    dst_var: str = escape_var(op.results[0].get_name())
    return (src, dst_var)

def map_eltwise_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src", "1": "dst"}.items():
        args[k] = args[v]
        del args[v]

def ref_abs(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, dst_var = __ref_init(op, var)
    var[dst_var] = torch.abs(src)


def mlir_abs(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_eltwise_args(args)
    return init_i1o1_module(args["arg0"], args["1"], lambda ctx, arg0: linalg.abs(arg0, outs=[args["1"].get_empty_op(ctx)]))

def ref_ceil(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, dst_var = __ref_init(op, var)
    var[dst_var] = torch.ceil(src)


def mlir_ceil(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_eltwise_args(args)
    return init_i1o1_module(args["arg0"], args["1"], lambda ctx, arg0: linalg.ceil(arg0, outs=[args["1"].get_empty_op(ctx)]))




