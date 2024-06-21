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
from benchgc.onednn_graph.mlir import init_i2o1_module
from typing import Dict, Tuple

from gc_mlir._mlir_libs._mlir.ir import BoolAttr
from gc_mlir.dialects import onednn_graph
import gc_mlir.ir


def __ref_init(
    op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    auto_broadcast: bool = True
    for attribute in op.attributes:
        if attribute.name == "auto_broadcast" and isinstance(attribute.attr, BoolAttr):
            auto_broadcast = attribute.attr.__bool__()

    src0 = var[op.operands[0].get_name().removeprefix("%").removeprefix("@")]
    src1 = var[op.operands[1].get_name().removeprefix("%").removeprefix("@")]
    if not auto_broadcast and src0.shape != src1.shape:
        raise Exception("shape mismatch %s and %s" % (src0.shape, src1.shape))

    dst_var: str = op.results[0].get_name().removeprefix("%").removeprefix("@")
    return (src0, src1, dst_var)


def map_binary_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src0", "arg1": "src1", "0": "dst"}.items():
        args[k] = args[v]
        del args[v]


def ref_add(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src0, src1, dst_var = __ref_init(op, var)
    var[dst_var] = torch.add(src0, src1)


def mlir_add(flags: argparse.Namespace, args: Dict[str, Arg]) -> gc_mlir.ir.Module:
    map_binary_args(args)
    return init_i2o1_module(
        flags,
        args,
        lambda arg0, arg1: onednn_graph.AddOp(
            arg0, arg1, auto_broadcast=(flags.auto_broadcast == "numpy")
        ).result,
    )
