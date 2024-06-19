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
from typing import Dict, Tuple, Callable, Any

from gc_mlir._mlir_libs._mlir.ir import BoolAttr
from gc_mlir.dialects import onednn_graph, func
import gc_mlir.ir

def __ref_init(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, str]:
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

def __mlir_init(flags: argparse.Namespace, args: Dict[str, Arg], op_func: Callable[[Any, Any], Any]) -> gc_mlir.ir.Module:

    args["arg0"] = args["src0"]
    args["arg1"] = args["src1"]
    args["0"] = args["dst"]
    del args["src0"]
    del args["src1"]
    del args["dst"]

    with gc_mlir.ir.Context() as ctx, gc_mlir.ir.Location.unknown():
        module = gc_mlir.ir.Module.create()
        with gc_mlir.ir.InsertionPoint(module.body):
            f = func.FuncOp(name = "entry", type = gc_mlir.ir.FunctionType.get(
                inputs = [args["arg0"].get_ranked_tensor_type(ctx), args["arg1"].get_ranked_tensor_type(ctx)],
                results = [args["0"].get_ranked_tensor_type(ctx)]))

            with gc_mlir.ir.InsertionPoint(f.add_entry_block()):
                arg_list: gc_mlir.ir.BlockArgumentList = f.entry_block.arguments
                src0: gc_mlir.ir.BlockArgument = arg_list[0]
                src1: gc_mlir.ir.BlockArgument = arg_list[1]
                dst: gc_mlir.ir.OpResult = op_func(src0, src1, auto_broadcast = (flags.auto_broadcast == "numpy")).result
                func.ReturnOp([dst])
        return module

def ref_add(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src0, src1, dst_var = __ref_init(op, var) 
    var[dst_var] = torch.add(src0, src1)

def mlir_add(flags: argparse.Namespace,args: Dict[str, Arg]) -> gc_mlir.ir.Module:
    return __mlir_init(flags, args, onednn_graph.AddOp)
