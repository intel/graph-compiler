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

from typing import Callable, List, Tuple

import gc_mlir.dialects.tensor
import gc_mlir.ir
from benchgc.mlir.arg import MLIRArg
from gc_mlir.dialects import func


def init_module(
    inputs: Tuple[MLIRArg, ...],
    outputs: Tuple[MLIRArg, ...],
    op_func: Callable[
        [gc_mlir.ir.Context, Tuple[gc_mlir.ir.BlockArgument, ...]],
        List[gc_mlir.ir.OpResult],
    ],
) -> gc_mlir.ir.Module:
    with gc_mlir.ir.Context() as ctx, gc_mlir.ir.Location.unknown():
        module = gc_mlir.ir.Module.create()
        with gc_mlir.ir.InsertionPoint(module.body):
            f = func.FuncOp(
                name="entry",
                type=gc_mlir.ir.FunctionType.get(
                    inputs=[x.get_mlir_type(ctx) for x in inputs],
                    results=[x.get_mlir_type(ctx) for x in outputs],
                ),
            )
            f.attributes["llvm.emit_c_interface"] = gc_mlir.ir.UnitAttr.get()

            with gc_mlir.ir.InsertionPoint(f.add_entry_block()):
                block_args = f.entry_block.arguments
                func.ReturnOp(op_func(ctx, *block_args))
        return module
