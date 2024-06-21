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

import gc_mlir.ir
import torch
from typing import Dict


def dfs_op(op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]):
    dialect_call: str = str(op.name)
    if dialect_call.startswith("onednn_graph"):
        from benchgc.onednn_graph import ref_op
    elif dialect_call.startswith("linalg"):
        from benchgc.linalg import ref_op
    else:
        for region in op.regions:
            dfs_region(region, tensors)
        return

    dialect_op: str = dialect_call.split(".")[1]
    if dialect_op not in ref_op:
        raise Exception("unknown op call %s" % dialect_call)
    ref_func = ref_op[dialect_op]
    ref_func(op, tensors)


def dfs_region(region: gc_mlir.ir.Region, tensors: Dict[str, torch.Tensor]):
    for block in region.blocks:
        dfs_block(block, tensors)


def dfs_block(block: gc_mlir.ir.Block, tensors: Dict[str, torch.Tensor]):
    for op in block.operations:
        dfs_op(op, tensors)


def ref_run(
    module: gc_mlir.ir.Module, tensors: Dict[str, torch.Tensor], entry: str = '"entry"'
):
    entry_op: gc_mlir.ir.OpView | None = None
    for op in module.operation.opview.regions[0].blocks[0].operations:
        if str(op.name) == entry:
            entry_op = op
            break
    if entry_op is None:
        raise Exception("entry function %s is not found at the top level" % entry)
    else:
        dfs_op(entry_op, tensors)
