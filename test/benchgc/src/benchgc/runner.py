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

import gc_mlir._mlir_libs
import gc_mlir.ir
import torch
from benchgc.mlir.util import MLIRCache
from typing import Dict
from benchgc.linalg import ref_op as linalg_ref_op
from benchgc.tensor import ref_op as tensor_ref_op
from benchgc.arith import ref_op as arith_ref_op

from typing import Tuple


def dfs_op(cache: MLIRCache, op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]):

    dialect_call: str = str(op.name)
    if dialect_call == "func.return":
        ret: Tuple[torch.Tensor, ...] = tuple()
        for name in cache.opr:
            ret = ret + (tensors[name],)
        return ret
    if dialect_call.startswith("linalg"):
        ref_op = linalg_ref_op
    elif dialect_call.startswith("tensor"):
        ref_op = tensor_ref_op
    elif dialect_call.startswith("arith"):
        ref_op = arith_ref_op
    else:
        build_cache = len(cache.next) == 0
        for i in range(len(op.regions)):
            if build_cache:
                # we do not need to cache things for region
                # keep an empty cache
                cache.next.append(MLIRCache())
            ret = dfs_region(cache.next[i], op.regions[i], tensors)
            if ret is not None:
                return ret
        return

    dialect_op: str = dialect_call.split(".")[1]
    if dialect_op not in ref_op:
        raise Exception("unknown op call %s" % dialect_call)
    ref_func = ref_op[dialect_op]
    # yield op may return value
    return ref_func(cache, op, tensors)


def dfs_region(
    cache: MLIRCache, region: gc_mlir.ir.Region, tensors: Dict[str, torch.Tensor]
):
    build_cache = len(cache.next) == 0
    for i in range(len(region.blocks)):
        if build_cache:
            _cache = MLIRCache()
            # we need to cache argument name for block object
            for arg in region.blocks[i].arguments:
                _cache.arg.append(arg.get_name())
            cache.next.append(_cache)
        ret = dfs_block(cache.next[i], region.blocks[i], tensors)
        if ret is not None:
            return ret


def dfs_block(
    cache: MLIRCache, block: gc_mlir.ir.Block, tensors: Dict[str, torch.Tensor]
):
    build_cache = len(cache.next) == 0
    for i in range(len(block.operations)):
        if build_cache:
            _cache = MLIRCache()
            # we need to cache operand name and result name
            for opr in block.operations[i].operands:
                _cache.opr.append(opr.get_name())

            for res in block.operations[i].results:
                _cache.res.append(res.get_name())
            cache.next.append(_cache)

        ret = dfs_op(cache.next[i], block.operations[i], tensors)
        if ret is not None:
            return ret


def ref_run(
    module: gc_mlir.ir.Module, tensors: Dict[str, torch.Tensor], entry: str = '"entry"'
):
    entry_op: gc_mlir.ir.OpView | None = None
    for op in module.operation.opview.regions[0].blocks[0].operations:
        if str(op.name) == entry:
            entry_op = op
            break

    # cache some information of block & op

    cache = MLIRCache()

    if entry_op is None:
        raise Exception("entry function %s is not found at the top level" % entry)
    else:
        ret = dfs_op(cache, entry_op, tensors)
        if ret is not None:
            return ret
