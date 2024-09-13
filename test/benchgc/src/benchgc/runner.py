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

from typing import Dict, Tuple

import gc_mlir._mlir_libs
import gc_mlir.dialects
import gc_mlir.dialects.func
import torch
from benchgc.arith import ref_op as arith_ref_op
from benchgc.linalg import ref_op as linalg_ref_op
from benchgc.math import ref_op as math_ref_op
from benchgc.mlir.util import MLIRCache
from benchgc.tensor import ref_op as tensor_ref_op
from gc_mlir import ir


def dfs_op(
    cache: MLIRCache, op: ir.OpView, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:

    dialect_call: str = str(op.name)
    if dialect_call in ["func.return", "linalg.yield"]:
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
    elif dialect_call.startswith("math"):
        ref_op = math_ref_op
    else:
        build_cache = len(cache.next) == 0
        for i in range(len(op.regions)):
            if build_cache:
                # we do not need to cache things for region
                # keep an empty cache
                cache.next.append(MLIRCache())
            ret = dfs_region(cache.next[i], op.regions[i], tensors)
            if len(ret) != 0:
                return ret
        return tuple()

    dialect_op: str = dialect_call.split(".")[1]
    if dialect_op not in ref_op:
        raise Exception(f"unknown op call {dialect_call}")
    ref_func = ref_op[dialect_op]
    for i, res in enumerate(ref_func(cache, op, tensors)):
        tensors[cache.res[i]] = res
    return tuple()


def dfs_region(
    cache: MLIRCache, region: ir.Region, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    build_cache = len(cache.next) == 0
    for i in range(len(region.blocks)):
        if build_cache:
            _cache = MLIRCache()
            # we need to cache argument name for block object
            for arg in region.blocks[i].arguments:
                _cache.arg.append(arg.get_name())
            cache.next.append(_cache)
        ret = dfs_block(cache.next[i], region.blocks[i], tensors)
        if len(ret) != 0:
            return ret
    return tuple()


def dfs_block(
    cache: MLIRCache, block: ir.Block, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
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
        if len(ret) != 0:
            return ret
    return tuple()


def ref_run(
    entry: gc_mlir.dialects.func.FuncOp, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # cache some information of block & op
    return dfs_op(MLIRCache(), entry, tensors)
