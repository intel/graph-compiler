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
from benchgc.mlir import MLIRCache
import gc_mlir.ir
import benchgc.runner
import benchgc.util

from typing import Dict, List, Tuple, Any

def generic_loop(cache: MLIRCache, op: gc_mlir.ir.OpView, depth: int, iterspace: Dict[str, Tuple[int, int, int]], affine_from: List[str], affine_to: List[List[str]], var: Dict[str, torch.Tensor], loop_var: Dict[str, torch.Tensor]):
    if depth == len(affine_from):
        # we need to execute the block here
        # we will need to read the block argument name and save it into the cache

        if len(cache.next) == 0:
            # region cache
            cache.next.append(MLIRCache())
        if len(cache.next[0].next) == 0:
            # region->block cache
            cache.next[0].next.append(MLIRCache())
            block: gc_mlir.ir.Block = op.regions[0].blocks[0]
            for arg in block.arguments:
                cache.next[0].next[0].arg.append(arg.get_name())

        block_cache = cache.next[0].next[0]
        block_arg: Dict[str, torch.Tensor] = {}
        for i in range(len(block.arguments)):
            index: Tuple[int,...] = tuple()
            aff: List[str] = affine_to[i]
            for d in aff:
                index = index + (int(loop_var[d].item()), )

            if i + len(op.results) < len(op.regions[0].blocks[0].arguments):
                # input argument
                block_arg[block_cache.arg[i]] = var[cache.opr[i]][index]
            else:
                # output argument
                block_arg[block_cache.arg[i]] = var[cache.res[i + len(op.results) - len(block.arguments)]][index]

        res: Tuple[Any,...] = benchgc.runner.dfs_block(cache.next[0].next[0], block, var | loop_var | block_arg)

        # perform the yield operation
        for i in range(len(op.results)):
            idx = -1 - i
            aff: List[str] = affine_to[idx]
            index: Tuple[int,...] = tuple()
            for d in aff:
                index = index + (int(loop_var[d].item()), )
            var[cache.res[idx]][index] = res[idx]

    else:
        it = iterspace[affine_from[depth]]
        for i in range(it[0], it[1], it[2]):
            loop_var[affine_from[depth]][0] = i
            generic_loop(cache, op, depth + 1, iterspace, affine_from, affine_to, var, loop_var)
    
def ref_generic(cache: MLIRCache, op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]):
    affine_from: List[str] = []
    affine_to: List[List[str]] = []

    for affine in op.attributes["indexing_maps"]:
        aff = str(affine)
        affine_from = aff[aff.find("<(") + 2: aff.find(") ->")].split(", ")
        affine_to.append(aff[aff.find("-> (") + 4: aff.find(")>")].split(", "))

    # try to find the iteration space
    # TODO: support affine expression

    iterspace:  Dict[str, Tuple[int, int, int]] = {}
    operands: List[gc_mlir.ir.OpOperand] = list(op.operands)

    loop_var: Dict[str, torch.Tensor] = {}
    for d in affine_from:
        iterspace[d] = (0, 0, 1)
        loop_var[d] = torch.zeros(size=[1], dtype=torch.int)

    for i in range(len(operands)):
        for j in range(len(operands[i].type.shape)):
            iterspace[affine_to[i][j]] = (0, operands[i].type.shape[j], 1)

    # create the buffer for result tensors
    for i in range(len(op.results)):
        idx = -1 - i
        tensors[cache.res[idx]] = tensors[cache.opr[idx]].clone()

    generic_loop(cache, op, 0, iterspace, affine_from, affine_to, tensors, loop_var)
    return



def reduce_loop(cache: MLIRCache, op: gc_mlir.ir.OpView, depth: int, in_shape: List[int], var: Dict[str, torch.Tensor], in_idx: List[int], out_idx: List[int], reduced_axis: int):
    if depth == len(in_shape):
        # we need to execute the block here
        # we will need to read the block argument name and save it into the cache

        block: gc_mlir.ir.Block = op.regions[0].blocks[0]

        if len(cache.next) == 0:
            # region cache
            cache.next.append(MLIRCache())
        if len(cache.next[0].next) == 0:
            # region->block cache
            cache.next[0].next.append(MLIRCache())
            for arg in block.arguments:
                cache.next[0].next[0].arg.append(arg.get_name())

        block_arg: Dict[str, torch.Tensor] = {
            # set input
            cache.next[0].next[0].arg[0]: var[cache.opr[0]][tuple(in_idx)],
            # set output 
            cache.next[0].next[0].arg[1]: var[cache.res[0]][tuple(out_idx)],
        }

        res: Tuple[Any,...] = benchgc.runner.dfs_block(cache.next[0].next[0], op.regions[0].blocks[0], var | block_arg)

        # perform the yield operation
        var[cache.res[0]][tuple(out_idx)] = res[0]
    else:
        dimensions: gc_mlir.ir.DenseI64ArrayAttr = op.attributes["dimensions"]
        reduce_axis: bool = list(dimensions).count(depth) > 0

        for i in range(in_shape[depth]):
            if reduce_axis:
                in_idx[depth] = i
                reduce_loop(cache, op, depth + 1, in_shape, var, in_idx, out_idx, reduced_axis + 1)
            else:
                in_idx[depth] = i
                out_idx[depth - reduce_axis] = i
                reduce_loop(cache, op, depth + 1, in_shape, var, in_idx, out_idx, reduced_axis)

def ref_reduce(cache: MLIRCache, op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]):
    # create the buffer for result tensors
    tensors[cache.res[0]] = tensors[cache.opr[-1]].clone()
    in_shape: List[int] = list(op.operands[0].type.shape)
    out_shape: List[int] = list(op.result.type.shape)

    reduce_loop(cache, op, 0, in_shape, tensors, [0] * len(in_shape), [0] * len(out_shape), 0)
    return

def ref_yield(cache: MLIRCache, op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor,...]:
    ret: Tuple[torch.Tensor,...] = tuple()
    for i in range(len(op.operands)):
        ret = ret + (tensors[cache.opr[i]],)
    return ret