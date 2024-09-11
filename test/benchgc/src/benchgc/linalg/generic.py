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

from typing import Any, Dict, List, Tuple

import benchgc.runner
import benchgc.util
import torch
from benchgc.mlir.util import MLIRCache
from gc_mlir import ir


def generic_loop(
    cache: MLIRCache,
    op: ir.OpView,
    depth: int,
    iterspace: Dict[str, Tuple[int, int, int]],
    affine_from: List[str],
    affine_to: List[List[str]],
    var: Dict[str, torch.Tensor],
    loop_var: Dict[str, torch.Tensor],
    result_tensors: Tuple[torch.Tensor, ...],
):
    if depth == len(affine_from):
        # we need to execute the block here
        # we will need to read the block argument name and save it into the cache

        if len(cache.next) == 0:
            # region cache
            cache.next.append(MLIRCache())

        block: ir.Block = op.regions[0].blocks[0]
        if len(cache.next[0].next) == 0:
            # region->block cache
            cache.next[0].next.append(MLIRCache())
            for arg in block.arguments:
                cache.next[0].next[0].arg.append(arg.get_name())

        block_cache = cache.next[0].next[0]
        block_arg: Dict[str, torch.Tensor] = {}
        for i in range(len(block.arguments)):
            index: Tuple[int, ...] = tuple()
            aff: List[str] = affine_to[i]
            for d in aff:
                index = index + (int(loop_var[d].item()),)

            if i + len(op.results) < len(op.regions[0].blocks[0].arguments):
                # input argument
                block_arg[block_cache.arg[i]] = var[cache.opr[i]][index]
            else:
                # output argument
                block_arg[block_cache.arg[i]] = result_tensors[
                    i + len(op.results) - len(block.arguments)
                ][index]

        res: Tuple[Any, ...] = benchgc.runner.dfs_block(
            cache.next[0].next[0], block, var | loop_var | block_arg
        )

        # perform the yield operation
        for i in range(len(op.results)):
            idx = -1 - i
            aff: List[str] = affine_to[idx]
            index: Tuple[int, ...] = tuple()
            for d in aff:
                index = index + (int(loop_var[d].item()),)
            result_tensors[idx][index] = res[idx]
    else:
        it = iterspace[affine_from[depth]]
        for i in range(it[0], it[1], it[2]):
            loop_var[affine_from[depth]][0] = i
            generic_loop(
                cache,
                op,
                depth + 1,
                iterspace,
                affine_from,
                affine_to,
                var,
                loop_var,
                result_tensors,
            )


def ref_generic(
    cache: MLIRCache, op: ir.OpView, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    affine_from: List[str] = []
    affine_to: List[List[str]] = []

    for affine in op.attributes["indexing_maps"]:
        aff = str(affine)
        affine_from = aff[aff.find("<(") + 2 : aff.find(") ->")].split(", ")
        affine_to.append(aff[aff.find("-> (") + 4 : aff.find(")>")].split(", "))

    # try to find the iteration space
    # TODO: support affine expression

    iterspace: Dict[str, Tuple[int, int, int]] = {}
    operands: List[ir.OpOperand] = list(op.operands)

    loop_var: Dict[str, torch.Tensor] = {}
    for d in affine_from:
        iterspace[d] = (0, 0, 1)
        loop_var[d] = torch.zeros(size=[1], dtype=torch.int)

    for i in range(len(operands)):
        for j in range(len(operands[i].type.shape)):
            iterspace[affine_to[i][j]] = (0, operands[i].type.shape[j], 1)

    result_tensors: Tuple[torch.Tensor, ...] = tuple()
    # create the buffer for result tensors
    for i in range(len(op.results)):
        result_tensors = result_tensors + (tensors[cache.opr[-1 - i]].clone(),)

    generic_loop(
        cache,
        op,
        0,
        iterspace,
        affine_from,
        affine_to,
        tensors,
        loop_var,
        result_tensors,
    )
    return result_tensors


