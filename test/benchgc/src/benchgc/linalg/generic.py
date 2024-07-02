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
import gc_mlir.ir
import benchgc.runner
import benchgc.util

from benchgc.mlir import escape_var

from typing import Dict, List, Tuple, Any

def generic_loop(op: gc_mlir.ir.OpView, depth: int, iterspace: Dict[str, Tuple[int, int, int]], affine_from: List[str], affine_to: List[List[str]], var: Dict[str, torch.Tensor], loop_var: Dict[str, torch.Tensor]):
    if depth == len(affine_from):
        block_arg: Dict[str, torch.Tensor] = {}
        for i in range(len(op.regions[0].blocks[0].arguments)):
            arg: gc_mlir.ir.BlockArgument = op.regions[0].blocks[0].arguments[i]
            index: Tuple[int,...] = tuple()
            aff: List[str] = affine_to[i]
            for d in aff:
                index = index + (int(loop_var[d].item()), )
            operand: gc_mlir.ir.OpOperand = op.operands[i]
            block_arg[escape_var(arg.get_name())] = var[escape_var(operand.get_name())][index]
        print(block_arg)
        res: Tuple[Any,...] = benchgc.runner.dfs_block(op.regions[0].blocks[0], var | loop_var | block_arg)

        print("here")
        print(res)
        # perform the yield operation
        for i in range(len(op.results)):
            idx = -1 - i
            aff: List[str] = affine_to[idx]
            index: Tuple[int,...] = tuple()
            for d in aff:
                index = index + (int(loop_var[d].item()), )
            var[escape_var(op.results[idx].get_name())][index] = res[idx]
            # FIXME: this will violate SSA rule
            var[escape_var(op.operands[idx].get_name())][index] = res[idx]

    else:
        it = iterspace[affine_from[depth]]
        for i in range(it[0], it[1], it[2]):
            loop_var[affine_from[depth]][0] = i
            generic_loop(op, depth + 1, iterspace, affine_from, affine_to, var, loop_var)
    
def ref_generic(op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]):
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
        print(operands[i].type.shape)
        for j in range(len(operands[i].type.shape)):
            iterspace[affine_to[i][j]] = (0, operands[i].type.shape[j], 1)
    # create the buffer for result tensors

    for i in range(len(op.results)):
        idx = -1 - i
        res = op.results[idx]
        operand: gc_mlir.ir.OpOperand = op.operands[idx]
        tensors[escape_var(res.get_name())] = torch.zeros(size = operand.type.shape, dtype= benchgc.util.get_dtype(str(operand.type.element_type)))

    generic_loop(op, 0, iterspace, affine_from, affine_to, tensors, loop_var)
    return

def ref_yield(op: gc_mlir.ir.OpView, tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor,...]:
    ret: Tuple[torch.Tensor,...] = tuple()
    for operand in op.operands:
        ret = ret + (tensors[escape_var(operand.get_name())],)
    return ret