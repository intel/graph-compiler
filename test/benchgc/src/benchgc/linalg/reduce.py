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

import argparse
from typing import Callable, Dict, List, Tuple

import benchgc.runner
import benchgc.util
import gc_mlir.ir as ir
import torch
from benchgc.arg.arg import Arg
from benchgc.mlir.module import init_reduce_module
from benchgc.mlir.util import MLIRCache
from gc_mlir.dialects import arith, math


def reduce_loop(
    cache: MLIRCache,
    op: ir.OpView,
    depth: int,
    in_shape: List[int],
    var: Dict[str, torch.Tensor],
    in_idx: List[int],
    out_idx: List[int],
    reduced_axis: int,
    result_tensor: torch.Tensor,
):
    if depth == len(in_shape):
        # we need to execute the block here
        # we will need to read the block argument name and save it into the cache

        block: ir.Block = op.regions[0].blocks[0]

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
            cache.next[0].next[0].arg[1]: result_tensor[tuple(out_idx)],
        }

        res: Tuple[torch.Tensor, ...] = benchgc.runner.dfs_block(
            cache.next[0].next[0], op.regions[0].blocks[0], var | block_arg
        )

        # perform the yield operation
        result_tensor[tuple(out_idx)] = res[0]
    else:
        dimensions: ir.DenseI64ArrayAttr = op.attributes["dimensions"]
        reduce_axis: bool = depth in list(dimensions)

        for i in range(in_shape[depth]):
            if reduce_axis:
                in_idx[depth] = i
                reduce_loop(
                    cache,
                    op,
                    depth + 1,
                    in_shape,
                    var,
                    in_idx,
                    out_idx,
                    reduced_axis + 1,
                    result_tensor,
                )
            else:
                in_idx[depth] = i
                out_idx[depth - reduced_axis] = i
                reduce_loop(
                    cache,
                    op,
                    depth + 1,
                    in_shape,
                    var,
                    in_idx,
                    out_idx,
                    reduced_axis,
                    result_tensor,
                )


case_to_ops: Dict[str, List[str]] = {
    "reduce.add": ["arith.addf", "arith.addi"],
    "reduce.max": ["arith.maximumf", "arith.maxsi"],
    "reduce.min": ["arith.minimumf", "arith.minsi"],
    "reduce.mul": ["arith.mulf", "arith.muli"],
}


def check_commutative_operand(opr0: str, opr1: str, arg0: str, arg1: str) -> bool:
    return opr0 == arg0 and opr1 == arg1 or opr0 == arg1 and opr1 == arg0


# do not execute the block step by step if the pattern matches
def match_reduce_type(op: ir.OpView) -> str:
    block = op.regions[0].blocks[0]
    block_arg = block.arguments
    arg0, arg1 = block_arg[0].get_name(), block_arg[1].get_name()
    block_op = block.operations
    if len(block_op) == 2 and len(block_op[0].operands) == 2:
        #   need to match the following pattern
        #      (%in: dt, %out: dt) {
        #        %0 = opcall %out, %in: dt / %in, %out
        #        linalg.yield %0: dt
        #      }
        op0, op1 = block_op[0], block_op[1]
        opr0, opr1 = op0.operands[0].get_name(), op0.operands[1].get_name()

        if (
            op1.name == "linalg.yield"
            and op1.operands[0].get_name() == op0.result.get_name()
            and check_commutative_operand(opr0, opr1, arg0, arg1)
        ):
            for case, ops in case_to_ops.items():
                if op0.name in ops:
                    return case
    elif len(block_op) == 3 and len(block_op[1].operands) == 2:
        op0, op1, op2 = block_op[0], block_op[1], block_op[2]
        opr0, opr1 = op1.operands[0].get_name(), op1.operands[1].get_name()
        #   match complex reduce here

        #   reduce L1
        #      (%in: dt, %out: dt) {
        #        %0 = math.absi/absf %in
        #        %1 = arith.addf/addi %out, %0 / %0, %out
        #        linalg.yield %1: dt
        #      }
        if (
            op0.name in ["math.absf", "math.absi"]
            and op0.operands[0].get_name() == arg0
            and op1.name in ["arith.addf", "arith.addi"]
            and check_commutative_operand(opr0, opr1, op0.result.get_name(), arg1)
            and op2.name == "linalg.yield"
            and op2.operands[0].get_name() == op1.result.get_name()
        ):
            return "reduce.l1"

        #   reduce square L2
        #      (%in: dt, %out: dt) {
        #        %0 = arith.muli/mulf %in, %in
        #        %1 = arith.addf/addi %out, %0 / %0, %out
        #        linalg.yield %1: dt
        #      }

        if (
            op0.name in ["arith.muli", "arith.mulf"]
            and op0.operands[0].get_name() == arg0
            and op0.operands[1].get_name() == arg0
            and op1.name in ["arith.addf", "arith.addi"]
            and check_commutative_operand(opr0, opr1, op0.result.get_name(), arg1)
            and op2.name == "linalg.yield"
            and op2.operands[0].get_name() == op1.result.get_name()
        ):
            return "reduce.l2_square"
    # not matched
    return ""


def ref_reduce(
    cache: MLIRCache, op: ir.OpView, tensors: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:

    reduce_type: str = match_reduce_type(op)
    if reduce_type == "":
        # create the buffer for result tensors
        tensors[cache.res[0]] = tensors[cache.opr[-1]].clone()
        in_shape: List[int] = list(op.operands[0].type.shape)
        out_shape: List[int] = list(op.result.type.shape)
        result_tensor: torch.Tensor = tensors[cache.opr[-1]].clone()
        reduce_loop(
            cache,
            op,
            0,
            in_shape,
            tensors,
            [0] * len(in_shape),
            [0] * len(out_shape),
            0,
            result_tensor,
        )
        return (result_tensor,)
    src = tensors[cache.opr[0]]
    reduce_buf = tensors[cache.opr[1]]
    dimensions: List[int] = [int(d) for d in op.attributes["dimensions"]]
    if reduce_type == "reduce.add":
        dst = reduce_buf + torch.sum(src, dimensions)
    elif reduce_type == "reduce.max":
        dst = torch.maximum(reduce_buf, torch.amax(src, dimensions))
    elif reduce_type == "reduce.min":
        dst = torch.minimum(reduce_buf, torch.amin(src, dimensions))
    elif reduce_type == "reduce.mul":
        dst = src
        for d in dimensions:
            dst = torch.prod(dst, d, keepdim=True)
        dst = dst.reshape_as(reduce_buf)
        dst = torch.mul(reduce_buf, dst)
    elif reduce_type == "reduce.l1":
        dst = torch.abs(src)
        dst = reduce_buf + torch.sum(dst, dimensions)
    elif reduce_type == "reduce.l2_square":
        dst = torch.square(src)
        dst = reduce_buf + torch.sum(dst, dimensions)
    else:
        raise Exception(f"Unsupport reduce type {reduce_type}")
    return (dst,)


def mlir_reduce(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:

    is_float: bool = benchgc.util.get_dtype(args[0].dtype).is_floating_point
    is_signed: bool = benchgc.util.get_dtype(args[0].dtype).is_signed

    buf: Callable[[ir.Context], ir.OpView] | None = None
    op: Callable[[ir.BlockArgument, ir.BlockArgument], ir.OpResult] | None = None

    if flags.case == "reduce.add" and is_float:
        buf = lambda ctx: args[1].get_zero_op(ctx)
        op = lambda arg0, arg1: arith.addf(arg0, arg1)
    elif flags.case == "reduce.add":
        buf = lambda ctx: args[1].get_zero_op(ctx)
        op = lambda arg0, arg1: arith.addi(arg0, arg1)
    elif flags.case == "reduce.mul" and is_float:
        buf = lambda ctx: args[1].get_constant_op(ctx, 1)
        op = lambda arg0, arg1: arith.mulf(arg0, arg1)
    elif flags.case == "reduce.mul":
        buf = lambda ctx: args[1].get_constant_op(ctx, 1)
        op = lambda arg0, arg1: arith.muli(arg0, arg1)
    elif flags.case == "reduce.max" and is_float:
        buf = lambda ctx: args[1].get_min_value_op(ctx)
        op = lambda arg0, arg1: arith.maximumf(arg0, arg1)
    elif flags.case == "reduce.max" and is_signed:
        buf = lambda ctx: args[1].get_min_value_op(ctx)
        op = lambda arg0, arg1: arith.maxsi(arg0, arg1)
    elif flags.case == "reduce.min" and is_float:
        buf = lambda ctx: args[1].get_max_value_op(ctx)
        op = lambda arg0, arg1: arith.minimumf(arg0, arg1)
    elif flags.case == "reduce.min" and is_signed:
        buf = lambda ctx: args[1].get_max_value_op(ctx)
        op = lambda arg0, arg1: arith.minsi(arg0, arg1)
    elif flags.case == "reduce.l1" and is_float:
        buf = lambda ctx: args[1].get_zero_op(ctx)
        op = lambda arg0, arg1: arith.addf(math.absf(arg0), arg1)
    elif flags.case == "reduce.l1":
        buf = lambda ctx: args[1].get_zero_op(ctx)
        op = lambda arg0, arg1: arith.addi(math.absi(arg0), arg1)
    elif flags.case == "reduce.l2_square" and is_float:
        buf = lambda ctx: args[1].get_zero_op(ctx)
        op = lambda arg0, arg1: arith.addf(arith.mulf(arg0, arg0), arg1)
    elif flags.case == "reduce.l2_square":
        buf = lambda ctx: args[1].get_zero_op(ctx)
        op = lambda arg0, arg1: arith.addi(arith.muli(arg0, arg0), arg1)
    else:
        raise Exception(
            f"not supported reduce case {flags.case} with data type {args[0].dtype}"
        )

    return init_reduce_module(flags.entry, args[0], args[1], flags.dimensions, buf, op)
