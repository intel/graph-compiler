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
from typing import Dict, List, Tuple

import torch
from benchgc.arg import Arg
from benchgc.mlir.module import init_module
from benchgc.mlir.util import MLIRCache
from gc_mlir import ir
from gc_mlir.dialects import linalg


def ref_add(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.add(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_add(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.add(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_powf(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.pow(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_powf(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.powf(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_div(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.div(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_div(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.div(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_max(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.max(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_max(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.max(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_min(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.min(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_min(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.min(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_mul(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.mul(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_mul(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.mul(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_sub(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.sub(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_sub(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.sub(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )
