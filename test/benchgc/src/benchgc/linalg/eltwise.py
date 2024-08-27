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

import gc_mlir.ir
import torch
from benchgc.arg import Arg
from benchgc.mlir.module import init_module
from benchgc.mlir.util import MLIRCache
from gc_mlir.dialects import linalg


def ref_abs(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.abs(var[cache.opr[0]]),)


def mlir_abs(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.abs(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_ceil(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.ceil(var[cache.opr[0]]),)


def mlir_ceil(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.ceil(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_floor(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.floor(var[cache.opr[0]]),)


def mlir_floor(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.floor(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_erf(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.erf(var[cache.opr[0]]),)


def mlir_erf(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.erf(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def mlir_log(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.log(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_log(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.log(var[cache.opr[0]]),)


def mlir_negf(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.negf(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_negf(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.neg(var[cache.opr[0]]),)


def ref_exp(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.exp(var[cache.opr[0]]),)


def mlir_exp(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.negf(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_round(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # torch.round is following the priciple "round half to even"
    # we need another implementation

    v = torch.floor(var[cache.opr[0]])
    return (v + torch.where(var[cache.opr[0]] - v >= 0.5, 1, 0),)


def mlir_round(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.round(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_rsqrt(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.rsqrt(var[cache.opr[0]]),)


def mlir_rsqrt(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.rsqrt(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_sqrt(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.sqrt(var[cache.opr[0]]),)


def mlir_sqrt(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.sqrt(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_square(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.square(var[cache.opr[0]]),)


def mlir_square(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.square(arg0, outs=[args[1].get_zero_op(ctx)])],
    )


def ref_tanh(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.tanh(var[cache.opr[0]]),)


def mlir_tanh(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0],),
        (args[1],),
        lambda ctx, arg0: [linalg.tanh(arg0, outs=[args[1].get_zero_op(ctx)])],
    )
