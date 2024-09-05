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
from gc_mlir.dialects.linalg.opdsl.lang.comprehension import TypeFnType


def ref_batch_matmul(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.matmul(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_batch_matmul(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_matmul(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_batch_matmul_transpose_a(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.bmm(var[cache.opr[0]].transpose(-1, -2), var[cache.opr[1]]),)


def mlir_batch_matmul_transpose_a(
    flags: argparse.Namespace, args: List[Arg]
) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_matmul_transpose_a(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_batch_matmul_transpose_b(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.bmm(var[cache.opr[0]], var[cache.opr[1]].transpose(-1, -2)),)


def mlir_batch_matmul_transpose_b(
    flags: argparse.Namespace, args: List[Arg]
) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_matmul_transpose_b(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_batch_matvec(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # pytorch does not support bmv
    return (
        torch.matmul(var[cache.opr[0]], var[cache.opr[1]].unsqueeze(-1)).squeeze(-1),
    )


def mlir_batch_matvec(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_matvec(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_batch_mmt4d(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # [B, m, k, m0, k0] -> [B, m, m0, k, k0]
    _src = var[cache.opr[0]].permute([0, 1, 3, 2, 4]).contiguous()
    # [B, n, k, n0, k0] -> [B, k, k0, n, n0]
    _wei = var[cache.opr[1]].permute([0, 2, 4, 1, 3]).contiguous()

    # [B, m, m0, k, k0] -> [B, M, K]
    src = _src.reshape(
        [_src.shape[0], _src.shape[1] * _src.shape[2], _src.shape[3] * _src.shape[4]]
    )
    # [B, k, k0, n, n0] -> [B, K, N]
    wei = _wei.reshape(
        [_wei.shape[0], _wei.shape[1] * _wei.shape[2], _wei.shape[3] * _wei.shape[4]]
    )

    dst = torch.bmm(src, wei)
    # [B, M, N] -> [B, m, m0, n, n0]
    dst = dst.reshape(
        [dst.shape[0], _src.shape[1], _src.shape[2], _wei.shape[-2], _wei.shape[-1]]
    )

    # [B, m, m0, n, n0] -> [B, m, n, m0, n0]
    return (dst.transpose(2, 3).contiguous(),)


def mlir_batch_mmt4d(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_mmt4d(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_batch_reduce_matmul(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        torch.addbmm(
            input=torch.zeros(tuple()),
            batch1=var[cache.opr[0]],
            batch2=var[cache.opr[1]],
            beta=0,
            alpha=1,
        ),
    )


def mlir_batch_reduce_matmul(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_reduce_matmul(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_batch_vecmat(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        torch.matmul(var[cache.opr[0]].unsqueeze(-2), var[cache.opr[1]]).squeeze(-2),
    )


def mlir_batch_vecmat(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.batch_vecmat(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_dot(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.dot(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_dot(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.dot(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_matmul(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.mm(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_matmul(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.matmul(
                arg0, arg1, outs=[args[2].get_zero_op(ctx)], cast=TypeFnType(flags.cast)
            )
        ],
    )


def ref_matmul_transpose_a(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.mm(var[cache.opr[0]].transpose(-1, -2), var[cache.opr[1]]),)


def mlir_matmul_transpose_a(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.matmul_transpose_a(
                arg0, arg1, outs=[args[2].get_zero_op(ctx)], cast=TypeFnType(flags.cast)
            )
        ],
    )


def ref_matmul_transpose_b(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.mm(var[cache.opr[0]], var[cache.opr[1]].transpose(-1, -2)),)


def mlir_matmul_transpose_b(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.matmul_transpose_b(
                arg0, arg1, outs=[args[2].get_zero_op(ctx)], cast=TypeFnType(flags.cast)
            )
        ],
    )


def ref_matvec(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (torch.mv(var[cache.opr[0]], var[cache.opr[1]]),)


def mlir_matvec(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.matvec(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_mmt4d(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # [m, k, m0, k0] -> [m, m0, k, k0]
    _src = var[cache.opr[0]].permute([0, 2, 1, 3]).contiguous()
    # [n, k, n0, k0] -> [k, k0, n, n0]
    _wei = var[cache.opr[1]].permute([1, 3, 0, 2]).contiguous()

    # [m, m0, k, k0] -> [M, K]
    src = _src.reshape([_src.shape[0] * _src.shape[1], _src.shape[2] * _src.shape[3]])
    # [k, k0, n, n0] -> [K, N]
    wei = _wei.reshape([_wei.shape[0] * _wei.shape[1], _wei.shape[2] * _wei.shape[3]])

    dst = torch.mm(src, wei)
    # [M, N] -> [m, m0, n, n0]
    dst = dst.reshape([_src.shape[0], _src.shape[1], _wei.shape[-2], _wei.shape[-1]])

    # [m, m0, n, n0] -> [m, n, m0, n0]
    return (dst.transpose(1, 2).contiguous(),)


def mlir_mmt4d(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.mmt4d(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )


def ref_vecmat(
    cache: MLIRCache, op: ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        torch.matmul(var[cache.opr[0]].unsqueeze(-2), var[cache.opr[1]]).squeeze(-2),
    )


def mlir_vecmat(flags: argparse.Namespace, args: List[Arg]) -> ir.Module:
    return init_module(
        flags.entry,
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.vecmat(arg0, arg1, outs=[args[2].get_zero_op(ctx)])
        ],
    )
