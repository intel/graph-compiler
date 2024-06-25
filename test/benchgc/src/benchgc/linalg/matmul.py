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
import argparse
import gc_mlir.ir

from benchgc.linalg.mlir import init_i2o1_module, escape_var

from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, Tuple


def __ref_init(
    op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str]:

    src = var[escape_var(op.operands[0].get_name())]
    wei = var[escape_var(op.operands[1].get_name())]
    dst_var: str = escape_var(op.results[0].get_name())
    return (src, wei, dst_var)


def map_matmul_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src", "arg1": "wei", "1": "dst"}.items():
        args[k] = args[v]
        del args[v]


def ref_batch_matmul(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)
    var[dst_var] = torch.matmul(src, wei)


def mlir_batch_matmul(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_matmul(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))


def ref_batch_matmul_transpose_a(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)
    var[dst_var] = torch.matmul(src.transpose(-1, -2), wei)


def mlir_batch_matmul_transpose_a(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_matmul_transpose_a(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))


def ref_batch_matmul_transpose_b(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)
    var[dst_var] = torch.matmul(src, wei.transpose(-1, -2))


def mlir_batch_matmul_transpose_b(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_matmul_transpose_b(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))


def ref_batch_matvec(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)
    var[dst_var] = torch.matmul(src, wei.unsqueeze(-1)).squeeze(-1)


def mlir_batch_matvec(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_matvec(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))


def ref_batch_mmt4d(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)

    # [N, m, k, m0, k0] -> [N, m, m0, k, k0]
    untiled_src = src.permute([0, 1, 3, 2, 4])
    # [N, n, k, n0, k0] -> [N, k, k0, n, n0]
    untiled_wei = wei.permute([0, 2, 4, 1, 3])

    # permute will change the stride
    # use copy_ to force reset the stride
    # we need to reshape and merge the axis later
    _src = torch.zeros(untiled_src.shape).copy_(untiled_src)
    _wei = torch.zeros(untiled_wei.shape).copy_(untiled_wei)

    # merge axis
    # [N, m, m0, k, k0] -> [N, m * m0, k * k0]
    _src = _src.reshape(
        [_src.shape[0], _src.shape[1] * _src.shape[2], _src.shape[3] * _src.shape[4]]
    )
    # [N, k, k0, n, n0] -> [N, k * k0, n * n0]
    _wei = _wei.reshape(
        [_wei.shape[0], _wei.shape[1] * _wei.shape[2], _wei.shape[3] * _wei.shape[4]]
    )

    var[dst_var] = torch.matmul(_src, _wei).reshape([src.shape[0], src.shape[1], src.shape[3], wei.shape[1], wei.shape[3]]).transpose(2, 3)


def mlir_batch_mmt4d(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_mmt4d(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))

def ref_batch_reduce_matmul(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)
    var[dst_var] = torch.sum(torch.matmul(src, wei), 0, keepdim=False)


def mlir_batch_reduce_matmul(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_reduce_matmul(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))

def ref_batch_vecmat(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)
    var[dst_var] = torch.matmul(src.unsqueeze(-2), wei).squeeze(-2)


def mlir_batch_vecmat(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.batch_vecmat(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))

