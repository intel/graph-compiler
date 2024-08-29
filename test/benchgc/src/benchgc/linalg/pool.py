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


def ref_pooling_nchw_max(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool2d(
            var[cache.opr[0]],
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        ),
    )


def mlir_pooling_nchw_max(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nchw_max(
                arg0,
                arg1,
                outs=[args[2].get_min_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nchw_sum(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    # pytorch does not support pooling on sum
    # avg_pool2d or lp_pool2d with p = 1 does not support dilation
    # we will use depthwise convolution with kernel equals to 1 to calculate the sum

    # FIXME: improve the code if pytorch support the sum pooling with dilation

    channel = var[cache.opr[0]].shape[1]
    kernel = var[cache.opr[1]]
    return (
        torch.conv2d(
            var[cache.opr[0]],
            torch.ones(channel, 1, kernel.shape[0], kernel.shape[1]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=channel,
        ),
    )


def mlir_pooling_nchw_sum(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nchw_sum(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_ncw_max(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool1d(
            var[cache.opr[0]],
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        ),
    )


def mlir_pooling_ncw_max(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_ncw_max(
                arg0,
                arg1,
                outs=[args[2].get_min_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_ncw_sum(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    # pytorch does not support pooling on sum
    # avg_pool1d or lp_pool1d with p = 1 does not support dilation
    # we will use depthwise convolution with kernel equals to 1 to calculate the sum

    # FIXME: improve the code if pytorch support the sum pooling with dilation

    channel = var[cache.opr[0]].shape[1]
    kernel = var[cache.opr[1]]
    return (
        torch.conv1d(
            var[cache.opr[0]],
            torch.ones(channel, 1, kernel.shape[0]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=channel,
        ),
    )


def mlir_pooling_ncw_sum(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_ncw_sum(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_ndhwc_max(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool3d(
            var[cache.opr[0]].permute([0, -1, 1, 2, 3]),
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 3, 4, 1])
        .contiguous(),
    )


def mlir_pooling_ndhwc_max(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_ndhwc_max(
                arg0,
                arg1,
                outs=[args[2].get_min_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_ndhwc_sum(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    # pytorch does not support pooling on sum
    # avg_pool3d or lp_pool3d with p = 1 does not support dilation
    # we will use depthwise convolution with kernel equals to 1 to calculate the sum

    # FIXME: improve the code if pytorch support the sum pooling with dilation

    channel = var[cache.opr[0]].shape[-1]
    kernel = var[cache.opr[1]]
    return (
        torch.conv3d(
            var[cache.opr[0]].permute([0, -1, 1, 2, 3]),
            torch.ones(channel, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=channel,
        )
        .permute([0, 2, 3, 4, 1])
        .contiguous(),
    )


def mlir_pooling_ndhwc_sum(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_ndhwc_sum(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nhwc_max(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool2d(
            var[cache.opr[0]].permute([0, -1, 1, 2]),
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 3, 1])
        .contiguous(),
    )


def mlir_pooling_nhwc_max(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nhwc_max(
                arg0,
                arg1,
                outs=[args[2].get_min_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nhwc_sum(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    # pytorch does not support pooling on sum
    # avg_pool2d or lp_pool2d with p = 1 does not support dilation
    # we will use depthwise convolution with kernel equals to 1 to calculate the sum

    # FIXME: improve the code if pytorch support the sum pooling with dilation

    channel = var[cache.opr[0]].shape[-1]
    kernel = var[cache.opr[1]]
    return (
        torch.conv2d(
            var[cache.opr[0]].permute([0, -1, 1, 2]),
            torch.ones(channel, 1, kernel.shape[0], kernel.shape[1]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=channel,
        )
        .permute([0, 2, 3, 1])
        .contiguous(),
    )


def mlir_pooling_nhwc_sum(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nhwc_sum(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nhwc_min(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool2d(
            var[cache.opr[0]].permute([0, -1, 1, 2]).neg(),
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 3, 1])
        .neg()
        .contiguous(),
    )


def mlir_pooling_nhwc_min(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nhwc_min(
                arg0,
                arg1,
                outs=[args[2].get_max_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nwc_max(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool1d(
            var[cache.opr[0]].permute([0, -1, 1]),
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 1])
        .contiguous(),
    )


def mlir_pooling_nwc_max(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nwc_max(
                arg0,
                arg1,
                outs=[args[2].get_min_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nwc_min(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.max_pool1d(
            var[cache.opr[0]].permute([0, -1, 1]).neg(),
            kernel_size=var[cache.opr[1]].shape,
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 1])
        .contiguous()
        .neg(),
    )


def mlir_pooling_nwc_min(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nwc_min(
                arg0,
                arg1,
                outs=[args[2].get_max_value_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_pooling_nwc_sum(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    # pytorch does not support pooling on sum
    # avg_pool3d or lp_pool3d with p = 1 does not support dilation
    # we will use depthwise convolution with kernel equals to 1 to calculate the sum

    # FIXME: improve the code if pytorch support the sum pooling with dilation

    channel = var[cache.opr[0]].shape[-1]
    kernel = var[cache.opr[1]]
    return (
        torch.conv1d(
            var[cache.opr[0]].permute([0, -1, 1]),
            torch.ones(channel, 1, kernel.shape[0]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=channel,
        )
        .permute([0, 2, 1])
        .contiguous(),
    )


def mlir_pooling_nwc_sum(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.pooling_nwc_sum(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )
