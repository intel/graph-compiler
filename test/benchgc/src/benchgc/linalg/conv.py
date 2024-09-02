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


def ref_conv_1d_ncw_fcw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv1d(
            var[cache.opr[0]],
            var[cache.opr[1]],
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        ),
    )


def mlir_conv_1d_ncw_fcw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_1d_ncw_fcw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_1d_nwc_wcf(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    # src: nwc -> ncw
    # wei: wcf -> fcw
    # dst: nwf -> nfw

    return (
        torch.conv1d(
            var[cache.opr[0]].permute([0, 2, 1]),
            var[cache.opr[1]].permute([2, 1, 0]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 1])
        .contiguous(),
    )


def mlir_conv_1d_nwc_wcf(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_1d_nwc_wcf(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_1d_ncw_fcw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv1d(
            var[cache.opr[0]],
            var[cache.opr[1]],
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        ),
    )


def mlir_conv_1d_ncw_fcw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_1d_ncw_fcw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_1d(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        torch.conv1d(
            var[cache.opr[0]].unsqueeze(0).unsqueeze(0),
            var[cache.opr[1]].unsqueeze(0).unsqueeze(0),
        )
        .squeeze(0)
        .squeeze(0),
    )


def mlir_conv_1d(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_1d(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
            )
        ],
    )


def ref_conv_2d_nchw_fchw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv2d(
            var[cache.opr[0]],
            var[cache.opr[1]],
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        ),
    )


def mlir_conv_2d_nchw_fchw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_2d_nchw_fchw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_2d_ngchw_fgchw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    src = var[cache.opr[0]]
    wei = var[cache.opr[1]]
    groups: int = src.shape[1]

    dst = torch.conv2d(
        src.reshape(
            [src.shape[0], src.shape[1] * src.shape[2], src.shape[3], src.shape[4]]
        ),  # merge group axis with channel
        wei.transpose(0, 1)
        .contiguous()
        .reshape(
            [wei.shape[0] * wei.shape[1], wei.shape[2], wei.shape[3], wei.shape[4]]
        ),  # merge group axis with output channel
        stride=[strides[i] for i in range(len(strides))],
        dilation=[dilations[i] for i in range(len(dilations))],
        groups=groups,
    )
    return (
        dst.reshape(
            [dst.shape[0], groups, dst.shape[1] // groups, dst.shape[2], dst.shape[3]]
        ),
    )  # split group axis from output channel


def mlir_conv_2d_ngchw_fgchw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_2d_ngchw_fgchw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_2d_ngchw_gfchw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]

    src = var[cache.opr[0]]
    wei = var[cache.opr[1]]
    groups: int = src.shape[1]

    dst = torch.conv2d(
        src.reshape(
            [src.shape[0], src.shape[1] * src.shape[2], src.shape[3], src.shape[4]]
        ),  # merge group axis with channel
        wei.reshape(
            [wei.shape[0] * wei.shape[1], wei.shape[2], wei.shape[3], wei.shape[4]]
        ),  # merge group axis with output channel
        stride=[strides[i] for i in range(len(strides))],
        dilation=[dilations[i] for i in range(len(dilations))],
        groups=groups,
    )
    return (
        dst.reshape(
            [dst.shape[0], groups, dst.shape[1] // groups, dst.shape[2], dst.shape[3]]
        ),
    )  # split group axis from output channel


def mlir_conv_2d_ngchw_gfchw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_2d_ngchw_gfchw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_2d_nhwc_fhwc(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv2d(
            var[cache.opr[0]].permute([0, 3, 1, 2]),
            var[cache.opr[1]].permute([0, 3, 1, 2]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 3, 1])
        .contiguous(),
    )


def mlir_conv_2d_nhwc_fhwc(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_2d_nhwc_fhwc(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_2d_nhwc_hwcf(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv2d(
            var[cache.opr[0]].permute([0, 3, 1, 2]),
            var[cache.opr[1]].permute([3, 2, 0, 1]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 3, 1])
        .contiguous(),
    )


def mlir_conv_2d_nhwc_hwcf(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_2d_nhwc_hwcf(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_2d(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        torch.conv2d(
            var[cache.opr[0]].unsqueeze(0).unsqueeze(0),
            var[cache.opr[1]].unsqueeze(0).unsqueeze(0),
        )
        .squeeze(0)
        .squeeze(0),
    )


def mlir_conv_2d(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_2d(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
            )
        ],
    )


def ref_conv_3d_ncdhw_fcdhw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv3d(
            var[cache.opr[0]],
            var[cache.opr[1]],
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        ),
    )


def mlir_conv_3d_ncdhw_fcdhw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_3d_ncdhw_fcdhw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_3d_ndhwc_dhwcf(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    return (
        torch.conv3d(
            var[cache.opr[0]].permute([0, 4, 1, 2, 3]),
            var[cache.opr[1]].permute([4, 3, 0, 1, 2]),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
        )
        .permute([0, 2, 3, 4, 1])
        .contiguous(),
    )


def mlir_conv_3d_ndhwc_dhwcf(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_3d_ndhwc_dhwcf(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_conv_3d(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    return (
        torch.conv3d(
            var[cache.opr[0]].unsqueeze(0).unsqueeze(0),
            var[cache.opr[1]].unsqueeze(0).unsqueeze(0),
        )
        .squeeze(0)
        .squeeze(0),
    )


def mlir_conv_3d(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.conv_3d(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
            )
        ],
    )


def ref_depthwise_conv_1d_ncw_cw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[1]
    return (
        torch.conv1d(
            var[cache.opr[0]],
            var[cache.opr[1]].unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        ),
    )


def mlir_depthwise_conv_1d_ncw_cw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_1d_ncw_cw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_1d_nwc_wc(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[-1]
    return (
        torch.conv1d(
            var[cache.opr[0]].transpose(-1, -2),
            var[cache.opr[1]].transpose(-1, -2).unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        )
        .transpose(-1, -2)
        .contiguous(),
    )


def mlir_depthwise_conv_1d_nwc_wc(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_1d_nwc_wc(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_1d_nwc_wcm(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    src = var[cache.opr[0]]
    groups: int = src.shape[-1]
    wei = var[cache.opr[1]]
    dst = (
        torch.conv1d(
            src.transpose(-1, -2),
            wei.reshape([wei.shape[0], wei.shape[1] * wei.shape[2]])
            .transpose(-1, -2)
            .unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        )
        .transpose(-1, -2)
        .contiguous()
    )
    return (dst.reshape([dst.shape[0], dst.shape[1], wei.shape[1], wei.shape[2]]),)


def mlir_depthwise_conv_1d_nwc_wcm(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_1d_nwc_wcm(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_2d_nchw_chw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[1]
    return (
        torch.conv2d(
            var[cache.opr[0]],
            var[cache.opr[1]].unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        ),
    )


def mlir_depthwise_conv_2d_nchw_chw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_2d_nchw_chw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_2d_nhwc_hwc(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[-1]
    return (
        torch.conv2d(
            var[cache.opr[0]].permute([0, 3, 1, 2]),
            var[cache.opr[1]].permute([2, 0, 1]).unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        )
        .permute([0, 2, 3, 1])
        .contiguous(),
    )


def mlir_depthwise_conv_2d_nhwc_hwc(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_2d_nhwc_hwc(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_2d_nhwc_hwcm(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[-1]
    wei = var[cache.opr[1]]
    dst = (
        torch.conv2d(
            var[cache.opr[0]].permute([0, 3, 1, 2]),
            wei.reshape([wei.shape[0], wei.shape[1], wei.shape[2] * wei.shape[3]])
            .permute([2, 0, 1])
            .unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        )
        .permute([0, 2, 3, 1])
        .contiguous()
    )
    return (
        dst.reshape(
            [dst.shape[0], dst.shape[1], dst.shape[2], wei.shape[-2], wei.shape[-1]]
        ),
    )


def mlir_depthwise_conv_2d_nhwc_hwcm(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_2d_nhwc_hwcm(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_3d_ncdhw_cdhw(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[1]
    return (
        torch.conv3d(
            var[cache.opr[0]],
            var[cache.opr[1]].unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        ),
    )


def mlir_depthwise_conv_3d_ncdhw_cdhw(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_3d_ncdhw_cdhw(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_3d_ndhwc_dhwc(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[-1]
    return (
        torch.conv3d(
            var[cache.opr[0]].permute([0, 4, 1, 2, 3]),
            var[cache.opr[1]].permute([3, 0, 1, 2]).unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        )
        .permute([0, 2, 3, 4, 1])
        .contiguous(),
    )


def mlir_depthwise_conv_3d_ndhwc_dhwc(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_3d_ndhwc_dhwc(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )


def ref_depthwise_conv_3d_ndhwc_dhwcm(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    strides: gc_mlir.ir.DenseIntElementsAttr = op.attributes["strides"]
    dilations: gc_mlir.ir.DenseIntElementsAttr = op.attributes["dilations"]
    groups: int = var[cache.opr[0]].shape[-1]
    wei = var[cache.opr[1]]
    dst = (
        torch.conv3d(
            var[cache.opr[0]].permute([0, 4, 1, 2, 3]),
            wei.reshape(
                [wei.shape[0], wei.shape[1], wei.shape[2], wei.shape[3] * wei.shape[4]]
            )
            .permute([3, 0, 1, 2])
            .unsqueeze(1),
            stride=[strides[i] for i in range(len(strides))],
            dilation=[dilations[i] for i in range(len(dilations))],
            groups=groups,
        )
        .permute([0, 2, 3, 4, 1])
        .contiguous()
    )
    return (
        dst.reshape(
            [
                dst.shape[0],
                dst.shape[1],
                dst.shape[2],
                dst.shape[3],
                wei.shape[-2],
                wei.shape[-1],
            ]
        ),
    )


def mlir_depthwise_conv_3d_ndhwc_dhwcm(
    flags: argparse.Namespace, args: List[Arg]
) -> gc_mlir.ir.Module:
    return init_module(
        (args[0], args[1]),
        (args[2],),
        lambda ctx, arg0, arg1: [
            linalg.depthwise_conv_3d_ndhwc_dhwcm(
                arg0,
                arg1,
                outs=[args[2].get_zero_op(ctx)],
                strides=flags.strides,
                dilations=flags.dilations,
            )
        ],
    )
