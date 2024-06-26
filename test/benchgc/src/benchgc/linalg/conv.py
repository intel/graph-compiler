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

from gc_mlir._mlir_libs._mlir.ir import DenseIntElementsAttr

from benchgc.arg import Arg
from typing import Dict, Tuple

def __ref_init(
    op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str]:

    src = var[escape_var(op.operands[0].get_name())]
    wei = var[escape_var(op.operands[1].get_name())]
    dst_var: str = escape_var(op.results[0].get_name())
    return (src, wei, dst_var)


def map_conv_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src", "arg1": "wei", "1": "dst"}.items():
        args[k] = args[v]
        del args[v]


def ref_conv_1d_ncw_fcw(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)

    strides: DenseIntElementsAttr = op.attributes["strides"]
    dilations: DenseIntElementsAttr = op.attributes["dilations"]

    var[dst_var] = torch.conv1d(input=src, weight=wei, stride=strides[0], dilation=dilations[0])


def mlir_conv_1d_ncw_fcw(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_conv_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.conv_1d_ncw_fcw(arg0, arg1, outs = [args["1"].get_empty_op(ctx)], strides = (flags.stride_w,), dilations = (flags.dilation_w,)))

def ref_conv_1d_nwc_wcf(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)

    strides: DenseIntElementsAttr = op.attributes["strides"]
    dilations: DenseIntElementsAttr = op.attributes["dilations"]

    var[dst_var] = torch.conv1d(
        input=src.transpose(1, 2), # nwc -> ncw
        weight=wei.transpose(0, 2), # wcf -> fcw
        stride=strides[0], dilation=dilations[0]).transpose(1, 2) # nfw -> nwf


def mlir_conv_1d_nwc_wcf(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_conv_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.conv_1d_nwc_wcf(arg0, arg1, outs = [args["1"].get_empty_op(ctx)], strides = (flags.stride_w,), dilations = (flags.dilation_w,)))


def ref_conv_1d(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)

    var[dst_var] = torch.conv1d(
        input=src.unsqueeze(0).unsqueeze(0), # w -> ncw
        weight=wei.unsqueeze(0).unsqueeze(0), # w -> fcw
        ).squeeze(0, -1) # nwf -> w


def mlir_conv_1d(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_conv_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.conv_1d(arg0, arg1, outs = [args["1"].get_empty_op(ctx)]))

def ref_conv_2d_nchw_fchw(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)

    strides: DenseIntElementsAttr = op.attributes["strides"]
    dilations: DenseIntElementsAttr = op.attributes["dilations"]

    var[dst_var] = torch.conv2d(input=src, weight=wei, stride=(strides[0], strides[1]), dilation=(dilations[0], dilations[1]))

def mlir_conv_2d_nchw_fchw(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_conv_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.conv_2d_nchw_fchw(arg0, arg1, outs = [args["1"].get_empty_op(ctx)], strides = (flags.stride_h, flags.stride_w), dilations = (flags.dilation_h, flags.dilation_w)))



def ref_conv_2d_ngchw_fgchw(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var)

    strides: DenseIntElementsAttr = op.attributes["strides"]
    dilations: DenseIntElementsAttr = op.attributes["dilations"]

    group: int = wei.shape[1]

    # src shape: [n, g, ic/g, h, w]
    
    nchw_src = src.reshape([src.shape[0], src.shape[1] * src.shape[2]] + list(src.shape[3:]))

    # wei shape: [oc/g, g, ic/g, h, w]
    # we need to merge axis 0 & 1 into oc
    # to merge axis, we need to generate a new buffer to ensure the axis we merge is contiguous

    fchw_wei = wei.transpose(0, 1).contiguous().reshape([wei.shape[0] * wei.shape[1]] + list(wei.shape[2:]))

    nfhw_dst = torch.conv2d(input=nchw_src, weight=fchw_wei, groups=group, stride=(strides[0], strides[1]), dilation=(dilations[0], dilations[1]))
    ngfhw_dst = nfhw_dst.reshape([nfhw_dst.shape[0], group, nfhw_dst.shape[1] // group] + list(nfhw_dst.shape[2:]))
    var[dst_var] = ngfhw_dst


def mlir_conv_2d_ngchw_fgchw(
    flags: argparse.Namespace, args: Dict[str, Arg]
) -> gc_mlir.ir.Module:
    map_conv_args(args)
    return init_i2o1_module(args["arg0"], args["arg1"], args["1"], lambda ctx, arg0, arg1: linalg.conv_2d_ngchw_fgchw(arg0, arg1, outs = [args["1"].get_empty_op(ctx)], strides = (flags.stride_h, flags.stride_w), dilations = (flags.dilation_h, flags.dilation_w)))




