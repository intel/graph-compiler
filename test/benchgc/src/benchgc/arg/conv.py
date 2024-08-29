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
from typing import Dict, List, Set, Tuple

import benchgc.util
import torch
from benchgc.arg.arg import Arg
from benchgc.arg.compare import p2p

op: Set[str] = set(
    [
        "linalg.conv_1d_ncw_fcw",
        "linalg.conv_1d_nwc_wcf",
        "linalg.conv_1d",
        "linalg.conv_2d_nchw_fchw",
        "linalg.conv_2d_ngchw_fgchw",
        "linalg.conv_2d_ngchw_gfchw",
        "linalg.conv_2d_nhwc_fhwc",
        "linalg.conv_2d_nhwc_hwcf",
        "linalg.conv_2d",
        "linalg.conv_3d_ncdhw_fcdhw",
        "linalg.conv_3d_ndhwc_dhwcf",
        "linalg.conv_3d",
        "linalg.depthwise_conv_1d_ncw_cw",
        "linalg.depthwise_conv_1d_nwc_wc",
        "linalg.depthwise_conv_1d_nwc_wcm",
        "linalg.depthwise_conv_2d_nchw_chw",
        "linalg.depthwise_conv_2d_nhwc_hwc",
        "linalg.depthwise_conv_2d_nhwc_hwcm",
        "linalg.depthwise_conv_3d_ncdhw_cdhw",
        "linalg.depthwise_conv_3d_ndhwc_dhwc",
        "linalg.depthwise_conv_3d_ndhwc_dhwcm",
    ]
)

# params format: [src | wei, src dt, wei dt, dst dt, amp]


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 2:
        raise Exception("conv fill: dst filling is not allowed")
    arg.fill_type = "D"
    arg.fill_param = [
        "conv",
        "src" if arg.index == 0 else "wei",
        arglist[0].dtype,
        arglist[1].dtype,
        arglist[2].dtype,
    ]

    # find the amplifier of the conv
    wei = arglist[1]
    nelem = wei.nelem()
    if flags.driver == "linalg":
        if flags.case in [
            "conv_1d_ncw_fcw",
            "conv_2d_nchw_fchw",
            "conv_2d_ngchw_fgchw",
            "conv_2d_nhwc_fhwc",
            "conv_3d_ncdhw_fcdhw",
        ]:
            arg.fill_param.append(str(nelem // wei.shape[0]))
        elif flags.case in ["conv_2d_ngchw_gfchw"]:
            arg.fill_param.append(str(nelem // wei.shape[1]))
        elif flags.case in [
            "conv_1d_nwc_wcf",
            "conv_2d_nhwc_hwcf",
            "conv_3d_ndhwc_dhwcf",
            "depthwise_conv_1d_nwc_wcm",
            "depthwise_conv_2d_nhwc_hwcm",
            "depthwise_conv_3d_ndhwc_dhwcm",
        ]:
            arg.fill_param.append(str(nelem // wei.shape[-1]))
        elif flags.case in [
            "conv_1d",
            "conv_2d",
            "conv_3d",
            "depthwise_conv_1d_ncw_cw",
            "depthwise_conv_1d_nwc_wc",
            "depthwise_conv_2d_nchw_chw",
            "depthwise_conv_2d_nhwc_hwc",
            "depthwise_conv_3d_ncdhw_cdhw",
            "depthwise_conv_3d_ndhwc_dhwc",
        ]:
            arg.fill_param.append(str(nelem))


def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    name, src_dt, wei_dt, dst_dt, amp = params

    arg_rng: List[Dict[torch.dtype, Tuple[int, int]]] = [
        {
            torch.float32: (-32, 32),
            torch.bfloat16: (-4, 4),
            torch.float16: (-4, 4),
        },  # src
        {
            torch.float32: (-32, 32),
            torch.bfloat16: (-8, 8),
            torch.float16: (-2, 2),
        },  # wei
    ]

    target = torch.empty(size=shape, dtype=torch.float32)
    target = target.view(-1)

    src_dt = benchgc.util.get_dtype(src_dt)
    wei_dt = benchgc.util.get_dtype(wei_dt)

    src_min, src_max = arg_rng[0][src_dt]
    wei_min, wei_max = arg_rng[1][wei_dt]
    max_value = max(abs(src_min), abs(src_max)) * max(abs(wei_min), abs(wei_max))
    safe_digits: int = min(
        benchgc.util.get_digits("f32"), benchgc.util.get_digits(dst_dt)
    )
    safe_n_acc = (1 << safe_digits) // max_value

    if name == "src":
        arg_min, arg_max = arg_rng[0][src_dt]
        density = 1.0
    elif name == "wei":
        arg_min, arg_max = arg_rng[1][wei_dt]
        density = min(safe_n_acc / int(amp), 1.0)
    else:
        raise Exception("unknown arg name %s", name)

    benchgc.util.torch_seed()

    density_t = torch.full(shape, density, dtype=torch.float32)
    bernoulli_t = torch.bernoulli(density_t)
    condi = density_t == 1
    is_one_t = torch.where(condi, True, bernoulli_t)
    gen_value = torch.randint(arg_min, arg_max + 1, size=shape)
    target = is_one_t * gen_value

    # make sure the first element is positive
    first_val = target.flatten()[0]
    if first_val <= 0.0:
        while first_val <= 0.0:
            first_val = torch.randint(arg_min, arg_max + 1, size=()).item()
        target_f = target.view(-1)
        target_f[0] = first_val
        target = target_f.view(shape)

    return target.to(dtype=dtype)


def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = ["conv", arg.dtype, flags.case]


def compare(
    param: List[str], ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = benchgc.util.get_dtype(param[0])

    ref = ref.to(torch.float)
    res = res.to(torch.float)
    return p2p(
        0.0,  # use a relax threshold if using wino
        70.0 if dtype == torch.uint8 else 85.0,
        ref,
        res,
        verbose,
    )
