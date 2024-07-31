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
import benchgc.util
from benchgc.arg.arg import Arg
from benchgc.arg.compare import p2p
import argparse
from typing import List, Dict, Tuple, Set

# params format: [src | wei, src dt, wei dt, dst dt, amp]
# use other filling type for bias

op: Set[str] = set(["linalg.matmul_transpose_b"])


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 1:
        raise Exception("matmul fill: dst filling is not allowed")
    arg.fill_type = "D"
    arg.fill_param = [
        "matmul",
        "src" if arg.index == 0 else "wei",
        arglist[0].dtype,
        arglist[1].dtype,
        arglist[2].dtype,
    ]

    # find the amplifier K of the matmul
    if flags.driver == "linalg" and flags.case == "matmul_transpose_b":
        arg.fill_param.append(str(arg.shape[-1]))


def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    name, src_dt, wei_dt, dst_dt, amp = params

    arg_rng: List[Dict[torch.dtype, Tuple[int, int]]] = [
        {
            torch.float32: (-64, 64),
            torch.bfloat16: (-4, 4),
            torch.float16: (-4, 4),
        },  # src
        {
            torch.float32: (-128, 128),
            torch.bfloat16: (-8, 8),
            torch.float16: (-2, 2),
        },  # wei
    ]

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
        arg_min, arg_max = arg_rng[0][wei_dt]
        density = min(safe_n_acc / int(amp), 1.0)
    else:
        raise Exception("unknown arg name %s", name)

    benchgc.util.torch_seed(1, 0 if name == "src" else 1)
    value = torch.bernoulli(torch.full(shape, density)) * torch.randint(
        arg_min, arg_max, shape
    )
    while value.flatten()[0] <= 0:
        value.flatten()[0] = torch.randint(arg_min, arg_max + 1, size=[1])[0].item()

    return value.to(dtype)


def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = ["matmul"]


def compare(
    ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = ref.dtype
    ref = ref.to(torch.float)
    res = res.to(torch.float)
    return p2p(benchgc.util.get_eps(dtype), 30.0, ref, res, verbose)
