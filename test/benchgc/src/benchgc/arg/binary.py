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

# op should use this filling

op: Set[str] = set(
    ["linalg.add", "linalg.div", "linalg.mul", "linalg.max", "linalg.min", "linalg.sub"]
)

# params format: [src0 | src1, src0 dt, src1 dt, dst dt]


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 1:
        raise Exception("binary fill: dst filling is not allowed")
    arg.fill_type = "D"
    arg.fill_param = [
        "binary",
        "src0" if arg.index == 0 else "src1",
        arglist[0].dtype,
        arglist[1].dtype,
        arglist[2].dtype,
    ]


def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    name, _, _, _ = params

    accept_name: Dict[str, int] = {"src0": 1, "src1": 2}
    if name in accept_name:
        arg: int = accept_name[name]
    else:
        raise Exception("unknown arg name %s", name)

    range_: int = 16
    f_min = 0 if dtype == torch.uint8 else -range_ // 2

    idx: torch.Tensor = torch.arange(
        benchgc.util.nelem(shape), dtype=torch.int
    ).reshape(shape)
    values: torch.Tensor = (f_min + (12 * idx + 5 * arg + 16) % (range_ + 1)) * 1.25
    if arg == 2:
        values = torch.where(values == 0.0, 1, values)
    return values.to(dtype=dtype)


# compare param: dtype, case


def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = ["binary", arg.dtype, flags.case]


def compare(
    param: List[str], ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = benchgc.util.get_dtype(param[0])

    ref = ref.to(torch.float)
    res = res.to(torch.float)

    if param[1] in ["div", "div_unsigned"]:
        abs_diff = (ref.to(torch.float) - res.to(torch.float)).abs()
        init_check = abs_diff < benchgc.util.get_eps(dtype)
    else:
        init_check = None

    return p2p(benchgc.util.get_eps(dtype), 30.0, ref, res, verbose, init_check)
