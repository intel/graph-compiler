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
from typing import List, Set, Tuple

import benchgc.util
import torch
from benchgc.arg.arg import Arg
from benchgc.arg.compare import p2p

op: Set[str] = set(
    [
        "linalg.pooling_nchw_max",
        "linalg.pooling_nchw_sum",
        "linalg.pooling_ncw_max",
        "linalg.pooling_ncw_sum",
        "linalg.pooling_ndhwc_max",
        "linalg.pooling_ndhwc_sum",
        "linalg.pooling_nhwc_max",
        "linalg.pooling_nhwc_sum",
        "linalg.pooling_nhwc_min",
        "linalg.pooling_nwc_max",
        "linalg.pooling_nwc_min",
        "linalg.pooling_nwc_sum",
    ]
)


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 1:
        raise Exception("pool fill: dst filling is not allowed")
    arg.fill_type = "D"
    arg.fill_param = ["pool"]


def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    arg_rng: Tuple[int, int] = {
        torch.float64: (-2048, 2048),
        torch.float32: (-2048, 2048),
        torch.int32: (-2048, 2048),
        torch.bfloat16: (-32, 32),
        torch.float16: (-32, 32),
        torch.int8: (-128, 127),
        torch.uint8: (0, 255),
    }[dtype]

    benchgc.util.torch_seed()
    target = torch.randint(arg_rng[0], arg_rng[1] + 1, size=[benchgc.util.nelem(shape)])
    # make sure the first element is not negative
    if target[0] <= 0.0:
        while target[0] <= 0.0:
            target[0] = torch.randint(arg_rng[0], arg_rng[1], size=(1,))[0].item()

    return target.reshape(shape).to(dtype=dtype)


def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = ["pool", arg.dtype, flags.case]


def compare(
    param: List[str], ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = benchgc.util.get_dtype(param[0])

    ref = ref.to(torch.float)
    res = res.to(torch.float)
    return p2p(
        benchgc.util.get_eps(dtype) * 10.0,
        99.0,
        ref,
        res,
        verbose,
    )
