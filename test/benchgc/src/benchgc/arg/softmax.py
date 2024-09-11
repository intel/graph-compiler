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
import operator
from functools import reduce
from typing import List, Set, Tuple

import benchgc.util
import torch
from benchgc.arg.arg import Arg
from benchgc.arg.compare import p2p

op: Set[str] = set(["linalg.softmax"])


# params format: [reduce dimension]


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 0:
        raise Exception("softmax fill: dst filling is not allowed")
    arg.fill_type = "D"
    arg.fill_param = ["softmax", str(flags.dimension)]


def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    dimension: int = int(params[0])

    outer: int = reduce(operator.mul, shape[:dimension], 1)
    inner: int = reduce(operator.mul, shape[dimension + 1 :], 1)
    benchgc.util.torch_seed()
    sign = torch.randint(0, 1, size=[1, shape[dimension], 1]) * 2 - 1
    value = torch.randint(87, 90, size=[outer, shape[dimension], inner])
    value = torch.where(value == 87, 0, value)
    value = value * sign
    value = torch.where(value == 0, torch.finfo(dtype).min, value)
    return value.reshape(shape).to(dtype)


# param: dtype, case, reduce size
def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = [
        "softmax",
        arg.dtype,
        flags.case,
        str(arg.shape[int(flags.dimension)]),
    ]


def compare(
    param: List[str], ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = benchgc.util.get_dtype(param[0])
    ref = ref.to(torch.float)
    res = res.to(torch.float)

    reduce_size = int(param[2])
    nzeros = (
        reduce_size - 1
        if dtype == torch.int8 or dtype == torch.uint8
        else max(0, reduce_size - 8)
    )

    return p2p(
        benchgc.util.get_eps(dtype) * (5.0 if dtype == torch.float else 1.0),
        100.0 * nzeros / reduce_size,
        ref,
        res,
        verbose,
    )
