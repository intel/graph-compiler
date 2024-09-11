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

import benchgc.arg
import benchgc.util
import torch
from benchgc.arg.arg import Arg
from benchgc.arg.compare import p2p

op: Set[str] = set(
    [
        "linalg.reduce.add",
        "linalg.reduce.mul",
        "linalg.reduce.max",
        "linalg.reduce.min",
        "linalg.reduce.l1",
        "linalg.reduce.l2_square",
    ]
)


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 0:
        raise Exception("reduce fill: dst filling is not allowed")
    arg.fill_param = [
        "reduce",
        flags.case,
        arglist[0].dtype,
        arglist[1].dtype,
        str(arglist[0].nelem() // arglist[1].nelem()),
    ]
    arg.fill_type = "D"

def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:

    op, sdtype, ddtype, amp = params

    sdtype = benchgc.util.get_dtype(sdtype)
    ddtype = benchgc.util.get_dtype(ddtype)

    safe_to_reduce_elems: int = benchgc.util.get_problem_bounds(op, sdtype)[0]

    neutral_value: float = 1.0 if op == "reduce.mul" else 0.0

    shift: float = (
        1.0
        if (op == "reduce.min" and not sdtype.is_signed and not ddtype.is_signed)
        else 0.0
    )

    value_range: int = benchgc.util.get_problem_bounds(op, sdtype)[1]

    is_mul_fp: bool = op == "reduce.mul" and sdtype.is_floating_point
    min_range: int = -value_range if is_mul_fp else 1

    index = torch.arange(benchgc.util.nelem(shape)).reshape(shape)

    benchgc.util.torch_seed()
    value = torch.randint(min_range, value_range + 1, size=shape)
    if is_mul_fp:
        value = torch.pow(2, value)
    if sdtype.is_signed:  # random choose positive or negative
        value = torch.where(torch.BoolTensor(size=shape), value, -value)

    non_neutral_mask = benchgc.util.flip_coin(
        index,
        torch.full(shape, safe_to_reduce_elems / int(amp), dtype=torch.float32),
    )
    if isinstance(non_neutral_mask, torch.Tensor):
        value = torch.where(non_neutral_mask, value, neutral_value)
    else:
        raise Exception("Flip coin failed when generate the reduce data filling")
    value = value + shift
    return value.to(dtype)


def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = ["reduce", arg.dtype, flags.case]

def compare(
    param: List[str], ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = ref.dtype
    ref = ref.to(torch.float)
    res = res.to(torch.float)
    return p2p(benchgc.util.get_eps(dtype), 30.0, ref, res, verbose)
