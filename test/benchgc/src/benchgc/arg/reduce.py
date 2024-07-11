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
import benchgc.arg

from typing import List, Tuple

def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:

    op, sdtype, ddtype, amp = params

    sdtype = benchgc.util.get_dtype(sdtype)
    ddtype = benchgc.util.get_dtype(ddtype)

    safe_to_reduce_elems: int = benchgc.util.get_problem_bounds(op, sdtype)[0]

    neutral_value: float = 1.0 if op == "mul" else 0.0

    shift: float = (
        1.0
        if (
            op == "mean"
            or op == "min"
            and not sdtype.is_signed
            and not ddtype.is_signed
        )
        else 0.0
    )

    value_range: int = benchgc.util.get_problem_bounds(op, sdtype)[1]

    is_mul_fp: bool = op == "mul" and sdtype.is_floating_point
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
        torch.full(
            shape, safe_to_reduce_elems / int(amp), dtype=torch.float32
        ),
    )
    if isinstance(non_neutral_mask, torch.Tensor):
        value = torch.where(non_neutral_mask, value, neutral_value)
    else:
        raise Exception("Flip coin failed when generate the reduce data filling")
    value = value + shift
    return value.to(dtype)

def compare(ref: torch.Tensor, res: torch.Tensor, verbose: int) -> Tuple[bool, bool | None]:
    dtype = ref.dtype
    ref = ref.to(torch.float)
    res = res.to(torch.float)
    return benchgc.arg.p2p(benchgc.util.get_eps(dtype), 30.0, ref, res, verbose)