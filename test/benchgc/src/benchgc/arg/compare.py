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

from typing import Callable, List, Tuple

import benchgc.util
import numpy
import torch


def iterate_tensor(tensor: torch.Tensor, fn: Callable[[Tuple[int, ...]], None]):
    if tensor.ndim == 0:
        fn(tuple())
        return
    index: List[int] = [0] * tensor.ndim

    def dfs(depth: int):
        if depth == tensor.ndim:
            fn(tuple(index))
        else:
            for i in range(tensor.shape[depth]):
                index[depth] = i
                dfs(depth + 1)

    dfs(0)


def norm(
    threshold: float, ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:

    f32_ref = ref.to(torch.float32)
    f32_res = res.to(torch.float32)
    if f32_ref.nelement() == 0:
        return (True, None)

    diff_square_sum = torch.square(torch.subtract(f32_ref, f32_res)).sum()
    square_sum = torch.square(f32_ref).sum()

    l2_diff_norm = torch.sqrt(diff_square_sum / square_sum).item()
    if verbose >= benchgc.util.COMPARE_VERBOSE:
        print(f"norm check: {l2_diff_norm:.10f} / threshold: {threshold:.10f}")

    return (l2_diff_norm < threshold, None)


def p2p(
    threshold: float,
    zero_percent: float,
    ref: torch.Tensor,
    res: torch.Tensor,
    verbose: int,
    init_check: torch.Tensor | None = None,
) -> Tuple[bool, bool | None]:

    if verbose >= benchgc.util.COMPARE_VERBOSE:
        print(f"p2p check: threshold: {threshold:.7f}")
    f32_ref = ref.to(torch.float32)
    f32_res = res.to(torch.float32)

    if init_check is None:
        check = torch.tensor(False)
    else:
        check = init_check

    check = check.bitwise_or(torch.bitwise_and(f32_ref.isnan(), f32_res.isnan()))
    check = check.bitwise_or(torch.bitwise_and(f32_ref.isneginf(), f32_res.isneginf()))
    check = check.bitwise_or(torch.bitwise_and(f32_ref.isposinf(), f32_res.isposinf()))

    # choose diff/rel_diff based on value
    abs_diff = (f32_ref - f32_res).abs()
    rel_diff = abs_diff / torch.where(
        f32_ref.abs() > numpy.finfo(numpy.float32).smallest_subnormal,
        f32_ref.abs(),
        1,
    )
    # pick a diff for comparison
    diff = torch.where(f32_ref.abs() > 1e-5, rel_diff, abs_diff)

    check = check.bitwise_or(diff <= threshold)

    if verbose >= benchgc.util.OUTPUT_VERBOSE:
        iterate_tensor(
            check,
            lambda idx: print(
                "%20s: ref: %12.7f res: %12.7f abs_diff: %12.7f rel_diff: %12.7f"
                % (
                    idx,
                    f32_ref[idx].item(),
                    f32_res[idx].item(),
                    abs_diff[idx].item(),
                    rel_diff[idx].item(),
                )
            ),
        )
    if check.all():
        # check mistrusted
        zero = res.nelement() - res.count_nonzero().item()
        if res.nelement() < 10:
            mistrust = False
        else:
            mistrust = zero * 100.0 / res.nelement() > zero_percent
        return (True, mistrust)
    else:
        if (
            verbose < benchgc.util.OUTPUT_VERBOSE
        ):  # skip verbose print if full output tensor is alrady printed
            fail = torch.argwhere(torch.where(check, 0, 1))
            if verbose < benchgc.util.ERROR_OUTPUT_VERBOSE:
                # only print top 10 failed data points if verbose level does not satisfied
                fail = fail[:10]
            for idx in fail:
                index: Tuple[int, ...] = tuple(idx.tolist())
                print(
                    "%20s: ref: %12.7f res: %12.7f abs_diff: %12.7f rel_diff: %12.7f"
                    % (
                        index,
                        f32_ref[index].item(),
                        f32_res[index].item(),
                        abs_diff[index].item(),
                        rel_diff[index].item(),
                    )
                )
        return (False, None)
