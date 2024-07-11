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

import numpy
import torch
from typing import Callable, Tuple, Any, Union, List
from functools import reduce
import operator
import ml_dtypes

# verbose level
NO_VERBOSE = 0
COMPARE_VERBOSE = 1  # + print threshold for comparison
ERROR_OUTPUT_VERBOSE = 2  # + print all error data points if failed
OUTPUT_VERBOSE = 3  # + print all result including passed tensor
INPUT_VERBOSE = 4  # + print input torch tensors

"""
acc | acc | elems | value_range | worst case
s32 | mul |  10   |       3     |     3^10=2^16, out of 2^30 (max integer)
f16 | mul |  10   |       1     | (2^1)^10=2^10, out of 2^16 (max exponent)
f32 | mul |  30   |       3     | (2^3)^30=2^90, out of 2^128 (max exponent)
s32 | sum | 10000 |      50     | 10000*50=2^19, out of 2^30 (max integer)
f16 | sum | 1000  |       8     | 1000*8=2^13, out of 2^10 (max mantissa/integer)
f32 | sum | 10000 |      16     | 10000*16=2^18, out of 2^23 (max mantissa/integer)
 min/max  |  all  |    1000     | no limits on accumulation chain

In f16 cases, the worst case exceeds the data type bounds, however it's rare
to reach these extreme cases as long as they're close (can't just use f32 bounds)
"""
# first: nonneutral elements
# second: maximum range
_problem_bounds = {
    "mul_int": (10, 3),
    "mul_fp16": (10, 1),
    "mul_fp32": (30, 3),
    "sum_int": (10000, 50),
    "sum_fp16": (1000, 8),
    "sum_fp32": (10000, 16),
    "minmax_int": (-1, 1000),
    "minmax_fp": (-1, 1000),
}
_dtype_2_range = {
    "f32": (-16777216, 16777216),
    "f64": (-16777216, 16777216),
    "f16": (-2048, 2048),
    "bf16": (-16777216, 16777216),
    "f8_e5m2": (-2048, 2048),
    "f8_e4m3": (-2048, 2048),
    "u8": (0, 255),
    "s8": (-128, 127),
    "s32": (-2147483648, 2147483520),
}


def flip_coin(
    seed: Union[Any, torch.Tensor], prob: Union[float, torch.Tensor]
) -> Union[bool, torch.Tensor]:
    big_prime: int = 1000003
    prime: int = 753737
    seed = seed * prime
    return (seed % big_prime) < (prob * big_prime)


def get_problem_bounds(kind: str, dt: torch.dtype) -> Tuple[int, int]:
    if not dt.is_floating_point:
        if kind in ["max", "min"]:
            return _problem_bounds["minmax_int"]
        elif kind == "mul":
            return _problem_bounds["mul_int"]
        else:
            return _problem_bounds["sum_int"]
    elif kind in ["max", "min"]:
        return _problem_bounds["minmax_fp"]
    elif kind == "mul":
        return (
            _problem_bounds["mul_fp16"]
            if dt == torch.float16
            else _problem_bounds["mul_fp32"]
        )
    else:
        return (
            _problem_bounds["sum_fp16"]
            if dt == torch.float16
            else _problem_bounds["sum_fp32"]
        )


def get_type_range(dt: str) -> Tuple[float, float]:
    return _dtype_2_range[dt]


# Lnorm, Bnorm & Conv
def get_digits(dtype: str) -> int:
    return {
        "f32": 24,
        "f64": 53,
        "s8": 7,
        "u8": 8,
        "f16": 11,
        "bf16": 8,
        "f8_e5m2": 3,
        "f8_e4m3": 4,
    }[dtype]


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "f32":
        return torch.float32
    elif dtype == "f64":
        return torch.float64
    elif dtype == "f16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "u8" or dtype == "ui8":
        return torch.uint8
    elif dtype == "s8" or dtype == "i8":
        return torch.int8
    elif dtype == "boolean":
        return torch.uint8
    elif dtype == "f8_e4m3":
        return torch.float8_e4m3fn
    elif dtype == "f8_e5m2":
        return torch.float8_e5m2
    elif dtype == "s32" or dtype == "i32":
        return torch.int32
    else:
        raise Exception("data type not support: %s" % dtype)


def tensor_to_ndarray(tensor: torch.Tensor) -> Any:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()

def get_eps(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).eps if dtype.is_floating_point else 0.0


_seed: int = 0


def set_seed(seed: int):
    global _seed
    _seed = seed


def torch_seed(seed_scale: int = 1, seed_shift: int = 0):
    torch.manual_seed(_seed * seed_scale + seed_shift)


def iterate_tensor(tensor: torch.Tensor, fn: Callable[[Tuple[int, ...]], None]):
    index: List[int] = [0] * tensor.ndim

    def dfs(depth: int):
        if depth == tensor.ndim:
            fn(tuple(index))
        else:
            for i in range(tensor.shape[depth]):
                index[depth] = i
                dfs(depth + 1)


# indicate how to check the result
class Checker:
    use_norm: bool
    # set if negative result is trancated to zero
    truncate_negative: bool
    eltwise_relax: bool
    threshold: float
    zero_percent: float
    # args: [ref, res, abs_diff, rel_diff]
    customized_checker: (
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        | None
    )

    def __init__(
        self,
        threshold: float,
        zero_percent: float,
        use_norm: bool = False,
        eltwise_relax: bool = False,
        truncate_negative: bool = False,
        checker: (
            Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
            ]
            | None
        ) = None,
    ) -> None:
        self.use_norm = use_norm
        self.eltwise_relax = eltwise_relax
        self.threshold = threshold
        self.zero_percent = zero_percent
        self.truncate_negative = truncate_negative
        self.customized_checker = checker

    def check(
        self, ref: torch.Tensor, res: torch.Tensor, verbose: int
    ) -> Tuple[bool, bool | None]:
        if self.use_norm:
            return self.norm(ref, res, verbose)
        else:
            return self.p2p(ref, res, verbose)

    def norm(
        self, ref: torch.Tensor, res: torch.Tensor, verbose: int
    ) -> Tuple[bool, bool | None]:

        f32_ref = ref.to(torch.float32)
        f32_res = res.to(torch.float32)
        if f32_ref.nelement() == 0:
            return (True, None)

        diff_square_sum = torch.square(torch.subtract(f32_ref, f32_res)).sum()
        square_sum = torch.square(f32_ref).sum()

        l2_diff_norm = torch.sqrt(diff_square_sum / square_sum).item()
        if verbose >= COMPARE_VERBOSE:
            print(
                "norm check: %.10f / threshold: %.10f" % (l2_diff_norm, self.threshold)
            )

        return (l2_diff_norm < self.threshold, None)

    def p2p(
        self, ref: torch.Tensor, res: torch.Tensor, verbose: int
    ) -> Tuple[bool, bool | None]:

        if verbose >= COMPARE_VERBOSE:
            print("p2p check: threshold: %.7f" % self.threshold)
        f32_ref = ref.to(torch.float32)
        f32_res = res.to(torch.float32)

        check = torch.BoolTensor([False])

        check = check.bitwise_or(torch.bitwise_and(f32_ref.isnan(), f32_res.isnan()))
        check = check.bitwise_or(
            torch.bitwise_and(f32_ref.isneginf(), f32_res.isneginf())
        )
        check = check.bitwise_or(
            torch.bitwise_and(f32_ref.isposinf(), f32_res.isposinf())
        )

        # choose diff/rel_diff based on value
        abs_diff = (f32_ref - f32_res).abs()
        rel_diff = abs_diff / torch.where(
            f32_ref.abs() > numpy.finfo(numpy.float32).smallest_subnormal,
            f32_ref.abs(),
            1,
        )
        # pick a diff for comparison
        diff = torch.where(f32_ref.abs() > 1e-5, rel_diff, abs_diff)

        check = check.bitwise_or(diff <= self.threshold)

        if self.eltwise_relax:
            check = check.bitwise_or(abs_diff <= max(torch.finfo(res.dtype).eps, 2e-5))

        if self.customized_checker is not None:
            check = check.bitwise_or(
                self.customized_checker(ref, res, abs_diff, rel_diff)
            )

        if verbose >= OUTPUT_VERBOSE:
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
            elif self.truncate_negative:
                mistrust = (
                    zero * 100.0 / res.nelement() > 50.0 + self.zero_percent / 2.0
                )
            else:
                mistrust = zero * 100.0 / res.nelement() > self.zero_percent
            return (True, mistrust)
        else:
            if (
                verbose < OUTPUT_VERBOSE
            ):  # skip verbose print if full output tensor is alrady printed
                fail = torch.argwhere(torch.where(check, 0, 1))
                if verbose < ERROR_OUTPUT_VERBOSE:
                    fail = fail[
                        :10
                    ]  # only print top 10 failed data points if verbose level does not satisfied
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


def nelem(shape: List[int]) -> int:
    return reduce(operator.mul, shape)
