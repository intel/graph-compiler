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
from typing import List, Tuple

def __fill(shape: List[int], dtype: torch.dtype, density: float, rng: Tuple[int, int]) -> torch.Tensor:
    target = torch.empty(size=shape, dtype=torch.float32)
    target = target.view(-1)

    arg_min, arg_max = rng

    benchgc.util.torch_seed()

    density_t = torch.full(shape, density, dtype=torch.float32)
    bernoulli_t = torch.bernoulli(density_t)
    condi = density_t == 1
    is_one_t = torch.where(condi, True, bernoulli_t)
    gen_value = arg_min + (arg_max - arg_min) * torch.rand(size=shape)
    target = is_one_t * gen_value

    # make sure the first element is positive
    first_val = target.flatten()[0]
    if first_val <= 0.0:
        while first_val <= 0.0:
            first_val = torch.rand(1)[0].item() * (arg_max - arg_min) + arg_min
        target_f = target.view(-1)
        target_f[0] = first_val
        target = target_f.view(shape)

    return target.to(dtype=dtype)


# params format: [src | wei, src dt, wei dt, dst dt, amp]

def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:

    name, src_dt, wei_dt, dst_dt, amp = params

    # calculate density for src filling
    src_dtype, wei_dtype = benchgc.util.get_dtype(src_dt), benchgc.util.get_dtype(wei_dt)
    src_rng: Tuple[int, int] = {
        torch.float32: (-32, 32),
        torch.bfloat16: (-4, 4),
        torch.float16: (-4, 4),
    }[src_dtype]
    wei_rng: Tuple[int, int] = {
        torch.float32: (-32, 32),
        torch.bfloat16: (-4, 4),
        torch.float16: (-2, 2),
    }[wei_dtype]

    if name == "src":
        return __fill(shape, src_dtype, 1.0, src_rng)
    elif name == "wei":
        max_value: int = src_rng[1] * wei_rng[1]
        safe_digits: int = min(
            benchgc.util.get_digits("f32"), benchgc.util.get_digits(dst_dt)
        )
        safe_n_acc = (1 << safe_digits) // max_value
        density = min(1.0, safe_n_acc / int(amp))
        return __fill(shape, wei_dtype, density, wei_rng)
    else:
        raise Exception("unknown arg name %s", name)
