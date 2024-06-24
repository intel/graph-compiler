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
from typing import List

# params format: [alg, alpha, beta]
def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    alg, alpha, beta = params
    nelems = benchgc.util.nelem(shape)

    float_limit: torch.finfo = torch.finfo(torch.float32)

    alpha = 0.0 if alpha == "" else float(alpha)
    beta = 0.0 if beta == "" else float(beta)

    coeff = torch.tensor(
        [1, -1, 1, -1, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 1, 1]
    )
    bias = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 88.0, 22.0, 44.0, alpha, beta])
    rand_int_mask = torch.tensor(
        [
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    rand_uni_mask = torch.tensor(
        [
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ]
    )

    if alg == "log":
        # append more value for Log validation
        coeff = torch.cat((coeff, torch.tensor([1, 1])), dim=0)
        bias = torch.cat(
            (bias, torch.tensor([float_limit.max, float_limit.min])), dim=0
        )
        rand_int_mask = torch.cat(
            (rand_int_mask, torch.tensor([False, False])), dim=0
        )
        rand_uni_mask = torch.cat(
            (rand_uni_mask, torch.tensor([False, False])), dim=0
        )

    repeats: int = (nelems + coeff.nelement() - 1) // coeff.nelement()

    coeff = coeff.repeat(repeats)[:nelems]
    bias = bias.repeat(repeats)[:nelems]

    rand_int_mask = rand_int_mask.repeat(repeats)[:nelems]
    rand_int = torch.where(rand_int_mask, torch.randint(0, 10, [nelems]), 0)

    rand_uni_mask = rand_uni_mask.repeat(repeats)[:nelems]
    rand_uni = torch.where(rand_uni_mask, torch.rand(nelems) * 0.09, 0)

    value = ((rand_int + rand_uni) * coeff + bias).to(dtype=dtype)
    return value.reshape(shape)

