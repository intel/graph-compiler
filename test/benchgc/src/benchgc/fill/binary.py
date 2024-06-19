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
import util
from typing import List

def fill(shape: List[int], dtype: torch.dtype, arg_name: str) -> torch.Tensor:

    if arg_name == "src0":
        arg = 1
    elif arg_name == "src1":
        arg = 2
    else:
        raise Exception("unknown arg name %s", arg_name)

    range_: int = 16
    f_min = 0 if dtype == torch.uint8 else -range_ // 2

    idx: torch.Tensor = torch.arange(util.nelem(shape) , dtype=torch.int).reshape(shape)
    values: torch.Tensor = (f_min + (12 * idx + 5 * arg + 16) % (range_ + 1)) * 1.25
    if arg == 1:
        values = torch.where(values == 0.0, 1, values)
    return values.to(dtype=dtype)
