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
from .. import gapi, util
from typing import List, Dict
from abc import ABC, abstractmethod


class Op(ABC):
    op: gapi.Op

    def __init__(self, op: gapi.Op):
        self.op = op

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]) -> None:
        raise Exception("No reference implementation for OP %s" % self.op.kind)

    def checker(self, offset: int) -> util.Checker:
        out = self.op.outputs[offset]
        eps = util.get_eps(util.get_dtype(out.dtype))
        return util.Checker(eps, 30.0, False, False, out.dtype == "u8")

    # generic fill without special design
    def generic_fill(
        self, lt: gapi.LogicalTensor, f_min: int = -32, f_max: int = 32
    ) -> torch.Tensor:
        shape: List[int] = lt.shape
        dtype: torch.dtype = util.get_dtype(lt.dtype)

        f_min = 0 if dtype == torch.uint8 and f_min < 0 else f_min
        if dtype == torch.float16:
            dt = 1
        elif dtype == torch.bfloat16:
            dt = 2
        elif dtype == torch.float32:
            dt = 3
        elif dtype == torch.bool:
            dt = 8
        elif dtype == torch.int8:
            dt = 5
        elif dtype == torch.uint8:
            dt = 6
        else:
            raise Exception("unsupported data type: %s" % dtype)
        util.torch_seed(dt, lt.nelem())
        value = torch.randint(f_min, f_max + 1, size=shape, dtype=torch.float32)
        return value.to(dtype=dtype)

    @abstractmethod
    def fill_data(self) -> Dict[int, torch.Tensor]:
        raise NotImplementedError("Subclasses must have their own implementations.")


def to_ncx_if_needed(data_format: str, x: torch.Tensor):
    if data_format == "NXC":
        if x.ndim < 3:
            return x
        perm: List[int] = [0, x.ndim - 1]
        for i in range(1, x.ndim - 1):
            perm.append(i)
        return x.permute(perm)
    else:
        return x


def to_nxc_if_needed(data_format: str, x: torch.Tensor):
    if data_format == "NXC":
        if x.ndim < 3:
            return x
        perm: List[int] = [0]
        for i in range(2, x.ndim):
            perm.append(i)
        perm.append(1)
        return x.permute(perm)
    else:
        return x