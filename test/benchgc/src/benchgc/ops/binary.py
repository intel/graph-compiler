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
from typing import List, Callable, Dict
from .op import Op, to_ncx_if_needed, to_nxc_if_needed


class BinaryOp(Op):
    auto_broadcast: str
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.auto_broadcast = op.get_optional_attr("auto_broadcast", str, "numpy")

    def check_auto_broadcast(self, x: torch.Tensor, y: torch.Tensor):
        if self.auto_broadcast == "none" and x.shape != y.shape:
            raise Exception("shape mismatch %s and %s" % (x.shape, y.shape))

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src_0, src_1 = ins[0], ins[1]
        self.check_auto_broadcast(src_0, src_1)
        dst = self.f(src_0, src_1)
        outs.append(dst)

    def fill_binary(self, lt: gapi.LogicalTensor, arg: int) -> torch.Tensor:
        shape: List[int] = lt.shape
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        range_: int = 16
        f_min = 0 if dtype == torch.uint8 else -range_ // 2

        idx: torch.Tensor = torch.arange(lt.nelem(), dtype=torch.int).reshape(shape)
        values: torch.Tensor = (f_min + (12 * idx + 5 * arg + 16) % (range_ + 1)) * 1.25
        if arg == 1:
            values = torch.where(values == 0.0, 1, values)
        return values.to(dtype=dtype)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}
        assert len(self.op.inputs) == 2
        res[self.op.inputs[0].id] = self.fill_binary(self.op.inputs[0], 1)
        res[self.op.inputs[1].id] = self.fill_binary(self.op.inputs[1], 2)
        return res


class AddOp(BinaryOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.f = torch.add


class BiasAddOp(BinaryOp):
    data_format: str

    def __init__(self, op: gapi.Op):
        self.data_format = op.get_optional_attr("data_format", str, "NXC")

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, bias = ins[0], ins[1]
        ncx_src = to_ncx_if_needed(self.data_format, src)

        shape_ = [1] * ncx_src.ndim
        shape_[1] = ncx_src.shape[1]

        ncx_dst = src + bias.reshape(shape_)
        dst = to_nxc_if_needed(self.data_format, ncx_dst)
        outs.append(dst)


class DivideOp(BinaryOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.f = torch.divide

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        out = self.op.outputs[offset]
        eps = util.get_eps(util.get_dtype(out.dtype))
        checker.customized_checker = lambda ref, res, abs_diff, rel_diff: abs_diff < eps
        return checker


class MaximumOp(BinaryOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.f = torch.maximum


class MinimumOp(BinaryOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.f = torch.minimum


class MultiplyOp(BinaryOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.f = torch.mul


class SubtractOp(BinaryOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.f = torch.subtract
