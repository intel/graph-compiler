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


from functools import reduce
import operator
from .. import gapi, util
from typing import List, Dict
import torch
from .op import Op


def fill_softmax(lt: gapi.LogicalTensor, axis: int) -> torch.Tensor:
    dtype = util.get_dtype(lt.dtype)
    outer: int = reduce(operator.mul, lt.shape[:axis], 1)
    inner: int = reduce(operator.mul, lt.shape[axis + 1 :], 1)
    util.torch_seed()
    sign = torch.randint(0, 1, size=[1, lt.shape[axis], 1]) * 2 - 1
    value = torch.randint(87, 90, size=[outer, lt.shape[axis], inner])
    value = torch.where(value == 87, 0, value)
    value = value * sign
    value = torch.where(value == 0, torch.finfo(dtype).min, value)
    return value.reshape(lt.shape).to(dtype)


class LogSoftmaxOp(Op):
    axis: int

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.axis = op.get_required_attr("axis", int)
        ndim: int = len(op.inputs[0].shape)
        self.axis = (self.axis + ndim) % ndim

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        dst = torch.log_softmax(src, self.axis)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        dt = util.get_dtype(self.op.outputs[0].dtype)
        if dt == torch.float32 or dt == torch.float64:
            checker.threshold *= 50.0
        else:
            checker.threshold = 0.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        res: Dict[int, torch.Tensor] = {lt.id: fill_softmax(lt, self.axis)}
        return res


class SoftMaxOp(Op):
    axis: int

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.axis = op.get_required_attr("axis", int)
        ndim: int = len(op.inputs[0].shape)
        self.axis = (self.axis + ndim) % ndim

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        dst = torch.softmax(src, self.axis)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        dt = util.get_dtype(self.op.outputs[0].dtype)
        if dt == torch.float32 or dt == torch.float64:
            checker.threshold *= 10.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        res: Dict[int, torch.Tensor] = {lt.id: fill_softmax(lt, self.axis)}
        return res
