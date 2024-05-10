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

from .. import util
import torch
from .. import gapi
from typing import List, Dict
from .op import Op

# Scales is a f32 1D tensor to be applied to the quantization formula.
# For qtype = per-tensor, there should be only one element in the scales tensor.
# For qtype = per-channel, the element number should be equal to the element number of src tensor along the dimension axis.
# Graph API only accepts those two types of qtype


# pass an op, not a LogicalTensor !
class QuantOp(Op):
    qtype: str
    axis: int

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.qtype = op.get_optional_attr("qtype", str, "per_tensor")

        if self.qtype == "per_channel":
            self.axis = op.get_optional_attr("axis", int, 1)
            ndim: int = len(op.inputs[0].shape)
            self.axis = (self.axis + ndim) % ndim

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.zero_percent = 80.0
        return checker

    def fill_scales(self, lt: gapi.LogicalTensor) -> torch.Tensor:
        util.torch_seed()
        return torch.pow(torch.Tensor([2]), torch.randint(-2, 3, lt.shape)).to(
            util.get_dtype(lt.dtype)
        )

    def fill_zps(self, lt: gapi.LogicalTensor) -> torch.Tensor:
        shape = lt.shape
        sdtype: torch.dtype = util.get_dtype(lt.dtype)
        if sdtype in [torch.int8, torch.uint8, torch.int32]:
            min_val = torch.iinfo(sdtype).min
            util.torch_seed()
            return torch.randint(min_val, 2, shape).to(sdtype)
        else:
            raise Exception("unsupported data type: %s" % sdtype)

    def fill_src(self, lt: gapi.LogicalTensor) -> torch.Tensor:
        c_min, c_max = util.get_type_range(lt.dtype)
        gen: torch.Tensor = torch.tensor(
            [c_max, c_min, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 16.0, 64.0]
        )
        return (
            gen.repeat((lt.nelem() + 9) // 10)[: lt.nelem()]
            .reshape(lt.shape)
            .to(util.get_dtype(lt.dtype))
        )

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}
        lt = self.op.inputs[0]
        res[lt.id] = self.fill_src(lt)
        return res


class DequantizeOp(QuantOp):
    scales: List[float]
    zps: List[int] | None

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.scales = op.get_required_attr("scales", list)
        self.zps = op.get_optional_attr("zps", list, None)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0].to(torch.float32)
        if self.qtype == "per_tensor":
            dst = (
                src * self.scales[0]
                if self.zps is None
                else (src - self.zps[0]) * self.scales[0]
            )
        else:
            # construct the shape for zps & scales
            shape = [1] * src.ndim
            shape[self.axis] = src.shape[self.axis]
            scales = torch.Tensor(self.scales).reshape(shape)
            if self.zps is None:
                dst = src * scales
            else:
                zps = torch.Tensor(self.zps).reshape(shape)
                dst = (src - zps) * scales
        outs.append(dst)


class DynamicDequantizeOp(QuantOp):
    scales: List[float]
    zps: List[int] | None

    def __init__(self, op: gapi.Op):
        super().__init__(op)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, scales = ins[0].to(torch.float32), ins[1]
        zps = ins[2] if len(ins) > 2 else None

        if self.qtype == "per_tensor":
            dst = src * scales[0] if zps is None else (src - zps[0]) * scales[0]
        else:
            # construct the shape for zps & scales
            shape = [1] * src.ndim
            shape[self.axis] = src.shape[self.axis]
            if zps is None:
                dst = src * scales.reshape(shape)
            else:
                dst = (src - zps.reshape(shape)) * scales.reshape(shape)
        outs.append(dst)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        # fill src first
        res = super().fill_data()

        lt = self.op.inputs[1]
        res[lt.id] = self.fill_scales(lt)
        if len(self.op.inputs) > 2:
            lt = self.op.inputs[2]
            res[lt.id] = self.fill_zps(lt)
        return res


class DynamicQuantizeOp(QuantOp):
    scales: List[float]
    zps: List[int] | None

    def __init__(self, op: gapi.Op):
        super().__init__(op)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, scales = ins[0].to(torch.float32), ins[1]
        zps = ins[2] if len(ins) > 2 else None

        if self.qtype == "per_tensor":
            dst = src / scales[0] if zps is None else src / scales[0] + zps[0]
        else:
            # construct the shape for zps & scales
            shape = [1] * src.ndim
            shape[self.axis] = src.shape[self.axis]
            if zps is None:
                dst = src / scales.reshape(shape)
            else:
                dst = src / scales.reshape(shape) + zps.reshape(shape)
        dst = torch.round(dst)
        outs.append(dst)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        # fill src first
        res = super().fill_data()

        lt = self.op.inputs[1]
        res[lt.id] = self.fill_scales(lt)
        if len(self.op.inputs) > 2:
            lt = self.op.inputs[2]
            res[lt.id] = self.fill_zps(lt)
        return res


class QuantizeOp(QuantOp):
    scales: List[float]
    zps: List[int] | None

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.scales = op.get_required_attr("scales", list)
        self.zps = op.get_optional_attr("zps", list, None)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0].to(torch.float32)
        if self.qtype == "per_tensor":
            dst = (
                src / self.scales[0]
                if self.zps is None
                else src / self.scales[0] + self.zps[0]
            )
        else:
            # construct the shape for zps & scales
            shape = [1] * src.ndim
            shape[self.axis] = src.shape[self.axis]
            scales = torch.Tensor(self.scales).reshape(shape)
            if self.zps is None:
                dst = src / scales
            else:
                zps = torch.Tensor(self.zps).reshape(shape)
                dst = src / scales + zps
        dst = torch.round(dst)
        outs.append(dst)


class TypeCastOp(Op):
    def __init__(self, op: gapi.Op):
        super().__init__(op)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        # do nothing, let the runner do the cast in the uniform way
        src = ins[0]
        dst = src
        outs.append(dst)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}
        lt = self.op.inputs[0]
        res[lt.id] = self.generic_fill(lt)
        return res
