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
from .op import Op


class EltwiseOp(Op):
    fn: Callable[[torch.Tensor], torch.Tensor]
    fn_args: Dict[str, float]

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn_args = {}

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        dst = self.fn(src, **self.fn_args)
        outs.append(dst)

    def relax_checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        dt = util.get_dtype(self.op.outputs[offset].dtype)
        if dt == torch.float32 or dt == torch.float64:
            checker.threshold = 4e-5
        return checker

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        dt = util.get_dtype(self.op.outputs[offset].dtype)
        if dt == torch.float32 or dt == torch.float64:
            checker.threshold = 4e-6
        checker.zero_percent = 65.0
        return checker

    def fill_eltwise(self, lt: gapi.LogicalTensor) -> torch.Tensor:
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        nelems = lt.nelem()

        float_limit: torch.finfo = torch.finfo(torch.float32)
        alpha: float = self.op.get_optional_attr("alpha", float, 0.0)
        beta: float = self.op.get_optional_attr("beta", float, 0.0)
        # handle clamp op
        min_attr: float = self.op.get_optional_attr("min", float, 0.0)
        max_attr: float = self.op.get_optional_attr("max", float, 0.0)
        alpha = alpha if alpha != 0.0 and min_attr == 0.0 else min_attr
        beta = beta if beta != 0.0 and max_attr == 0.0 else max_attr

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

        if self.op.kind == "Log":
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
        return value.reshape(lt.shape)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {
            self.op.inputs[0].id: self.fill_eltwise(self.op.inputs[0]),
        }

        return res


class AbsOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.abs, "abs"

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.truncate_negative = True
        return checker


class ClampOp(EltwiseOp):
    max: float
    min: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.min = op.get_required_attr("min", float)
        self.max = op.get_required_attr("max", float)
        self.fn, self.gc_op = torch.clamp, "clamp"
        self.fn_args["min"] = self.min
        self.fn_args["max"] = self.max

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        if self.min == 0:
            checker.truncate_negative = True
            if self.max == 0:
                checker.zero_percent = 100.0
        return checker


class EluOp(EltwiseOp):
    alpha: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.alpha = op.get_required_attr("alpha", float)
        self.fn, self.gc_op = torch.nn.functional.elu, "elu"
        self.fn_args["alpha"] = self.alpha

    def checker(self, offset: int) -> util.Checker:
        return self.relax_checker(offset)


class ExpOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.exp, "exp"


class GELUOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.nn.functional.gelu, "gelu"


# pytorch is differ from graph api
# calculate the reference based on the formula
class HardSigmoidOp(EltwiseOp):
    alpha: float
    beta: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.alpha = op.get_required_attr("alpha", float)
        self.beta = op.get_required_attr("beta", float)
        self.gc_op = "hardsigmoid"
        self.fn = lambda inp: (inp * self.alpha + self.beta).clamp(0, 1)


class HardSwishOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.nn.functional.hardswish, "hardswish"


class LeakyReLUOp(EltwiseOp):
    alpha: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.alpha = op.get_required_attr("alpha", float)
        self.fn, self.gc_op = torch.nn.functional.leaky_relu, "leaky_relu"
        self.fn_args["negative_slope"] = self.alpha


class LogOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.log, "log"

    def checker(self, offset: int) -> util.Checker:
        return self.relax_checker(offset)


class MishOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.nn.functional.mish, "mish"

    def checker(self, offset: int) -> util.Checker:
        return self.relax_checker(offset)


class PowOp(EltwiseOp):
    beta: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.beta = op.get_required_attr("beta", float)
        self.gc_op = "pow"

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]) -> None:
        src = ins[0]
        dst = torch.pow(src, self.beta)
        outs.append(dst)


class ReciprocalOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.reciprocal, "reciprocal"


class ReLUOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.nn.functional.relu, "relu"

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.truncate_negative = True
        return checker


class RoundOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.round, "round"


class SigmoidOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.nn.functional.sigmoid, "sigmoid"


class SoftPlusOp(EltwiseOp):
    beta: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.beta = op.get_optional_attr("alpha", float, 1.0)
        self.fn, self.gc_op = torch.nn.functional.softplus, "soft_plus"
        self.fn_args["beta"] = self.beta

    def checker(self, offset: int) -> util.Checker:
        return self.relax_checker(offset)


class SqrtOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.sqrt, "squared_root"


class SquareOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.square, "square"


class TanhOp(EltwiseOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn, self.gc_op = torch.tanh, "tanh"

    def checker(self, offset: int) -> util.Checker:
        return self.relax_checker(offset)
