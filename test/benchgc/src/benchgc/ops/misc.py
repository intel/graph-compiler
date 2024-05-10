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
from typing import List, Dict, Tuple
import torch
from .op import Op
import math


class ConcatOp(Op):
    axis: int

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.axis = op.get_required_attr("axis", int)
        ndim: int = len(op.inputs[0].shape)
        self.axis = (self.axis + ndim) % ndim

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        dst = torch.concat(ins, self.axis)
        outs.append(dst)

    def __fill(self, offset: int) -> torch.Tensor:
        lt = self.op.inputs[offset]
        target: torch.Tensor = torch.empty(size=lt.shape, dtype=torch.float32)

        # set proper range of valid values
        min_val: int = torch.iinfo(torch.int8).min
        max_val: int = torch.iinfo(torch.uint8).max

        util.torch_seed(1, offset)
        target = torch.randint(min_val, max_val, lt.shape)

        return target.to(dtype=util.get_dtype(lt.dtype))

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}

        for i in range(len(self.op.inputs)):
            res[self.op.inputs[i].id] = self.__fill(i)
        return res


class MatMulOp(Op):
    transpose_a: bool
    transpose_b: bool

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.transpose_a = op.get_optional_attr("transpose_a", bool, False)
        self.transpose_b = op.get_optional_attr("transpose_b", bool, False)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, weights = ins[0], ins[1]

        src = src.transpose(-1, -2) if self.transpose_a else src
        weights = weights.transpose(-1, -2) if self.transpose_b else weights

        dst = torch.matmul(src, weights)

        if len(ins) > 2:
            bias = ins[2]
            dst = dst + bias
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        dt = util.get_dtype(self.op.outputs[offset].dtype)
        if dt == torch.float32:
            checker.threshold = 1e-6
        checker.zero_percent = 90.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}
        arg_rng: List[Dict[torch.dtype, Tuple[int, int]]] = [
            {
                torch.float32: (-64, 64),
                torch.bfloat16: (-4, 4),
                torch.float16: (-4, 4),
            },  # src
            {
                torch.float32: (-128, 128),
                torch.bfloat16: (-8, 8),
                torch.float16: (-2, 2),
            },  # wei
            {
                torch.float32: (-8, 8),
                torch.bfloat16: (-8, 8),
                torch.float16: (-8, 8),
            },  # bias
        ]

        # calculate density for src

        src_dt = util.get_dtype(self.op.inputs[0].dtype)
        wei_dt = util.get_dtype(self.op.inputs[1].dtype)
        src_min, src_max = arg_rng[0][src_dt]
        wei_min, wei_max = arg_rng[1][wei_dt]
        max_value = max(abs(src_min), abs(src_max)) * max(abs(wei_min), abs(wei_max))

        safe_digits: int = min(
            util.get_digits("f32"), util.get_digits(self.op.outputs[0].dtype)
        )
        safe_n_acc = (1 << safe_digits) // max_value
        k: int = (
            self.op.inputs[0].shape[0]
            if self.transpose_a
            else self.op.inputs[0].shape[1]
        )
        src_density = min(safe_n_acc / k, 1.0)

        for i in range(len(self.op.inputs)):
            lt = self.op.inputs[i]
            dtype = util.get_dtype(lt.dtype)
            arg_min, arg_max = arg_rng[i][dtype]

            util.torch_seed(1, i)
            density = 1.0 if i else src_density
            value = torch.bernoulli(torch.full(lt.shape, density)) * torch.randint(
                arg_min, arg_max, lt.shape
            )

            util.torch_seed(1, i)
            while value.flatten()[0] <= 0:
                value.flatten()[0] = torch.randint(arg_min, arg_max + 1, size=[1])[
                    0
                ].item()
            res[lt.id] = value.to(dtype=dtype)
        return res


class ReorderOp(Op):
    reorder: List[int]

    def __init__(self, op: gapi.Op):
        super().__init__(op)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        outs.append(ins[0])

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.zero_percent = 80.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]

        c_min, c_max = util.get_type_range(lt.dtype)
        gen: torch.Tensor = torch.tensor(
            [c_max, c_min, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 16.0, 64.0]
        )
        return {
            lt.id: gen.repeat((lt.nelem() + 9) // 10)[: lt.nelem()]
            .reshape(lt.shape)
            .to(util.get_dtype(lt.dtype))
        }


class SelectOp(Op):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.auto_broadcast = op.get_optional_attr("auto_broadcast", str, "numpy")

    def check_auto_broadcast(self, x: torch.Tensor, y: torch.Tensor):
        if self.auto_broadcast == "none" and x.shape != y.shape:
            raise Exception("shape mismatch %s and %s" % (x.shape, y.shape))

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        cond, src_0, src_1 = ins[0], ins[1], ins[2]
        self.check_auto_broadcast(cond, src_0)
        self.check_auto_broadcast(cond, src_1)
        self.check_auto_broadcast(src_0, src_1)

        dst = torch.where(cond.to(torch.bool), src_0, src_1)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.zero_percent = 100.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}
        for i in range(len(self.op.inputs)):
            f_min: int = 0
            f_max: int = 2 if i == 0 else 1
            lt = self.op.inputs[i]
            res[lt.id] = self.generic_fill(lt, f_min, f_max)
        return res


class StaticReshapeOp(Op):
    shape: List[int]
    special_zero: bool

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.shape = op.get_required_attr("shape", list)
        self.special_zero = op.get_required_attr("special_zero", bool)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        shape: List[int] = self.shape.copy()
        src = ins[0]

        if self.special_zero:
            for i in range(len(shape)):
                if shape[i] == 0:
                    shape[i] = src.shape[i]
        dyn_dim: int | None = None
        nelem: int = 1
        for i in range(len(shape)):
            if shape[i] == -1:
                if dyn_dim is not None:
                    dyn_dim = i
                else:
                    raise Exception("multiple -1 dimension found in shape\n")
            else:
                nelem *= shape[i]
        if dyn_dim is not None:
            shape[dyn_dim] = src.nelement() // nelem
        dst = src.reshape(shape)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.zero_percent = 100.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        res: Dict[int, torch.Tensor] = {lt.id: self.generic_fill(lt)}
        return res


class StaticTransposeOp(Op):
    reorder: List[int]

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.order = op.get_required_attr("order", list)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        dst = src.permute(self.order)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.zero_percent = 100.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        res: Dict[int, torch.Tensor] = {lt.id: self.generic_fill(lt)}
        return res


class LayerNormOp(Op):
    keep_stats: bool
    begin_norm_axis: int
    use_affine: bool
    epsilon: float

    beta: torch.Tensor | None = None

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.keep_stats = op.get_optional_attr("keep_stats", bool, True)
        self.begin_norm_axis = op.get_optional_attr("begin_norm_axis", int, -1)
        self.use_affine = op.get_optional_attr("use_affine", bool, True)
        self.epsilon = op.get_optional_attr("epsilon", float, 1e-5)

        ndim: int = len(op.inputs[0].shape)
        self.begin_norm_axis = (self.begin_norm_axis + ndim) % ndim

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        gamma = ins[1] if len(ins) == 3 else None
        beta = ins[2] if len(ins) == 3 else None

        begin_norm_axis = self.begin_norm_axis
        normalized_shape = [src.shape[begin_norm_axis]]
        normalized_dim = [begin_norm_axis]

        while begin_norm_axis != -1 and begin_norm_axis != src.ndim - 1:
            begin_norm_axis += 1
            normalized_shape.append(src.shape[begin_norm_axis])
            normalized_dim.append(begin_norm_axis)

        if self.use_affine:
            if gamma is None or beta is None:
                raise Exception(
                    "gamma and beta should be provided if use_affine is true"
                )
            gamma_, beta_ = gamma.reshape(normalized_shape), beta.reshape(
                normalized_shape
            )
        else:
            gamma_, beta_ = None, None

        # pytorch not support fp16 lnorm, cast it to fp32 before calculation
        if src.dtype == torch.float16:
            src_ = src.to(torch.float32)
            dst = torch.nn.functional.layer_norm(
                src_, normalized_shape, weight=gamma_, bias=beta_, eps=self.epsilon
            )
        else:
            dst = torch.nn.functional.layer_norm(
                src, normalized_shape, weight=gamma_, bias=beta_, eps=self.epsilon
            )

        outs.append(dst)
        if self.keep_stats:
            # mean, variance = src.mean(normalized_dim), src.var(normalized_dim)
            mean, variance = src.mean(normalized_dim), (
                src - src.mean(normalized_dim, keepdim=True)
            ).square().mean(normalized_dim)
            outs.append(mean)
            outs.append(variance)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        safe_digits: int = max(
            0, util.get_digits("f32") - util.get_digits(self.op.outputs[0].dtype)
        )

        checker.threshold = 5e-7 * (1 << safe_digits) if offset == 0 else 0
        if self.op.outputs[0].dtype == "u8":
            checker.zero_percent = 60.0

        if offset == 0 and self.beta is not None:  # use beta

            def layernorm_checker(
                ref: torch.Tensor,
                res: torch.Tensor,
                abs_diff: torch.Tensor,
                rel_diff: torch.Tensor,
            ) -> torch.Tensor:
                abs_ref = torch.abs(ref)
                norm_denorm = torch.where(
                    abs_ref > torch.finfo(torch.float32).smallest_normal, abs_ref, 1.0
                )
                abs_ref_delta = torch.abs(ref - self.beta)
                maybe_cancel_error = (abs_ref_delta / norm_denorm) > 1.0

                diff_ax = torch.abs((ref - self.beta) - (res - self.beta))
                rel_diff_ax = diff_ax / torch.where(
                    abs_ref_delta > torch.finfo(torch.float32).smallest_normal,
                    abs_ref_delta,
                    1.0,
                )
                return maybe_cancel_error & (rel_diff_ax <= checker.threshold)

            checker.customized_checker = layernorm_checker
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:

        n_channel: int = reduce(
            operator.mul, self.op.inputs[0].shape[self.begin_norm_axis :]
        )
        exact_bits: int = util.get_digits(self.op.inputs[0].dtype)
        free_bits: int = int((exact_bits - math.log2(float(n_channel))) / 2 - 1)
        want_flex_bits: int = int(min(6, exact_bits / 2))
        check_alg: int = (
            1 if free_bits >= 3 else (2 if want_flex_bits == exact_bits / 2 else 0)
        )
        flex_bits: int = (
            min(exact_bits, free_bits) if check_alg == 1 else want_flex_bits
        )
        density: float = (
            (1.0 * (1 << (exact_bits - 2 * flex_bits)) / n_channel)
            if check_alg == 0
            else 1.0
        )
        flex_mask: int = (1 << flex_bits) - 1
        flag: int = 6 if self.use_affine and len(self.op.inputs) == 3 else 0

        res: Dict[int, torch.Tensor] = {
            self.op.inputs[0].id: self.fill_src(
                self.op.inputs[0], check_alg, density, flex_mask, flex_bits, flag
            )
        }
        if self.use_affine:
            res[self.op.inputs[1].id] = self.fill_gamma(self.op.inputs[1], flag)
            res[self.op.inputs[2].id] = self.fill_gamma(self.op.inputs[2], flag)
        return res

    def fill_src(
        self,
        lt: gapi.LogicalTensor,
        check_alg: int,
        density: float,
        flex_mask: int,
        flex_bits: int,
        flag: int,
    ) -> torch.Tensor:
        dtype = util.get_dtype(lt.dtype)
        coeff: float = 0.25 if dtype.is_floating_point else 1.0

        n_channel: int = reduce(operator.mul, lt.shape[self.begin_norm_axis :])
        nelem_per_c: int = reduce(operator.mul, lt.shape[: self.begin_norm_axis])

        # calculate the mean
        if check_alg or flag:
            index: torch.Tensor = torch.arange(n_channel, dtype=torch.int)
            mean = coeff * (1 << (index % 7))
        else:
            mean = torch.zeros(size=[n_channel])

        util.torch_seed()
        if check_alg == 2:
            is_even = torch.arange(n_channel, dtype=torch.int) % 2 == 0
            sign = torch.where(is_even, 1.0, -1.0)

            prob_tensor = torch.full((nelem_per_c, n_channel), 0.5)
            bernoulli_tensor = torch.bernoulli(prob_tensor)

            bigger_val = torch.where(is_even, bernoulli_tensor, 0.0)
            value_shift = sign * coeff
            value = mean + value_shift + 3.0 * bigger_val * value_shift

        else:
            grid_elem, grid_channel = torch.meshgrid(
                torch.arange(nelem_per_c, dtype=torch.int64),
                torch.arange(n_channel, dtype=torch.int64),
            )

            l = grid_channel + grid_elem * 239 * 2

            gen = (l // 2 * 1637) & flex_mask
            sign = torch.where(l % 2 == 0, 1.0, -1.0)
            value = sign * gen / (1 << flex_bits)

            if check_alg == 0:
                value = mean * (value + 1.0)
            if flag & 1:
                value = (l % 65) - 32

            if check_alg == 0:
                mask = util.flip_coin(l // 2 * 257, density)
                if mask is torch.Tensor:
                    value = torch.where(mask, value, 0.0)
                else:
                    raise Exception("Cannot determine density")
        return value.reshape(lt.shape).to(dtype=dtype)

    def fill_gamma(self, lt: gapi.LogicalTensor, flag: int) -> torch.Tensor:
        value = 1 << (torch.arange(lt.nelem(), dtype=torch.int) % 7)
        if flag & 1 == 0:
            value = value / 8.0
        return value.to(dtype=util.get_dtype(lt.dtype))

    def fill_beta(self, lt: gapi.LogicalTensor, flag: int) -> torch.Tensor:
        index = torch.arange(lt.nelem(), dtype=torch.int)
        value = (index % 3 - 1) * (1 << (index % 7))
        if flag & 1 == 0:
            value = value / 512.0
        # keep beta for comparison
        self.beta = value.to(dtype=util.get_dtype(lt.dtype))
        return self.beta
