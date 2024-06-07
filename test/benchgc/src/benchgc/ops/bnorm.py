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
from .op import Op, to_ncx_if_needed, to_nxc_if_needed
import math


class BatchNormOp(Op):
    data_format: str
    epsilon: float
    beta: torch.Tensor | None = None

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.data_format = op.get_optional_attr("data_format", str, "NXC")
        self.epsilon = op.get_required_attr("epsilon", float)

    def fill_mean(self, lt: gapi.LogicalTensor, flag: int) -> torch.Tensor:
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        if flag & 1 == 0:
            return torch.zeros([lt.nelem()], dtype=dtype)
        else:
            index: torch.Tensor = torch.arange(lt.nelem(), dtype=torch.int)
            value: torch.Tensor = torch.pow(torch.Tensor([2]), index % 7 - 2)
            return value.to(dtype=dtype)

    def fill_src(
        self, lt: gapi.LogicalTensor, mean: torch.Tensor, flag: int
    ) -> torch.Tensor:

        if self.data_format == "NCX":
            c: int = lt.shape[1]
            xshape: List[int] = lt.shape[2:]
        else:
            c: int = lt.shape[-1]
            xshape: List[int] = lt.shape[1:-1]

        xsize: int = lt.nelem() // lt.shape[0] // c
        n_grid, c_grid = torch.meshgrid(
            torch.arange(lt.shape[0]) * xsize, torch.arange(c) * 239 * 2
        )
        # shape: n x c x 1...1
        l_base = (n_grid + c_grid).reshape([lt.shape[0], c] + [1] * len(xshape))

        # shape: 1 x 1 x [xshape]
        x = torch.arange(xsize).reshape([1, 1] + xshape)

        l = l_base + x
        nelem_per_c: int = lt.nelem() // c
        if flag & 1:
            value = (l % 65) - 32
        else:
            exact_bits: int = util.get_digits(lt.dtype)
            free_bits: int = (
                exact_bits - math.ceil(math.log2(float(nelem_per_c)))
            ) // 2 - 1
            want_flex_bits: int = min(6, exact_bits // 2)
            flex_bits: int = (
                min(exact_bits, free_bits) if free_bits > 3 else want_flex_bits
            )
            flex_mask: int = 1 << flex_bits - 1
            gen = (l // 2 * 1637) & flex_mask
            sgn = (l & 1) * 2 - 1
            value = (sgn * gen / (1 << flex_bits) + 1) * mean.reshape(
                [1, c] + [1] * len(xshape)
            )
            if nelem_per_c % 2 == 1:
                # remove the last value for each channel
                value = torch.split(value, value.shape[-1] - 1, -1)[0]
                mean = mean.reshape([1, c] + [1] * len(xshape)).broadcast_to(
                    value.shape[:-1] + (1,)
                )
                # set the last value as referenced mean value
                value = torch.concat([value, mean], -1)

        if self.data_format == "NXC":
            perm: List[int] = [0]
            for i in range(2, value.ndim):
                perm.append(i)
            perm.append(1)
            value.permute(perm)

        dtype: torch.dtype = util.get_dtype(lt.dtype)
        return value.to(dtype=dtype)

    def fill_variance(
        self,
        lt: gapi.LogicalTensor,
        src: torch.Tensor,
        mean: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        c: int = lt.nelem()
        if flag & 1:
            index: torch.Tensor = torch.arange(c, dtype=torch.int)
            value: torch.Tensor = (index % 7) << 1
        elif self.data_format == "NXC":
            mean = mean.reshape([1] * (src.ndim - 1) + [c])
            diff_squared = (src - mean) ** 2
            # reduce the sum except the channel axis
            value = diff_squared.mean(
                dim=tuple(i for i in range(diff_squared.ndim - 1))
            )
        else:  # NCX
            mean = mean.reshape([1, c] + [1] * (src.ndim - 2))
            diff_squared = (src - mean) ** 2
            # reduce the sum except the channel axis
            value = diff_squared.mean(
                dim=tuple(i for i in range(diff_squared.ndim) if i != 1)
            )
        return value.to(dtype=dtype)

    def fill_gamma(self, lt: gapi.LogicalTensor, flag: int) -> torch.Tensor:
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        index = torch.arange(lt.nelem(), dtype=torch.int)
        value = 1 << (index % 7)
        if flag & 1 == 0:
            value = value / 8.0
        return value.to(dtype=dtype)

    def fill_beta(self, lt: gapi.LogicalTensor, flag: int) -> torch.Tensor:
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        index = torch.arange(lt.nelem(), dtype=torch.int)
        value = (index % 3 - 1) * (1 << (index % 7))
        if flag & 1 == 0:
            value = value / 512.0
        # keep the reference of beta for comparison
        self.beta = value.to(dtype=dtype)
        return self.beta

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        safe_digits: int = max(
            0, util.get_digits("f32") - util.get_digits(self.op.outputs[offset].dtype)
        )
        checker.threshold = 6e-7 * (1 << safe_digits) if offset == 0 else 0
        if self.op.outputs[offset].dtype == "bf16":
            checker.zero_percent = 99.0

        if offset == 0 and self.beta is not None:  # use beta

            def batchnorm_checker(
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

            checker.customized_checker = batchnorm_checker
        return checker


class BatchNormForwardTrainingOp(BatchNormOp):
    momentum: float

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.momentum = op.get_optional_attr("momentum", float, 0.1)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, mean, variance = ins[0], ins[1], ins[2]
        gamma = None if len(ins) == 3 else ins[3]
        beta = None if len(ins) == 3 else ins[4]

        ncx_src = to_ncx_if_needed(self.data_format, src)
        ncx_dst = torch.nn.functional.batch_norm(
            ncx_src,
            running_mean=None,
            running_var=None,
            weight=gamma,
            bias=beta,
            training=True,
            eps=self.epsilon,
        )
        dst = to_nxc_if_needed(self.data_format, ncx_dst)

        batch_dim = [0]
        for i in range(2, ncx_src.ndim):
            batch_dim.append(i)

        batch_mean = torch.mean(ncx_src, dim=batch_dim)
        batch_variance = torch.var(ncx_src, dim=batch_dim, unbiased=False)
        running_mean = batch_mean * (1.0 - self.momentum) + mean * self.momentum
        running_variance = (
            batch_variance * (1.0 - self.momentum) + variance * self.momentum
        )

        outs.append(dst)
        outs.append(running_mean)
        outs.append(running_variance)
        outs.append(batch_mean)
        outs.append(batch_variance)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}

        flag: int = 6 if len(self.op.inputs) != 3 else 0

        mean = self.fill_mean(self.op.inputs[1], flag)
        res[self.op.inputs[1].id] = mean
        src = self.fill_src(self.op.inputs[0], mean, flag)
        res[self.op.inputs[0].id] = src
        res[self.op.inputs[2].id] = self.fill_variance(
            self.op.inputs[2], src, mean, flag
        )
        if len(self.op.inputs) > 3:
            res[self.op.inputs[3].id] = self.fill_gamma(self.op.inputs[3], flag)
        if len(self.op.inputs) > 4:
            res[self.op.inputs[4].id] = self.fill_beta(self.op.inputs[4], flag)

        return res


class BatchNormInferenceOp(BatchNormOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, gamma, beta, mean, variance = ins[0], ins[1], ins[2], ins[3], ins[4]

        ncx_src = to_ncx_if_needed(self.data_format, src)

        shape_: List[int] = [1] * ncx_src.ndim
        shape_[1] = ncx_src.shape[1]

        ncx_dst = (ncx_src - mean.reshape(shape_)) / torch.sqrt(
            variance.reshape(shape_) + self.epsilon
        ) * gamma.reshape(shape_) + beta.reshape(shape_)
        dst = to_nxc_if_needed(self.data_format, ncx_dst)
        outs.append(dst)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}

        mean = self.fill_mean(self.op.inputs[3], 7)
        res[self.op.inputs[3].id] = mean
        src = self.fill_src(self.op.inputs[0], mean, 7)
        res[self.op.inputs[0].id] = src
        res[self.op.inputs[4].id] = self.fill_variance(self.op.inputs[4], src, mean, 7)
        res[self.op.inputs[1].id] = self.fill_gamma(self.op.inputs[1], 7)
        res[self.op.inputs[2].id] = self.fill_beta(self.op.inputs[2], 7)

        return res
