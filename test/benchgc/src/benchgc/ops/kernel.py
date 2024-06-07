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
from typing import List, Tuple, Dict
from .op import Op, to_ncx_if_needed, to_nxc_if_needed


class KernelOp(Op):
    strides: List[int]
    pads_begin: List[int]
    pads_end: List[int]
    auto_pad: str

    # kernel is a required attr in AvgPool, but not exist in Convolution
    # set this attr in per op init function
    kernel: List[int]
    # dilations is required in Convolution, optional in MaxPool, but not exist in AvgPool
    # set this attr in per op init function
    dilations: List[int]

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.strides = op.get_required_attr("strides", list)
        self.pads_begin = op.get_required_attr("pads_begin", list)
        self.pads_end = op.get_required_attr("pads_end", list)
        self.auto_pad = op.get_optional_attr("auto_pad", str, "none")

    # calculate the left/right padding on a specified dimension
    def pad(self, dim: int, data_dim: int) -> Tuple[int, int]:
        stride_dim = self.strides[dim]
        kernel_dim = self.kernel[dim]
        dilation_dim = self.dilations[dim]

        auto_pad = self.auto_pad.lower()
        if auto_pad == "none":
            return (self.pads_begin[dim], self.pads_end[dim])
        elif auto_pad == "valid":
            return (0, 0)
        elif auto_pad == "same_upper":
            padlr = (
                data_dim * (stride_dim - 1)
                + dilation_dim * (kernel_dim - 1)
                - stride_dim
                + 1
            )
            return (padlr - padlr // 2, padlr // 2)
        elif auto_pad == "same_lower":
            padlr = (
                data_dim * (stride_dim - 1)
                + dilation_dim * (kernel_dim - 1)
                - stride_dim
                + 1
            )
            return (padlr // 2, padlr - padlr // 2)
        raise Exception("unsupported auto_pad: %s" % auto_pad)


def fill_pool(lt: gapi.LogicalTensor) -> torch.Tensor:
    dtype: torch.dtype = util.get_dtype(lt.dtype)
    arg_limits: Dict[torch.dtype, Tuple[int, int]] = {
        torch.float32: (-2048, 2048),
        torch.bfloat16: (-32, 32),
        torch.float16: (-32, 32),
    }
    util.torch_seed()
    target = torch.randint(arg_limits[dtype][0], arg_limits[dtype][1] + 1, [lt.nelem()])
    # make sure the first element is not negative
    if target[0] <= 0.0:
        while target[0] <= 0.0:
            target[0] = torch.randint(
                arg_limits[dtype][0], arg_limits[dtype][1], size=(1,)
            )[0].item()

    return target.reshape(lt.shape).to(dtype=dtype)


class AvgPoolOp(KernelOp):
    data_format: str
    exclude_pad: bool
    rounding_type: str

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.data_format = op.get_optional_attr("data_format", str, "NXC")
        self.kernel = op.get_required_attr("kernel", list)
        self.exclude_pad = op.get_required_attr("exclude_pad", bool)
        self.rounding_type = op.get_optional_attr("rounding_type", str, "floor")
        self.dilations = [1] * len(self.kernel)

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        ncx_src = to_ncx_if_needed(self.data_format, src)

        if ncx_src.ndim == 3:
            pre_pad = self.pad(-1, ncx_src.shape[-1])
            rt_pad_l = (pre_pad[0],)
            rt_pad_r = (pre_pad[1],)
            f = torch.nn.functional.avg_pool1d
        elif ncx_src.ndim == 4:
            pre_pad = self.pad(-1, ncx_src.shape[-1]) + self.pad(-2, ncx_src.shape[-2])
            rt_pad_l = (pre_pad[2], pre_pad[0])
            rt_pad_r = (pre_pad[3], pre_pad[1])
            f = torch.nn.functional.avg_pool2d
        elif ncx_src.ndim == 5:
            pre_pad = (
                self.pad(-1, ncx_src.shape[-1])
                + self.pad(-2, ncx_src.shape[-2])
                + self.pad(-3, ncx_src.shape[-3])
            )
            rt_pad_l = (pre_pad[4], pre_pad[2], pre_pad[0])
            rt_pad_r = (pre_pad[5], pre_pad[3], pre_pad[1])
            f = torch.nn.functional.avg_pool3d
        else:
            raise Exception("unsupport avgpool src dimension")
        if not self.exclude_pad:
            pad = (0,) * (ncx_src.ndim - 2)
            ncx_src = torch.nn.functional.pad(ncx_src, pre_pad)
        elif rt_pad_l == rt_pad_r:
            pad = rt_pad_l
        else:
            raise Exception(
                "pytorch do not support asymmetric padding in avgpool if exclude pad\n"
            )

        if ncx_src.dtype == torch.float16:
            # avgpool_2d not implented for Half in pytorch
            ncx_src_ = ncx_src.to(torch.float32)
        else:
            ncx_src_ = ncx_src
        ncx_dst = f(
            ncx_src_,
            kernel_size=self.kernel,
            stride=self.strides,
            padding=pad,
            ceil_mode=(self.rounding_type == "ceil"),
            count_include_pad=not self.exclude_pad,
        )
        dst = to_nxc_if_needed(self.data_format, ncx_dst)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.threshold *= 10.0
        checker.zero_percent = 99.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        return {lt.id: fill_pool(lt)}


class ConvolutionOp(KernelOp):
    data_format: str
    groups: int
    weights_format: str

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.data_format = op.get_optional_attr("data_format", str, "NXC")
        self.dilations = op.get_required_attr("dilations", list)
        self.groups = op.get_optional_attr("groups", int, 1)
        self.weights_format = op.get_optional_attr("weights_format", str, "XIO")

    def to_oix_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if self.weights_format == "XIO":
            perm: List[int] = [-1, -2]
            for i in range(x.ndim - 2):
                perm.append(i)
            return x.permute(perm)
        else:
            return x

    def to_xio_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if self.weights_format == "XIO":
            perm: List[int] = []
            for i in range(x.ndim - 2):
                perm.append(i + 2)
            perm.append(1)
            perm.append(0)
            return x.permute(perm)
        else:
            return x

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src, weights = ins[0], ins[1]
        bias = None if len(ins) < 3 else ins[2]

        ncx_src = to_ncx_if_needed(self.data_format, src)
        oix_weights = self.to_oix_if_needed(weights)

        # set in attr for pad calculation
        self.kernel = list(oix_weights.shape[2:])

        if ncx_src.ndim == 3:
            pre_pad = self.pad(-1, ncx_src.shape[-1])
            f = torch.nn.functional.conv1d
        elif ncx_src.ndim == 4:
            pre_pad = self.pad(-1, ncx_src.shape[-1]) + self.pad(-2, ncx_src.shape[-2])
            f = torch.nn.functional.conv2d
        elif ncx_src.ndim == 5:
            pre_pad = (
                self.pad(-1, ncx_src.shape[-1])
                + self.pad(-2, ncx_src.shape[-2])
                + self.pad(-3, ncx_src.shape[-3])
            )
            f = torch.nn.functional.conv3d
        else:
            raise Exception("unsupport convolution src dimension")

        ncx_src = torch.nn.functional.pad(ncx_src, pre_pad)

        ncx_dst = f(
            ncx_src,
            oix_weights,
            bias,
            stride=self.strides,
            dilation=self.dilations,
            groups=self.groups,
        )

        dst = to_nxc_if_needed(self.data_format, ncx_dst)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        # TODO: please relax the threshold if gc uses wino algorithm
        checker.threshold = 0.0
        mistrust: bool = self.op.outputs[offset].dtype == "u8"

        # check the padding in output
        if self.data_format == "NCX":
            for i in range(2, len(self.op.inputs[0].shape)):
                mistrust = mistrust or (
                    self.op.inputs[0].shape[i] < self.op.outputs[0].shape[i]
                )
        else:
            for i in range(1, len(self.op.inputs[0].shape) - 1):
                mistrust = mistrust or (
                    self.op.inputs[0].shape[i] < self.op.outputs[0].shape[i]
                )

        checker.zero_percent = 0.85 if mistrust else 0.7
        return checker

    def __fill(
        self, lt: gapi.LogicalTensor, density: float, rng: Tuple[int, int]
    ) -> torch.Tensor:
        shape: List[int] = lt.shape
        dtype: torch.dtype = util.get_dtype(lt.dtype)
        target = torch.empty(size=shape, dtype=torch.float32)
        target = target.view(-1)

        arg_min, arg_max = rng

        util.torch_seed()

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

    def fill_data(self) -> Dict[int, torch.Tensor]:
        res: Dict[int, torch.Tensor] = {}

        # calculate density for src filling
        src, wei = self.op.inputs[0], self.op.inputs[1]
        src_dtype, wei_dtype = util.get_dtype(src.dtype), util.get_dtype(wei.dtype)
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

        max_value: int = src_rng[1] * wei_rng[1]
        safe_digits: int = min(
            util.get_digits("f32"), util.get_digits(self.op.outputs[0].dtype)
        )
        safe_n_acc = (1 << safe_digits) // max_value
        n_acc: int
        if self.weights_format == "XIO":
            n_acc = wei.nelem() // wei.shape[-1]
        else:
            n_acc = wei.nelem() // wei.shape[0]
        src_density = min(1.0, safe_n_acc / n_acc)

        # fill data for src/wei/bias
        res[src.id] = self.__fill(src, src_density, src_rng)
        res[wei.id] = self.__fill(wei, 1.0, wei_rng)
        if len(self.op.inputs) > 2:
            bias = self.op.inputs[2]
            bias_dtype = util.get_dtype(bias.dtype)
            bias_rng: Tuple[int, int] = {
                torch.float32: (-8, 8),
                torch.bfloat16: (-8, 8),
                torch.float16: (-8, 8),
            }[bias_dtype]
            res[bias.id] = self.__fill(bias, 1.0, bias_rng)
        return res


class MaxPoolOp(KernelOp):
    rounding_type: str
    data_format: str

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.data_format = op.get_optional_attr("data_format", str, "NXC")
        self.kernel = op.get_required_attr("kernel", list)
        self.rounding_type = op.get_optional_attr("rounding_type", str, "floor")
        self.dilations = op.get_optional_attr("dilations", list, [1] * len(self.kernel))

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        ncx_src = to_ncx_if_needed(self.data_format, src)

        if ncx_src.ndim == 3:
            pre_pad = self.pad(-1, ncx_src.shape[-1])
            f = torch.nn.functional.max_pool1d
        elif ncx_src.ndim == 4:
            pre_pad = self.pad(-1, ncx_src.shape[-1]) + self.pad(-2, ncx_src.shape[-2])
            f = torch.nn.functional.max_pool2d
        elif ncx_src.ndim == 5:
            pre_pad = (
                self.pad(-1, ncx_src.shape[-1])
                + self.pad(-2, ncx_src.shape[-2])
                + self.pad(-3, ncx_src.shape[-3])
            )
            f = torch.nn.functional.max_pool3d
        else:
            raise Exception("unsupport maxpool src dimension")

        # pad the min value of the data type
        ncx_src = torch.nn.functional.pad(
            ncx_src, pre_pad, "constant", torch.finfo(ncx_src.dtype).min
        )

        ncx_dst = f(
            ncx_src,
            kernel_size=self.kernel,
            stride=self.strides,
            ceil_mode=(self.rounding_type == "ceil"),
        )
        dst = to_nxc_if_needed(self.data_format, ncx_dst)
        outs.append(dst)

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.threshold *= 0.0
        checker.zero_percent = 99.0
        return checker

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        return {lt.id: fill_pool(lt)}
