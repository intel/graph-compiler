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
from typing import List, Callable, Dict, Any
from .op import Op


class ReduceOp(Op):
    axes: List[int]
    keep_dims: bool

    fn: Callable[[torch.Tensor], torch.Tensor]
    fn_args: Dict[str, Any]

    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.axes = op.get_optional_attr("axes", list, [])
        ndim: int = len(op.inputs[0].shape)
        self.axes = list(map(lambda d: (d + ndim) % ndim, self.axes))
        self.keep_dims = op.get_optional_attr("keep_dims", bool, False)
        self.fn_args = {"dim": self.axes, "keepdim": self.keep_dims}

    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        dst = self.fn(src, **self.fn_args)
        outs.append(dst)

    def fill_reduce(self, lt: gapi.LogicalTensor) -> torch.Tensor:
        shape: List[int] = lt.shape
        sdtype: torch.dtype = util.get_dtype(lt.dtype)
        ddtype: torch.dtype = util.get_dtype(self.op.outputs[0].dtype)
        assert sdtype == ddtype
        kind: str = self.op.kind

        src_dims: List[int] = shape
        dst_dims: List[int] = self.op.outputs[0].shape
        assert len(src_dims) == len(dst_dims)
        nelems_to_reduce: int = 1
        for axis in self.axes:
            nelems_to_reduce *= lt.shape[axis]

        safe_to_reduce_elems: int = util.get_problem_bounds(kind, sdtype)[0]
        if safe_to_reduce_elems == -1:
            safe_to_reduce_elems = nelems_to_reduce

        neutral_value: float = 1.0 if kind == "ReduceMul" else 0.0
        shift: float = (
            1.0
            if (
                kind == "ReduceMean"
                or kind == "ReduceMin"
                and not sdtype.is_signed
                and not ddtype.is_signed
            )
            else 0.0
        )

        value_range: int = util.get_problem_bounds(kind, sdtype)[1]

        is_mul_fp: bool = kind == "ReduceMul" and sdtype.is_floating_point
        min_range: int = -value_range if is_mul_fp else 1

        index = torch.arange(lt.nelem()).reshape(lt.shape)

        util.torch_seed()
        value = torch.randint(min_range, value_range + 1, size=lt.shape)
        if is_mul_fp:
            value = torch.pow(2, value)
        if sdtype.is_signed:  # random choose positive or negative
            value = torch.where(torch.BoolTensor(size=lt.shape), value, -value)

        non_neutral_mask = util.flip_coin(
            index,
            torch.full(
                lt.shape, safe_to_reduce_elems / nelems_to_reduce, dtype=torch.float32
            ),
        )
        if isinstance(non_neutral_mask, torch.Tensor):
            value = torch.where(non_neutral_mask, value, neutral_value)
        else:
            raise Exception("Flip coin failed when generate the reduce data filling")
        value = value + shift
        return value.to(dtype=sdtype)

    def fill_data(self) -> Dict[int, torch.Tensor]:
        lt = self.op.inputs[0]
        res: Dict[int, torch.Tensor] = {
            lt.id: self.fill_reduce(lt),
        }
        return res


class ReduceL1Op(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn = torch.linalg.vector_norm
        self.fn_args["ord"] = 1

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.threshold *= 5.0
        return checker


class ReduceL2Op(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.gc_op = "reduce_l2"
        self.fn = torch.linalg.vector_norm
        self.fn_args["ord"] = 2

    def checker(self, offset: int) -> util.Checker:
        checker = super().checker(offset)
        checker.threshold *= 5.0
        return checker


class ReduceMaxOp(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn = torch.amax


class ReduceMeanOp(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn = torch.mean


class ReduceMinOp(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn = torch.amin


class ReduceProdOp(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)

    # torch.prod can only reduce 1 dimension for each call
    # we need to reduce the axes one by one
    def ref(self, ins: List[torch.Tensor], outs: List[torch.Tensor]):
        src = ins[0]
        self.axes.sort(reverse=True)
        dst = src.clone()
        for axis in self.axes:
            dst = torch.prod(dst, dim=axis, keepdim=self.keep_dims)
        outs.append(dst)


class ReduceSumOp(ReduceOp):
    def __init__(self, op: gapi.Op):
        super().__init__(op)
        self.fn = torch.sum
