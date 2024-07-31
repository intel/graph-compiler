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
import ctypes

from typing import List, Any
from benchgc.mlir.util import dtype_to_ctype, str_to_mlir_dtype
import gc_mlir.dialects.tensor
import gc_mlir.ir


# scalar should give a address
# map torch.Tensor -> memref
# map int address -> scalar value
def get_mlir_args(args: List[torch.Tensor | int]):
    mlir_args: List[Any] = []

    for arg in args:
        if isinstance(arg, torch.Tensor):
            mlir_args.append(ctypes.pointer(ctypes.pointer(get_md(arg))))
        else:
            mlir_args.append(ctypes.c_void_p(arg))

    return mlir_args


def get_md(tensor: torch.Tensor):
    if tensor.ndim == 0:

        class _0dMemrefDescriptor(ctypes.Structure):
            _fields_ = [
                ("allocated", ctypes.c_longlong),
                ("aligned", ctypes.POINTER(dtype_to_ctype(tensor.dtype))),
                ("offset", ctypes.c_longlong),
            ]

        md = _0dMemrefDescriptor()
    else:
        ctype_shape = ctypes.c_longlong * tensor.ndim
        ctype_strides = ctypes.c_longlong * tensor.ndim

        class _ndMemrefDescriptor(ctypes.Structure):
            _fields_ = [
                ("allocated", ctypes.c_longlong),
                ("aligned", ctypes.POINTER(dtype_to_ctype(tensor.dtype))),
                ("offset", ctypes.c_longlong),
                ("shape", ctype_shape),
                ("strides", ctype_strides),
            ]

        md = _ndMemrefDescriptor()
        md.shape = ctype_shape(*tensor.shape)
        md.strides = ctype_strides(*tensor.stride())

    md.allocated = tensor.data_ptr()
    md.aligned = ctypes.cast(
        ctypes.c_void_p(tensor.data_ptr()), ctypes.POINTER(dtype_to_ctype(tensor.dtype))
    )
    md.offset = ctypes.c_longlong(0)
    return md


class MLIRArg:
    dtype: str
    shape: List[int]

    scalar: bool

    def __init__(self) -> None:
        self.dtype = ""

    # md format:
    # 0d memref/tensor: 0xf32
    # nd memref/tensor: 2x3xf32
    # scalar: f32
    def set_md(self, md: str):
        splited: List[str] = md.split("x")
        self.dtype = splited[-1]
        self.shape = []

        for dim in splited[:-1]:
            self.shape.append(int(dim))
        self.set_scalar()

    def set_scalar(self):
        # use 0xf32 to represent memref<f32>
        # use f32 to represent f32
        if self.shape == [0]:
            self.shape = []
            self.scalar = False
        elif self.shape == []:
            self.scalar = True
        else:
            self.scalar = False

    def get_mlir_type(self, ctx: gc_mlir.ir.Context) -> gc_mlir.ir.Type:
        if self.shape == []:
            return str_to_mlir_dtype(ctx, self.dtype)
        else:
            return gc_mlir.ir.RankedTensorType.get(
                self.shape, str_to_mlir_dtype(ctx, self.dtype)
            )

    def get_ranked_tensor_type(
        self, ctx: gc_mlir.ir.Context
    ) -> gc_mlir.ir.RankedTensorType:
        return gc_mlir.ir.RankedTensorType.get(
            self.shape, str_to_mlir_dtype(ctx, self.dtype)
        )

    def get_empty_op(self, ctx: gc_mlir.ir.Context) -> gc_mlir.dialects.tensor.EmptyOp:
        if self.shape == []:
            raise Exception("shape is unknown")
        return gc_mlir.dialects.tensor.EmptyOp(
            self.shape, str_to_mlir_dtype(ctx, self.dtype)
        )
