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

import ctypes
from typing import Any, List

import gc_mlir.ir
import torch
from gc_mlir.dialects import func

# only python 3.11 support
# from typing import Self


def get_entry(module: gc_mlir.ir.Module, entry: str = '"entry"') -> func.FuncOp:
    for op in module.operation.opview.regions[0].blocks[0].operations:
        if str(op.name) == entry:
            return op
    raise Exception(f"entry function {entry} is not found at the top level")


# calling python binding consumes a lot of time e.g. get_name()
# we need to cache some result to avoid duplicate call
class MLIRCache:
    # operand name cache
    opr: List[str]
    # result name cache
    res: List[str]
    # argument name cache
    arg: List[str]
    # next hierarchy
    next = []  # List[Self]

    def __init__(self):
        self.opr = []
        self.res = []
        self.arg = []
        self.next = []


def dtype_to_ctype(dtype: torch.dtype):
    if dtype == torch.float32:
        return ctypes.c_float
    elif dtype == torch.float64:
        return ctypes.c_double
    elif dtype == torch.int32:
        return ctypes.c_int
    elif dtype == torch.int64:
        return ctypes.c_longlong
    elif dtype == torch.uint8:
        return ctypes.c_ubyte
    elif dtype == torch.int8:
        return ctypes.c_byte
    elif dtype == torch.int16 or dtype == torch.bfloat16 or torch.float16:
        return ctypes.c_short
    elif dtype == torch.bool:
        return ctypes.c_bool
    else:
        raise ValueError(f"Unsupported torch dtype: {dtype}")


def str_to_mlir_dtype(ctx: gc_mlir.ir.Context, dtype: str) -> gc_mlir.ir.Type:
    if dtype == "f32":
        return gc_mlir.ir.F32Type.get(ctx)
    elif dtype == "f64":
        return gc_mlir.ir.F64Type.get(ctx)
    elif dtype == "f16":
        return gc_mlir.ir.F16Type.get(ctx)
    elif dtype == "bf16":
        return gc_mlir.ir.BF16Type.get(ctx)
    elif dtype == "u8":
        return gc_mlir.ir.IntegerType.get_unsigned(8, ctx)
    elif dtype == "s8":
        return gc_mlir.ir.IntegerType.get_signed(8, ctx)
    elif dtype == "boolean":
        return gc_mlir.ir.IntegerType.get_unsigned(1, ctx)
    elif dtype == "f8_e4m3":
        return gc_mlir.ir.Float8E4M3FNType.get(ctx)
    elif dtype == "f8_e5m2":
        return gc_mlir.ir.Float8E5M2Type.get(ctx)
    elif dtype == "s32":
        return gc_mlir.ir.IntegerType.get_signed(32, ctx)
    else:
        raise Exception(f"data type not support: {dtype}")


def str_to_mlir_typed_attr(
    ctx: gc_mlir.ir.Context, dtype: str, value: Any
) -> gc_mlir.ir.Attribute:
    mlir_dtype = str_to_mlir_dtype(ctx, dtype)
    if dtype in ["f32", "f64", "bf16", "f16", "f8_e4m3", "f8_e5m2"]:
        return gc_mlir.ir.FloatAttr.get(mlir_dtype, value)
    elif dtype in ["u8", "s8", "s32"]:
        return gc_mlir.ir.IntegerAttr.get(mlir_dtype, value)
    elif dtype == "boolean":
        return gc_mlir.ir.BoolAttr.get(value)
    else:
        raise Exception(f"data type not support: {dtype}")
