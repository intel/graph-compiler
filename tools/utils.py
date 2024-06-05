################################################################################
# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################

import ctypes
from typing import List

import ml_dtypes
import numpy as np
from enhanced_np_to_memref import (
    BF16,
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
)
from gc_mlir import ir
from gc_mlir.dialects import arith, func, memref

MLIR_TYPE_TO_NUMPY_TYPE = {
    "bf16": ml_dtypes.bfloat16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i32": np.int32,
    "i64": np.int64,
}

MLIR_TYPE_TO_C_TYPE = {
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i32": ctypes.c_int,
    "i8": ctypes.c_byte,
    "bf16": BF16,
}


def emit_nano_time() -> func.FuncOp:
    nanoTime = func.FuncOp(
        "nanoTime", ([], [ir.IntegerType.get_signless(64)]), visibility="private"
    )
    nanoTime.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nanoTime


def emit_benchmark_wrapped_main_func(
    kernel_func: func.FuncOp, timer_func: func.FuncOp
) -> func.FuncOp:
    memref_of_i64_type = ir.MemRefType.get([1], ir.IntegerType.get_signless(64))
    wrapped_func_name = "wrapped_main"
    assert wrapped_func_name != str(
        kernel_func.name
    ), "wrapped function name should be different from kernel function name"
    wrapped_func = func.FuncOp(
        wrapped_func_name,
        (kernel_func.arguments.types + [memref_of_i64_type], kernel_func.type.results),
        visibility="public",
    )
    wrapped_func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(wrapped_func.add_entry_block()):
        timer_buffer = wrapped_func.arguments[-1]
        start = func.CallOp(timer_func, [])
        call_op = func.CallOp(
            kernel_func,
            list(wrapped_func.arguments[:-1]),
        )
        end = func.CallOp(timer_func, [])
        time_taken = arith.SubIOp(end, start)
        zero = arith.ConstantOp.create_index(0)
        memref.StoreOp(time_taken, timer_buffer, [zero])
        func.ReturnOp(call_op.results)
    return wrapped_func




def np_args_to_mlir_args(np_args: List[np.ndarray]) -> List:
    mlir_args = []
    for arg in np_args:
        mlir_args.append(
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
        )
    return mlir_args


def get_mlir_args(module: ir.Module, entry: str, np_args: List[np.ndarray]):
    f = get_kernel_func_from_module(module, entry)
    compiled_func_args = []
    for res in f.type.results:
        compiled_func_args.append(
            ctypes.pointer(
                ctypes.pointer(
                    make_nd_memref_descriptor(
                        len(res.shape), MLIR_TYPE_TO_C_TYPE[str(res.element_type)]
                    )()
                )
            )
        )
    for arg in np_args:
        compiled_func_args.append(
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
        )
    return compiled_func_args


def mlir_type(s, ctx):
    type_mapping = {
        "f32": ir.F32Type.get(ctx),
        "f64": ir.F64Type.get(ctx),
        "bf16": ir.BF16Type.get(ctx),
        "i32": ir.IntegerType.get_signed(32),
        "i8": ir.IntegerType.get_signed(8),
    }
    return type_mapping[s]


def make_tensor(tensor_type):
    return np.zeros(
        tensor_type.shape, MLIR_TYPE_TO_NUMPY_TYPE[str(tensor_type.element_type)]
    )


def get_kernel_func_from_module(
    module: ir.Module, func_name: str = "main_entry"
) -> func.FuncOp:
    assert (
        len(module.operation.regions) == 1
    ), "Expected kernel module to have only one region"
    assert (
        len(module.operation.regions[0].blocks) == 1
    ), "Expected kernel module to have only one block"
    for f in module.operation.regions[0].blocks[0].operations:
        if type(f) is func.FuncOp and str(f.name).strip('"') == func_name:
            return f
    raise ValueError("can not find the entry function")


def get_default_passes():
    passes = """
        any(gc-cpu-pipeline)
    """
    return passes


def to_int_vector(s: str) -> List[int]:
    if not s or len(s) == 0:
        return []
    return [int(i) for i in s.strip().split("x")]


def to_bool_vector(s: str) -> List[bool]:
    if not s or len(s) == 0:
        return []
    return [bool(i) for i in s.strip().split("x")]


def load_mlir_from_path(path: str) -> str:
    with open(path, "r") as file:
        content = file.read()
    return content
