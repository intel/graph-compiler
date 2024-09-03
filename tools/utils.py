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
from gc_mlir import ir
from gc_mlir.dialects import arith, func, memref
from gc_mlir.runtime.np_to_memref import (
    BF16,
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
)

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


def STR_TO_MLIR_TYPE(type: str, ctx: ir.Context):
    type_map = {
        "f32": ir.F32Type.get(ctx),
        "f64": ir.F64Type.get(ctx),
        "bf16": ir.BF16Type.get(ctx),
        "i32": ir.IntegerType.get_signed(32, ctx),
        "i8": ir.IntegerType.get_signed(8, ctx),
    }
    return type_map[type]


def emit_nano_time() -> func.FuncOp:
    """Emit a nanoTime function that returns the current time in nanoseconds."""
    nanoTime = func.FuncOp(
        "nanoTime", ([], [ir.IntegerType.get_signless(64)]), visibility="private"
    )
    nanoTime.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nanoTime


def emit_benchmark_wrapped_main_func(
    kernel_func: func.FuncOp, timer_func: func.FuncOp
) -> func.FuncOp:
    """Emit a wrapped main function that calls the kernel function and records the time taken."""
    memref_of_i64_type = ir.MemRefType.get([1], ir.IntegerType.get_signless(64))
    wrapped_func_name = "wrapped_main"
    assert wrapped_func_name != str(
        kernel_func.name
    ), "wrapped function name should be different from kernel function name"
    wrapped_func = func.FuncOp(
        wrapped_func_name,
        ([memref_of_i64_type] + kernel_func.arguments.types, kernel_func.type.results),
        visibility="public",
    )
    wrapped_func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(wrapped_func.add_entry_block()):
        timer_buffer = wrapped_func.arguments[0]
        start = func.CallOp(timer_func, [])
        call_op = func.CallOp(
            kernel_func,
            list(wrapped_func.arguments[1:]),
        )
        end = func.CallOp(timer_func, [])
        time_taken = arith.SubIOp(end, start)
        zero = arith.ConstantOp.create_index(0)
        memref.StoreOp(time_taken, timer_buffer, [zero])
        func.ReturnOp(call_op.results)
    return wrapped_func


def get_mlir_args(
    module: ir.Module,
    entry: str,
    np_args: List[np.ndarray],
    disable_results_to_params=False,
):
    """Convert numpy arrays to MLIR args and return a list of pointers to them"""
    f = get_kernel_func_from_module(module, entry)
    compiled_func_args = []
    if disable_results_to_params:
        assert len(np_args) == len(f.arguments), "input args mismatch"
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


def make_mlir_ndarray(tensor_type):
    import math
    shape = tensor_type.shape
    dtype = MLIR_TYPE_TO_NUMPY_TYPE[str(tensor_type.element_type)]
    dtypeSize = np.dtype(dtype).itemsize
    raw_size = math.prod(shape)
    size = int(raw_size + 64 / dtypeSize)
    raw = np.empty(size, dtype)
    rawptr, read_only_fla = raw.__array_interface__['data']
    offset = int((64 - rawptr % 64) / dtypeSize)
    result = raw[offset:offset+raw_size].reshape(shape)
    return result


def get_kernel_func_from_module(
    module: ir.Module, func_name: str = "main_entry"
) -> func.FuncOp:
    """Get the func op by the name from a module"""
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


def to_int_list(s: str) -> List[int]:
    """
    Parsing the cmd for list of int values

    Args:
        s (str): int values in cmd, example: 2x3x4

    Returns:
        List[int]: int values in list, example: [2, 3, 4]
    """
    if not s or len(s) == 0:
        return []
    return [int(i) for i in s.strip().split("x")]


def to_bool_list(s: str) -> List[bool]:
    """
    Parsing the cmd for list of bool values

    Args:
        s (str): bools in cmd, example: 1x0x1

    Returns:
        List[bool]: bools in list, example: [True, False, True]
    """
    if not s or len(s) == 0:
        return []
    return [bool(int(i)) for i in s.strip().split("x")]


def load_mlir_from_path(path: str) -> str:
    """Load MLIR content from path"""
    with open(path, "r") as file:
        content = file.read()
    return content
