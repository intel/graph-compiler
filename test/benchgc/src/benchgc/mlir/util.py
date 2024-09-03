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
import os
from typing import Any, List

import cpuinfo
import torch
from gc_mlir import ir
from gc_mlir.dialects import arith, func, memref


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


def str_to_mlir_dtype(ctx: ir.Context, dtype: str) -> ir.Type:
    if dtype == "f32":
        return ir.F32Type.get(ctx)
    elif dtype == "f64":
        return ir.F64Type.get(ctx)
    elif dtype == "f16":
        return ir.F16Type.get(ctx)
    elif dtype == "bf16":
        return ir.BF16Type.get(ctx)
    elif dtype == "u8":
        return ir.IntegerType.get_unsigned(8, ctx)
    elif dtype == "s8":
        return ir.IntegerType.get_signed(8, ctx)
    elif dtype == "boolean":
        return ir.IntegerType.get_unsigned(1, ctx)
    elif dtype == "f8_e4m3":
        return ir.Float8E4M3FNType.get(ctx)
    elif dtype == "f8_e5m2":
        return ir.Float8E5M2Type.get(ctx)
    elif dtype == "s32":
        return ir.IntegerType.get_signed(32, ctx)
    else:
        raise Exception(f"data type not support: {dtype}")


def str_to_mlir_typed_attr(ctx: ir.Context, dtype: str, value: Any) -> ir.Attribute:
    mlir_dtype = str_to_mlir_dtype(ctx, dtype)
    if dtype in ["f32", "f64", "bf16", "f16", "f8_e4m3", "f8_e5m2"]:
        return ir.FloatAttr.get(mlir_dtype, value)
    elif dtype in ["u8", "s8", "s32"]:
        return ir.IntegerAttr.get(mlir_dtype, value)
    elif dtype == "boolean":
        return ir.BoolAttr.get(value)
    else:
        raise Exception(f"data type not support: {dtype}")


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


def get_kernel_func_from_module(
    module: ir.Module, func_name: str = "entry"
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


def attch_dlti(module: ir.Module):
    if "dlti.target_system_spec" in module.operation.attributes:
        return
    info = cpuinfo.get_cpu_info()
    from gc_mlir.dialects import cpuruntime
    cpuruntime.get_cpu_info()
    print(info)
    l1_data_cache_size = info.get("l1_data_cache_size")
    l2_cache_size = info.get("l2_cache_size")
    l3_cache_size = info.get("l3_cache_size")
    if "OMP_NUM_THREADS" not in os.environ:
        print("OMP_NUM_THREADS is not found, using 1 as default")
    num_threads = os.environ.get("OMP_NUM_THREADS", 1)
    flags = info.get("flags")
    max_vector_width = 64
    for flag in flags:
        if "avx512f" == flag:
            max_vector_width = max(512, max_vector_width)
        elif "avx2" == flag or "avx" == flag:
            max_vector_width = max(256, max_vector_width)
        elif "sse" in flag:
            max_vector_width = max(128, max_vector_width)

    dlti_template = f"""
    module attributes {{
        dlti.target_system_spec = #dlti.target_system_spec<
        "CPU": #dlti.target_device_spec<
            #dlti.dl_entry<"L1_cache_size_in_bytes", {l1_data_cache_size} : ui32>,
            #dlti.dl_entry<"L2_cache_size_in_bytes", {l2_cache_size} : ui64>,
            #dlti.dl_entry<"L3_cache_size_in_bytes", {l3_cache_size} : ui64>,
            #dlti.dl_entry<"num_threads", {num_threads} : i32>,
            #dlti.dl_entry<"max_vector_width", {max_vector_width} : i64>>
        >}} {{}}
    """
    print(dlti_template)
    with module.context:
        template_module = ir.Module.parse(dlti_template)
        module.operation.attributes["dlti.target_system_spec"] = (
            template_module.operation.attributes["dlti.target_system_spec"]
        )