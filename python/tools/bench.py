from copy import deepcopy
from time import sleep
from typing import Sequence
from gc_mlir import ir
from gc_mlir.passmanager import *
from gc_mlir.execution_engine import *
import ctypes
from gc_mlir.dialects import func, arith, memref
from gc_mlir import runtime
from timeit import timeit, repeat
import random
from gc_mlir.graph_compiler import GraphCompiler
from utils import (
    get_kernel_func_from_module,
    emit_nano_time,
    emit_benchmark_wrapped_main_func,
)
import numpy as np
import os


def py_timeit_bench(
    ctx: ir.Context,
    ir_module: ir.Module,
    entry_name: str,
    passes: str,
    mlir_args: list,
    shared_libs: Sequence,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    engine = GraphCompiler(
        shared_libs,
        passes,
    ).compile_and_jit(ir_module, ir_printing=ir_printing)
    func = engine.lookup(entry_name)
    packed_args = (ctypes.c_void_p * len(mlir_args))()
    for argNum in range(len(mlir_args)):
        packed_args[argNum] = ctypes.cast(mlir_args[argNum], ctypes.c_void_p)

    def run_bench(func, arg):
        func(arg)

    timeit(lambda: run_bench(func, packed_args), number=warm_up)
    total_time = timeit(lambda: run_bench(func, packed_args), number=repeat_time)
    return total_time * 1000 / repeat_time


def mlir_wrapper_bench(
    ctx: ir.Context,
    ir_module: ir.Module,
    entry_name: str,
    passes: str,
    mlir_args: list,
    shared_libs: Sequence,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    kernel_func = get_kernel_func_from_module(ir_module, entry_name)

    wrapper_module = ir_module
    with ir.InsertionPoint(wrapper_module.body):
        emit_benchmark_wrapped_main_func(kernel_func, emit_nano_time()) 
    complier = GraphCompiler(
        shared_libs,
        passes,
    )
    print(wrapper_module)
    engine = complier.compile_and_jit(wrapper_module, ir_printing=ir_printing)
    np_timers_ns = np.array([0], dtype=np.int64)
    arg2_memref_ptr = ctypes.pointer(
        ctypes.pointer(runtime.get_ranked_memref_descriptor(np_timers_ns))
    )
    total_time = 0
    ns_to_ms_scale = 1e-6

    def run(engine_invoke, bench_func_name, *mlir_args):
        engine_invoke(bench_func_name, *mlir_args)

    for i in range(repeat_time + warm_up):
        run(engine.invoke, "wrapped_main", *mlir_args, arg2_memref_ptr)
        if i >= warm_up:
            total_time += int(np_timers_ns[0]) * ns_to_ms_scale
    return total_time / repeat_time


# for test
def fake_bench() -> float:
    return float(random.randint(1, 100))
