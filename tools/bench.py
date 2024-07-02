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
import random
import timeit
from time import sleep
from typing import Sequence

import numpy as np
from gc_mlir import ir, runtime
from gc_mlir.dialects import arith, func, memref
from gc_mlir.graph_compiler import GraphCompiler
from utils import (
    emit_benchmark_wrapped_main_func,
    emit_nano_time,
    get_kernel_func_from_module,
)


def py_timeit_bench(
    ir_module: ir.Module,
    entry_name: str,
    pipeline: str,
    mlir_args: list,
    shared_libs: Sequence,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    
    compiler = GraphCompiler(
        pipeline,
        shared_libs,
    )
    compile_begin = timeit.default_timer()
    engine = compiler.compile_and_jit(ir_module, ir_printing=ir_printing)
    compile_cost = (timeit.default_timer() - compile_begin) * 1000

    func = engine.lookup(entry_name)
    packed_args = (ctypes.c_void_p * len(mlir_args))()
    for argNum in range(len(mlir_args)):
        packed_args[argNum] = ctypes.cast(mlir_args[argNum], ctypes.c_void_p)

    def run_bench(func, arg):
        func(arg)

    timeit.timeit(lambda: run_bench(func, packed_args), number=warm_up)
    total_time = timeit.timeit(lambda: run_bench(func, packed_args), number=repeat_time)
    execute_cost = total_time * 1000 / repeat_time
    return (execute_cost, compile_cost)


def mlir_wrapper_bench(
    ir_module: ir.Module,
    entry_name: str,
    pipeline: str,
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
    compiler = GraphCompiler(
        pipeline,
        shared_libs,
    )
    compile_begin = timeit.default_timer()
    engine = compiler.compile_and_jit(wrapper_module, ir_printing=ir_printing)
    compile_cost = (timeit.default_timer() - compile_begin) * 1000

    np_timers_ns = np.array([0], dtype=np.int64)
    time_arg = ctypes.pointer(
        ctypes.pointer(runtime.get_ranked_memref_descriptor(np_timers_ns))
    )
    total_time = 0
    ns_to_ms_scale = 1e-6
    def run(engine_invoke, bench_func_name, *mlir_args):
        engine_invoke(bench_func_name, *mlir_args)

    for i in range(repeat_time + warm_up):
        run(engine.invoke, "wrapped_main", time_arg, *mlir_args)
        if i >= warm_up:
            total_time += int(np_timers_ns[0]) * ns_to_ms_scale
    execute_cost = total_time / repeat_time
    return (execute_cost, compile_cost)


# for test
def fake_bench(
    ir_module: ir.Module = None,
    entry_name: str = None,
    pipeline: str = None,
    mlir_args: list = None,
    shared_libs: Sequence = None,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    execute_cost = float(random.randint(1, 100))
    compile_cost = float(random.randint(1, 100))
    return (execute_cost, compile_cost)
