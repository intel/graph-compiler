from time import sleep
from gc_mlir import ir
from gc_mlir.passmanager import *
from gc_mlir.execution_engine import *
import ctypes
from gc_mlir.dialects import func, arith, memref
from gc_mlir import runtime
from timeit import timeit, repeat
import random
from graphcomplier import GraphComplier
import numpy as np


def python_bench(
    ctx: ir.Context,
    ir_module: ir.Module,
    entry_name: str,
    passes: str,
    mlir_args: list,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    print("python timeit")
    engine = GraphComplier([], passes).compile_and_jit(
        ir_module, ir_printing=ir_printing
    )
    func = engine.lookup(entry_name)
    packed_args = (ctypes.c_void_p * len(mlir_args))()
    for argNum in range(len(mlir_args)):
        packed_args[argNum] = ctypes.cast(mlir_args[argNum], ctypes.c_void_p)

    def run_bench(func, arg):
        func(arg)

    timeit(lambda: run_bench(func, packed_args), number=warm_up)
    total_time = timeit(lambda: run_bench(func, packed_args), number=repeat_time)
    print(total_time * 1000 / repeat_time, "ms")
    return total_time * 1000 / repeat_time


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


def emit_timer_func() -> func.FuncOp:
    i64_type = ir.IntegerType.get_signless(64)
    nanoTime = func.FuncOp("nanoTime", ([], [i64_type]), visibility="private")
    nanoTime.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nanoTime


def emit_benchmark_wrapped_main_func(kernel_func, timer_func):
    i64_type = ir.IntegerType.get_signless(64)
    memref_of_i64_type = ir.MemRefType.get([1], i64_type)
    wrapped_func = func.FuncOp(
        "_main",
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


def MBR_bench(
    ctx: ir.Context,
    ir_module: ir.Module,
    entry_name: str,
    passes: str,
    mlir_args: list,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    kernel_func = get_kernel_func_from_module(ir_module, entry_name)
    timer_func = emit_timer_func()
    wrapped_func = emit_benchmark_wrapped_main_func(kernel_func, timer_func)
    main_module_with_benchmark = ir.Module.parse(
        str(timer_func) + str(wrapped_func) + str(kernel_func)
    )
    engine = GraphComplier([], passes)
    engine.compile_and_jit(main_module_with_benchmark, ir_printing=ir_printing)
    np_timers_ns = np.array([0], dtype=np.int64)
    arg2_memref_ptr = ctypes.pointer(
        ctypes.pointer(runtime.get_ranked_memref_descriptor(np_timers_ns))
    )
    total_time = 0
    ns_to_ms_scale = 1e-6

    def run(engine_invoke, bench_func_name, *mlir_args):
        engine_invoke(bench_func_name, *mlir_args)

    for i in range(repeat_time + warm_up):
        run(engine.invoke, "_main", *mlir_args, arg2_memref_ptr)
        if i >= warm_up:
            total_time += int(np_timers_ns[0]) * ns_to_ms_scale

    print(total_time / repeat_time, "ms")
    return total_time / repeat_time


def fake_bench() -> float:
    sleep(1)
    return float(random.randint(1, 100))
