import ctypes

import numpy as np
from gc_mlir import ir
from gc_mlir import runtime as rt
from gc_mlir.dialects import arith, func, memref


def emit_timer_func() -> func.FuncOp:
    i64_type = ir.IntegerType.get_signless(64)
    nanoTime = func.FuncOp("nanoTime", ([], [i64_type]), visibility="private")
    nanoTime.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nanoTime


def emit_benchmark_wrapped_main_func(
    kernel_func: func.FuncOp, timer_func: func.FuncOp
) -> func.FuncOp:
    i64_type = ir.IntegerType.get_signless(64)
    memref_of_i64_type = ir.MemRefType.get([1], i64_type)
    wrapped_func = func.FuncOp(
        # Same signature and an extra buffer of indices to save timings.
        "main",
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



def numpy_to_ctypes(np_dtype):
    if np_dtype == np.int32:
        return ctypes.c_int
    elif np_dtype == np.float64:
        return ctypes.c_double
    elif np_dtype == np.uint8:
        return ctypes.c_ubyte
    elif np_dtype == np.int8:
        return ctypes.c_byte
    elif np_dtype == np.uint16:
        return ctypes.c_ushort
    elif np_dtype == np.int16:
        return ctypes.c_short
    elif np_dtype == np.uint32:
        return ctypes.c_uint
    elif np_dtype == np.int64:
        return ctypes.c_longlong
    elif np_dtype == np.uint64:
        return ctypes.c_ulonglong
    elif np_dtype == np.float32:
        return ctypes.c_float
    else:
        raise ValueError("Unsupported NumPy data type")


def np_args_to_mlir_args(np_args: "list[np.ndarray]") -> "list":
    mlir_args = []
    for arg in np_args:
        mlir_args.append(
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arg)))
        )
    return mlir_args


def np_res_to_mlir_res(np_res: "list[np.ndarray]") -> "list":
    mlir_res = []
    for res in np_res:
        mlir_res.append(
            ctypes.pointer(
                ctypes.pointer(
                    rt.make_nd_memref_descriptor(res.ndim, numpy_to_ctypes(res.dtype))()
                )
            )
        )
    return mlir_res


def get_tensor_mlir_args(f: "func.FuncOp", np_args: "list[np.ndarray]"):
    compiled_program_args = []
    for res in np_args[: len(f.type.results)]:
        compiled_program_args.append(
            ctypes.pointer(
                ctypes.pointer(
                    rt.make_nd_memref_descriptor(res.ndim, numpy_to_ctypes(res.dtype))()
                )
            )
        )
    for arg in np_args[len(f.type.results) :]:
        compiled_program_args.append(
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arg)))
        )
    return compiled_program_args


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
    type_mapping = {
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "u64": np.uint64,
    }
    return np.ones(tensor_type.shape, type_mapping[str(tensor_type.element_type)])


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
        builtin.module(    
            func.func(linalg-generalize-named-ops),
            func.func(linalg-fuse-elementwise-ops),
            convert-shape-to-std,
            one-shot-bufferize,
            cse,
            func-bufferize,
            func.func(bufferization-bufferize),
            func.func(finalizing-bufferize),
            func.func(buffer-deallocation-pipeline),
            func.func(convert-linalg-to-parallel-loops),
            func.func(lower-affine),  
            convert-scf-to-cf, 
            func.func(arith-expand),
            func.func(convert-math-to-llvm),
            convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,lower-affine,convert-bufferization-to-memref,finalize-memref-to-llvm,
            func.func(convert-arith-to-llvm),
            convert-func-to-llvm,
            convert-cf-to-llvm,convert-complex-to-llvm,reconcile-unrealized-casts
        )
    """
    return passes


def to_int_vector(s: str) -> 'list[int]':
    if not s or len(s) == 0:
        return []
    return [int(i) for i in s.strip().split("x")]


def to_bool_vector(s: str) -> 'list[bool]':
    if not s or len(s) == 0:
        return []
    return [bool(i) for i in s.strip().split("x")]


def load_mlir_from_path(path: str) -> str:
    with open(path, "r") as file:
        content = file.read()
    return content
