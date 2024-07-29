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


from ast import pattern
import sys
import argparse

from numpy import add

import torch

from benchgc.arg import Arg, fill_tensor, compare_tensor
from typing import Dict, List
import runner
import benchgc.util
import benchgc.mlir.util
import gc_mlir.ir
from gc_mlir.graph_compiler import GraphCompiler
import os
from benchgc.mlir.arg import get_mlir_args
from benchgc.mlir.bench import py_timeit_bench, mlir_wrapper_bench
import json
from benchgc.pattern.mlp import MLP


def correctness_testing(module, ins, outs):
    """
    Correctness testing
    """
    gc_args: List[torch.Tensor | int] = []
    gc_out: List[torch.Tensor] = []
    tensors: Dict[str, torch.Tensor] = {}
    for i in range(len(ins)):
        tensor = fill_tensor(flags, ins[i], i)
        tensors["%arg" + str(i)] = tensor
        # gc is sharing the same input with reference
        if ins[i].scalar:
            gc_args.append(tensor.data_ptr())
        else:
            gc_args.append(tensor)

    for i in range(len(outs)):
        tensor = torch.zeros(
            size=outs[i].shape, dtype=benchgc.util.get_dtype(outs[i].dtype)
        )
        gc_args.append(tensor)
        gc_out.append(tensor)

    ref_out = runner.ref_run(module, tensors)

    entry = "entry"

    mlir_args = get_mlir_args(gc_args)
    passes = "any(gc-cpu-pipeline)"
    shared_libs = [
        os.environ["MLIR_C_RUNNER_UTILS"],
        os.environ["MLIR_RUNNER_UTILS"],
    ]

    with module.context:
        compiler = GraphCompiler(passes, shared_libs)
        engine = compiler.compile_and_jit(module)
        engine.invoke(entry, *mlir_args)

    fail, mistrust = False, False
    for i in range(len(outs)):
        out = outs[i]
        out.set_default_compare_param(flags, ins, outs)
        res = compare_tensor(out, ref_out[i], gc_out[i], flags.verbose)
        fail = fail or (not res[0])
        if res[1] is not None:
            mistrust = mistrust | res[1]
    if fail:
        print("FAIL: %s.%s" % (flags.driver, flags.case))
        sys.exit(1)
    elif mistrust:
        print("MISTRUST: %s.%s" % (flags.driver, flags.case))
    else:
        print("PASSED: %s.%s" % (flags.driver, flags.case))

def performance_testing(module, ins, outs):
    '''
    Performance testing
    '''
    gc_args: List[torch.Tensor | int] = []
    gc_out: List[torch.Tensor] = []
    tensors: Dict[str, torch.Tensor] = {}
    for i in range(len(ins)):
        tensor = fill_tensor(flags, ins[i], i)
        tensors["%arg" + str(i)] = tensor
        # gc is sharing the same input with reference
        if ins[i].scalar:
            gc_args.append(tensor.data_ptr())
        else:
            gc_args.append(tensor)

    for i in range(len(outs)):
        tensor = torch.zeros(
            size=outs[i].shape, dtype=benchgc.util.get_dtype(outs[i].dtype)
        )
        gc_args.append(tensor)
        gc_out.append(tensor)
    
    with module.context:
        mlir_args = get_mlir_args(gc_args)  
        bench_kind = py_timeit_bench if flags.bench_kind == "py" else mlir_wrapper_bench
        execute_cost, compile_cost = bench_kind(
            module,
            "entry",
            "any(gc-cpu-pipeline)",
            mlir_args,
            [os.environ["MLIR_C_RUNNER_UTILS"], os.environ["MLIR_RUNNER_UTILS"]],
            flags.print_ir,
            flags.repeat,
            flags.warm_up,
            )
        print("===========bench result===========")
        json_res = json.dumps(
            {
                "args": vars(flags),
                "compile_cost(ms)": compile_cost,
                "execute_cost(ms)": execute_cost,
            },
            indent=4,
        )
        print(json_res)

def add_common_options(parser: argparse.ArgumentParser):
    '''add bench bc common options'''
    parser.add_argument(
        "--mode",
        required=False,
        help="specify the test mode, C for correctness testing, P for performance testing",
        choices=["C", "P"],
        default="C",
        type=str,
    )
    parser.add_argument(
        "--driver",
        required=False,
        help="specify the test driver",
        choices=["linalg", "tensor", "mlir", "pattern"],
        type=str,
    )
    parser.add_argument(
        "--case",
        required=False,
        help="test which operation in the specified driver",
        type=str,
    )
    parser.add_argument(
        "-i",
        required=False,
        default=[],
        action="append",
        help="define the input arg name, data type, shape and filling type, eg. src:bf16:2x3x4:add:src",
        type=str,
    )
    parser.add_argument(
        "-o",
        required=False,
        default=None,
        action="append",
        help="define the output arg name, data type, shape and check type, eg. dst:bf16:2x3x4:add:dst",
        type=str,
    )
    parser.add_argument(
        "--seed",
        required=False,
        default=0,
        type=int,
        help="a seed value to generate data filling",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=benchgc.util.NO_VERBOSE,
        help="verbose level",
        choices=[
            benchgc.util.NO_VERBOSE,
            benchgc.util.MODULE_VERBOSE,
            benchgc.util.COMPARE_VERBOSE,
            benchgc.util.ERROR_OUTPUT_VERBOSE,
            benchgc.util.OUTPUT_VERBOSE,
            benchgc.util.INPUT_VERBOSE,
        ],
    )
    parser.add_argument(
        "--cast",
        required=False,
        default="cast_signed",
        help="define attribute supported by linalg op such as matmul_transpose_b",
        choices=["cast_signed", "cast_unsigned"],
        type=str,
    )
    parser.add_argument(
        "--dimensions",
        required=False,
        default=None,
        action="append",
        help="define the dimensions attribute in linalg op",
        type=int,
    )


def add_bench_options(parser: argparse.ArgumentParser):
    ''' add options for bench mode'''
    if parser.parse_known_args()[0].mode == "P":
        parser.add_argument("-p", "--print_ir", 
                           action="store_true",
                           help="if need print the IR after pipeline",
                           required=False
                           )
        parser.add_argument(
        "--disable_results_to_params", 
            action="store_true", 
            default=False
        )       
        parser.add_argument(
        "--bench_kind", 
         type=str, choices=["py", "wrapper"],default="py"
        )       
        parser.add_argument("--warm_up", type=int, default=100)
        parser.add_argument("--repeat", type=int, default=100)
        parser.add_argument("--entry", type=str, default="main_entry")

def get_pattern_clz(diver_str: str):
    """Function getting Pattern class by name."""
    clz = {"mlp": MLP}[diver_str]
    return clz

def add_pattern_options(parser: argparse.ArgumentParser):
    '''add options for each pattern'''
    if parser.parse_known_args()[0].driver == "pattern":
        pattern_name = parser.parse_known_args()[0].driver
        get_pattern_clz(pattern_name).add_args(parser)

def get_ins_and_outs_from_arg(flags):
    ins: List[Arg] = []
    outs: List[Arg] = []
    for i in range(len(flags.i)):
        ins.append(Arg(flags.i[i], i))
    for i in range(len(flags.o)):
        outs.append(Arg(flags.o[i], i))
    for i in range(len(ins)):
        if ins[i].type == "D" and len(ins[i].param) == 0:
            ins[i].set_default_fill_param(flags, ins, outs)
    return ins, outs


def get_mlir_module(flags):
    if flags.driver == "linalg":
        from .linalg import mlir_op
        mlir_func = mlir_op[flags.case]
        module = mlir_func(flags, ins, outs)
    elif flags.driver == "mlir":
        with open(flags.case, "r") as mlir_file:
            with gc_mlir.ir.Context() as ctx:
                module = gc_mlir.ir.Module.parse(mlir_file.read())
    elif flags.driver == "pattern":
        with gc_mlir.ir.Context() as ctx:
            clz = get_pattern_clz(flags.case)
            module = clz(ctx, flags).ir_module
    else:
        raise Exception("unsupported driver %s" % flags.driver)
    if flags.verbose >= benchgc.util.MODULE_VERBOSE:
        print(module)
    return module
    
if __name__ == "__main__":
    from config import MLIR_C_RUNNER_UTILS
    print(f"The value of MLIR_C_RUNNER_UTILS is: {MLIR_C_RUNNER_UTILS}")
    
    arg_parser = argparse.ArgumentParser(prog="benchmark tool for graph compiler")
    add_common_options(arg_parser)
    add_bench_options(arg_parser)
    add_pattern_options(arg_parser)
    flags = arg_parser.parse_args()
    benchgc.util.set_seed(flags.seed)
    ins, outs = get_ins_and_outs_from_arg(flags)
    ir_module = get_mlir_module(flags)
    if flags.mode == "C":
        correctness_testing(ir_module, ins, outs)
    elif flags.mode == "P":
        performance_testing(ir_module, ins, outs)
    else:
        pass