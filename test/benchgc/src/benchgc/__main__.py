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


import sys
import argparse

import torch

from benchgc.arg.arg import Arg
from benchgc.arg import (
    fill_tensor,
    compare_tensor,
    set_default_compare,
    set_default_fill,
)
from typing import Dict, List
import runner
import benchgc.util
import benchgc.mlir.util
import gc_mlir.ir
from gc_mlir.graph_compiler import GraphCompiler
import os
from benchgc.mlir.arg import get_mlir_args
from benchgc.pattern.mlp import MLP
from benchgc.mlir.bench import py_timeit_bench, mlir_wrapper_bench
import json

def performance_testing(flags, module, args):
    gc_args: List[torch.Tensor | int] = []
    gc_tensors: Dict[str, torch.Tensor] = {}  
    for i in range(len(args)):
        tensor = fill_tensor(flags, args[i], i)
        gc_tensors["%arg" + str(i)] = tensor
        if args[i].scalar:
            gc_args.append(tensor.data_ptr())
        else:
            gc_args.append(tensor)
    
    mlir_args = get_mlir_args(gc_args)
    with module.context as ctx: 
        if flags.print_ir:
            ctx.enable_multithreading(False) 
        bench_kind = py_timeit_bench if flags.bench_kind == "py" else mlir_wrapper_bench
        execute_cost, compile_cost = bench_kind(
            module,
            "entry",
            "any(gc-cpu-pipeline)",
            mlir_args,
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

def correctness_testing(flags, module, args):
    '''correctness testing'''
    ref_args: List[torch.Tensor] = []
    gc_args: List[torch.Tensor | int] = []
    ref_tensors: Dict[str, torch.Tensor] = {}
    gc_tensors: Dict[str, torch.Tensor] = {}

    for i in range(len(args)):
        tensor = fill_tensor(flags, args[i], i)
        gc_tensors["%arg" + str(i)] = tensor
        ref_tensors["%arg" + str(i)] = tensor.clone()
        ref_args.append(ref_tensors["%arg" + str(i)])
        if args[i].scalar:
            gc_args.append(tensor.data_ptr())
        else:
            gc_args.append(tensor)

    # ref_out contains return value of the entry
    entry = benchgc.mlir.util.get_entry(module)
    ref_out = runner.ref_run(entry, ref_tensors)

    # we need to swap the result into the args if some arg is the return value
    if ref_out is not None:
        for i in range(len(ref_out)):
            ref_args[0 - i - 1] = ref_out[0 - i - 1]

    entry = "entry"

    mlir_args = get_mlir_args(gc_args)
    passes = "any(gc-cpu-pipeline)"

    with module.context:
        compiler = GraphCompiler(passes)
        engine = compiler.compile_and_jit(module)
        engine.invoke(entry, *mlir_args)

    fail, mistrust = False, False
    for i in range(len(args)):
        # gc_arg contains address for scalar value
        # we need to find result by arg name
        res = compare_tensor(
            args[i], ref_args[i], gc_tensors["%arg" + str(i)], flags.verbose
        )
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

def add_common_options(parser: argparse.ArgumentParser):
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
        "--md",
        required=False,
        help="format: #ARG:SHAPExTYPE",
        type=str,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--fill",
        required=False,
        help="format: #ARG:type:parameter",
        type=str,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--cmp",
        required=False,
        help="format: #ARG:type:parameter",
        type=str,
        default=[],
        action="append",
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
            benchgc.util.ARG_VERBOSE,
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

def get_moudle_and_args(flags):
    '''get module args'''  
    args: List[Arg] = []
    if flags.driver == "mlir":
        # we need to find all args by reading the entry function
        with open(flags.case, "r") as mlir_file:
            with gc_mlir.ir.Context() as ctx:
                module = gc_mlir.ir.Module.parse(mlir_file.read())
                entry = benchgc.mlir.util.get_entry(module)
                idx: int = 0
                # FIXME: only support RankTensorType now
                for i in entry.type.inputs:
                    args.append(Arg(idx))
                    args[-1].dtype = str(i.element_type)
                    args[-1].shape = list(i.shape)
                    args[-1].set_scalar()
                    idx += 1

                for o in entry.type.results:
                    args.append(Arg(idx))
                    args[-1].dtype = str(o.element_type)
                    args[-1].shape = list(o.shape)
                    args[-1].set_scalar()
                    idx += 1
    elif flags.driver in ["linalg"]:
        # all arg shape/dt should be provided in single op test
        for i in range(len(flags.md)):
            args.append(Arg(i))

        for md in flags.md:
            colon = md.find(":")
            if colon == -1:
                raise Exception("Wrong md format: %s", md)
            idx = int(md[:colon])
            args[idx].set_md(md[colon + 1 :])

        for fill in flags.fill:
            colon = fill.find(":")
            if colon == -1:
                raise Exception("Wrong fill format: %s", fill)
            idx = int(fill[:colon])
            args[idx].set_fill(fill[colon + 1 :])

        for cmp in flags.cmp:
            colon = cmp.find(":")
            if colon == -1:
                raise Exception("Wrong fill format: %s", cmp)
            idx = int(cmp[:colon])
            args[idx].set_cmp(cmp[colon + 1 :])

        from .linalg import mlir_op

        mlir_func = mlir_op[flags.case]
        module = mlir_func(flags, args)
    elif flags.driver in ["pattern"]:
        with gc_mlir.ir.Context() as ctx:
            clz = get_pattern_clz(flags.case)
            module = clz(ctx, flags).ir_module
            entry = benchgc.mlir.util.get_entry(module)
            idx: int = 0
            # FIXME: only support RankTensorType now
            for i in entry.type.inputs:
                args.append(Arg(idx))
                args[-1].dtype = str(i.element_type)
                args[-1].shape = list(i.shape)
                args[-1].set_scalar()
                idx += 1

            for o in entry.type.results:
                args.append(Arg(idx))
                args[-1].dtype = str(o.element_type)
                args[-1].shape = list(o.shape)
                args[-1].set_scalar()
                idx += 1
    else:
        raise Exception("unsupported driver %s" % flags.driver)
    
    for i in range(len(args)):
        # use zero filling if the arg is return value
        set_default_fill(flags, args[i], args, i >= len(entry.type.inputs))
        set_default_compare(flags, args[i], args)

    for arg in args:
        arg.print_verbose(flags.verbose)

    if flags.verbose >= benchgc.util.MODULE_VERBOSE:
        print(module)
    
    return module, args


if __name__ == "__main__":  
    arg_parser = argparse.ArgumentParser(prog="benchmark tool for graph compiler")
    add_common_options(arg_parser)
    add_bench_options(arg_parser)
    add_pattern_options(arg_parser)
    flags = arg_parser.parse_args()
    benchgc.util.set_seed(flags.seed)
    ir_module, module_args = get_moudle_and_args(flags)
    if flags.mode == "C":
        correctness_testing(flags, ir_module, module_args)
    elif flags.mode == "P":
        performance_testing(flags, ir_module, module_args)
    else:
        pass