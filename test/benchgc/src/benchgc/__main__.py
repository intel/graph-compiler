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

from benchgc.arg import Arg, fill_tensor, compare_tensor
from typing import Dict, List, Any
import runner
import benchgc.util
import benchgc.mlir
import gc_mlir.ir
from gc_mlir.graph_compiler import GraphCompiler
import os
from tools.utils import get_mlir_args

try:
    parser = argparse.ArgumentParser(prog="benchmark tool for graph compiler")
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
    flags = parser.parse_args()
    benchgc.util.set_seed(flags.seed)


except argparse.ArgumentError:
    sys.stderr.write("Argument parse failed\n")
    sys.exit(1)

ins: List[Arg] = []
outs: List[Arg] = []

for i in range(len(flags.i)):
    ins.append(Arg(flags.i[i], i))

for i in range(len(flags.o)):
    outs.append(Arg(flags.o[i], i))

for i in range(len(ins)):
    if ins[i].type == "D" and len(ins[i].param) == 0:
        ins[i].set_default_fill_param(flags, ins, outs)

if flags.driver == "linalg":
    from .linalg import mlir_op

    mlir_func = mlir_op[flags.case]
    module = mlir_func(flags, ins, outs)
elif flags.driver == "mlir":
    with open(flags.case, "r") as mlir_file:
        with gc_mlir.ir.Context() as ctx:
            module = gc_mlir.ir.Module.parse(mlir_file.read())
else:
    raise Exception("unsupported driver %s" % flags.driver)

if flags.verbose >= benchgc.util.MODULE_VERBOSE:
    print(module)

gc_args: List[Any] = []
gc_out: List[torch.Tensor] = []
tensors: Dict[str, torch.Tensor] = {}
for i in range(len(ins)):
    tensor = fill_tensor(flags, ins[i], i)
    tensors["%arg" + str(i)] = tensor
    # gc is sharing the same input with reference
    gc_args.append(benchgc.util.tensor_to_ndarray(tensor))

for i in range(len(outs)):
    tensor = torch.zeros(
        size=outs[i].shape, dtype=benchgc.util.get_dtype(outs[i].dtype)
    )
    gc_args.append(benchgc.util.tensor_to_ndarray(tensor))
    gc_out.append(tensor)

ref_out = runner.ref_run(module, tensors)

entry = "entry"

mlir_args = get_mlir_args(module, entry, gc_args)
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
