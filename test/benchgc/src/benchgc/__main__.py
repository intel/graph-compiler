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

from .arg import Arg
from typing import Dict
import runner
import benchgc.fill
import benchgc.util

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
        default=None,
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
            benchgc.util.COMPARE_VERBOSE,
            benchgc.util.ERROR_OUTPUT_VERBOSE,
            benchgc.util.OUTPUT_VERBOSE,
            benchgc.util.INPUT_VERBOSE,
        ],
    )

    parser.add_argument(
        "--dimensions",
        required=False,
        default=None,
        action="append",
        help="define the dimensions attribute in linalg op",
        type=int,
    )
    parser.add_argument(
        "--stride_w",
        required=False,
        default=1,
        help="define the stride attribute",
        type=int,
    )
    parser.add_argument(
        "--stride_h",
        required=False,
        default=1,
        help="define the stride attribute",
        type=int,
    )
    parser.add_argument(
        "--stride_d",
        required=False,
        default=1,
        help="define the stride attribute",
        type=int,
    )

    parser.add_argument(
        "--dilation_w",
        required=False,
        default=1,
        help="define the dilation attribute",
        type=int,
    )
    parser.add_argument(
        "--dilation_h",
        required=False,
        default=1,
        help="define the dilation attribute",
        type=int,
    )
    parser.add_argument(
        "--dilation_d",
        required=False,
        default=1,
        help="define the dilation attribute",
        type=int,
    )

    flags = parser.parse_args()
    benchgc.util.set_seed(flags.seed)



except argparse.ArgumentError:
    sys.stderr.write("Argument parse failed\n")
    sys.exit(1)

ins: Dict[str, Arg] = {}
outs: Dict[str, Arg] = {}

for i in flags.i:
    a = Arg(i)
    ins[a.name] = a

for o in flags.o:
    a = Arg(o)
    outs[a.name] = a

args = ins | outs

for _, arg in ins.items():
    if arg.type == "D" and len(arg.param) == 0:
        benchgc.fill.set_default_fill_param(flags, args, arg)

if flags.driver == "linalg":
    from .linalg import mlir_op
else:
    raise Exception("unsupported driver %s" % flags.driver)


mlir_func = mlir_op[flags.case]
module = mlir_func(flags, args)
print(module)

tensors: Dict[str, torch.Tensor] = {}
for k, v in ins.items():
    tensors[k] = benchgc.fill.fill_tensor(flags, v)

# map tensors arg to mlir arg
for name, arg in args.items():
    if name != arg.name and arg.name in tensors:
        tensors[name] = tensors[arg.name]
        del tensors[arg.name]

runner.ref_run(module, tensors)

for k, v in tensors.items():
    print(k)
    print(v)
