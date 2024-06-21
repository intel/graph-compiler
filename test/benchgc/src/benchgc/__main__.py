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
        choices=["onednn_graph", "linalg", "tensor", "mlir", "pattern"],
        type=str,
    )
    parser.add_argument(
        "--case",
        required=False,
        help="test which operation in the specified driver",
        type=str,
    )
    parser.add_argument(
        "--arg",
        required=False,
        default=None,
        action="append",
        help="define the arg name, data type, shape and filling type, eg. src:bf16:2x3x4:add:src",
        type=str,
    )
    parser.add_argument(
        "--auto_broadcast",
        required=False,
        choices=["numpy", "none"],
        default="numpy",
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

    flags = parser.parse_args()
    benchgc.util.set_seed(flags.seed)
except argparse.ArgumentError:
    sys.stderr.write("Argument parse failed\n")
    sys.exit(1)

args: Dict[str, Arg] = {}


for argument in flags.arg:
    a = Arg(argument)
    args[a.name] = a

for _, arg in args.items():
    if arg.fill_type == "D" and len(arg.fill_param) == 0:
        benchgc.fill.set_default_fill_param(flags, args, arg)

if flags.driver == "onednn_graph":
    from .onednn_graph import mlir_op
elif flags.driver == "linalg":
    from .linalg import mlir_op
else:
    raise Exception("unsupported driver %s" % flags.driver)

mlir_func = mlir_op[flags.case]
module = mlir_func(flags, args)
print(module)

tensors: Dict[str, torch.Tensor] = {}
for k, v in args.items():
    t: torch.Tensor | None = benchgc.fill.fill_tensor(flags, args, v)
    if t is not None:
        tensors[k] = t

runner.ref_run(module, tensors)

for k, v in tensors.items():
    print(k)
    print(v)
