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
import gc_mlir.ir
import gc_mlir.dialects.onednn_graph
from . import graph, runner, gapi, util

try:
    parser = argparse.ArgumentParser(prog="benchmark tool for graph compiler")
    parser.add_argument("--mlir", default=None, required=False, help="a mlir case file", type=str)
    parser.add_argument("--entry", default=None, required=False, help="main entry function", type=str)
    parser.add_argument(
        "--json",
        required=False,
        default=None,
        help="a json file case file",
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
        default=util.NO_VERBOSE,
        help="verbose level",
        choices=[
            util.NO_VERBOSE,
            util.COMPARE_VERBOSE,
            util.ERROR_OUTPUT_VERBOSE,
            util.OUTPUT_VERBOSE,
            util.INPUT_VERBOSE,
        ],
    )

    args = parser.parse_args()
    util.set_seed(args.seed)
except argparse.ArgumentError:
    sys.stderr.write("Argument parse failed\n")
    sys.exit(1)

if args.mlir is not None:
    with open(args.mlir, "r") as mlir_file:
        with gc_mlir.ir.Context() as ctx:
            gc_mlir.dialects.onednn_graph.register_dialect()
            module = gc_mlir.ir.Module.parse(mlir_file.read())
            mlir_graph = gapi.MLIRGraph(module)
            graph_object = mlir_graph.convert_to_json('"' + args.entry + '"')
            json_graph = gapi.Graph(graph_object)
            ref_graph = graph.Graph(json_graph)
            ref_graph.prepare_input(args.verbose)
            ref_runner = runner.RefRunner(ref_graph)
            ref_runner.execute()
elif args.json is not None:
    # TODO 
    pass
else:
    raise Exception("No mlir or json case provided")
