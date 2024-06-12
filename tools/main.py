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

import argparse
import json
import os
from timeit import repeat, timeit

import numpy as np
from bench import *
from drivers import *
from gc_mlir import ir
from gc_mlir import runtime as rt
from gc_mlir.passmanager import *
from utils import get_mlir_args


def get_driver_clz(diver_str: str):
    clz = {"mlp": MLP, "load_mlir": LoadMLIR}[diver_str]
    return clz


def add_driver_args(parser):
    driver = parser.parse_known_args()[0].driver
    get_driver_clz(driver).add_args(parser)


def do_bench(args):
    with ir.Context() as ctx, ir.Location.unknown():
        driver_clz = get_driver_clz(args.driver)
        driver = driver_clz(ctx, args)
        if args.print_ir:
            ctx.enable_multithreading(False)
        np_args = driver.prepare_np_args(args.disable_results_to_params)

        # TODO need data filling
        # for test, fill all data with 1
        for i in range(len(np_args)):
            np.ndarray.fill(np_args[i], 1)

        mlir_args = get_mlir_args(
            driver.ir_module, driver.main_entry, np_args, args.disable_results_to_params
        )

        print("===========bench func name: ", driver.main_entry, "===========")
        bench_alg = py_timeit_bench if args.bench_alg == "py" else mlir_wrapper_bench
        execute_cost, compile_cost = bench_alg(
            driver.ir_module,
            driver.main_entry,
            driver.get_passes(),
            mlir_args,
            [os.environ["MLIR_C_RUNNER_UTILS"], os.environ["MLIR_RUNNER_UTILS"]],
            args.print_ir,
            args.repeat,
            args.warm_up,
        )
        print("===========bench result===========")
        json_res = json.dumps(
            {
                "args": vars(args),
                "compile_cost": compile_cost,
                "execute_cost": execute_cost,
            },
            indent=4,
        )
        print(json_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["bench", "tune"], default="bench")
    parser.add_argument(
        "--driver", type=str, choices=["load_mlir", "mlp"], required=True
    )
    parser.add_argument(
        "--disable_results_to_params", action="store_true", default=False
    )
    add_driver_args(parser)
    if parser.parse_known_args()[0].type == "bench":
        parser.add_argument("-p", "--print_ir", action="store_true")
        parser.add_argument("--warm_up", type=int, default=20)
        parser.add_argument("--repeat", type=int, default=100)
        parser.add_argument(
            "--bench_alg", type=str, choices=["py", "wrapper"], default="py"
        )
        args = parser.parse_args()
        do_bench(args)
    else:
        # TODO
        print("tuning is not support yet")
