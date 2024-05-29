from gc_mlir import ir
from gc_mlir import runtime as rt

from gc_mlir.passmanager import *
import argparse
from utils import np_res_to_mlir_res, np_args_to_mlir_args
from load_mlir import *
from mlp import *
from timeit import timeit, repeat
import os
import numpy as np
from bench import *


def get_driver(args, ctx):
    clz = {"mlp": MLP, "load_mlir": LoadMLIR}[args.driver]
    return clz(ctx, vars(args))


def do_bench(args):
    with ir.Context() as ctx, ir.Location.unknown():
        driver = get_driver(args, ctx)
        if isinstance(driver, MLP):
            print("can not bench onednn graph dialect for now")
            return
        if args.print_ir:
            ctx.enable_multithreading(False)
        np_res = driver.prepare_np_res()
        np_args = driver.prepare_np_args()

        # todo need data filling
        # for temporary test
        np.ndarray.fill(np_args[0], 10.0)
        np.ndarray.fill(np_args[1], 10.0)

        mlir_args = np_res_to_mlir_res(np_res) + np_args_to_mlir_args(np_args)

        print("===========bench func name: ", driver.main_entry, "===========")

        python_bench(
            ctx,
            driver.ir_module,
            driver.main_entry,
            driver.get_passes(),
            mlir_args,
            args.print_ir,
            args.repeat,
            args.warm_up,
        )

        # MBR_bench(
        #     ctx,
        #     driver.ir_module,
        #     driver.main_entry,
        #     driver.get_passes(),
        #     mlir_args,
        #     args.print_ir,
        #     args.repeat,
        #     args.warm_up,
        # )
        print("===========bench over===========")
        # c = rt.ranked_memref_to_numpy(mlir_args[0][0])
        # print(c)


def check_args_and_env(args):
    c_runner_utils = os.getenv("MLIR_C_RUNNER_UTILS", "")
    runner_utils = os.getenv("MLIR_RUNNER_UTILS", "")
    assert os.path.exists(
        c_runner_utils
    ), f"{c_runner_utils} not found, please set a valid MLIR_C_RUNNER_UTILS"
    assert os.path.exists(
        runner_utils
    ), f"{runner_utils} not found, please set a valid MLIR_RUNNER_UTILS"


def add_driver_args(parser):
    driver = parser.parse_known_args()[0].driver
    driver_to_params: dict[str, DriverParams] = {
        "load_mlir": LoadMLIRParams,
        "mlp": MLPParams,
    }
    driver_to_params[driver].add_args(parser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["bench", "tune"], default="bench")
    parser.add_argument(
        "--driver", type=str, choices=["load_mlir", "mlp"], required=True
    )
    add_driver_args(parser)
    if parser.parse_known_args()[0].type == "bench":
        parser.add_argument("-p", "--print_ir", action="store_true", help="")
        parser.add_argument("--warm_up", type=int, default=20, help="")
        parser.add_argument("--repeat", type=int, default=100, help="")
        args = parser.parse_args()
        check_args_and_env(args)
        do_bench(args)
