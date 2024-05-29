import argparse

from driver import *
from gc_mlir.ir import FunctionType, InsertionPoint, Location, Module, RankedTensorType
from gc_mlir.dialects import func, onednn_graph
from utils import (
    get_kernel_func_from_module,
    make_tensor,
    mlir_type,
    to_bool_vector,
    to_int_vector,
)


class MLPParams(DriverParams):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--hidden_size_list", type=str, default="")
        parser.add_argument("--has_bias", type=str, default="")
        parser.add_argument("--has_ln", type=str, default="")
        parser.add_argument(
            "--act_type", type=str, choices=["noop", "relu", "sigmoid"], default="noop"
        )
        parser.add_argument(
            "--dtype",
            type=str,
            choices=[
                "f32",
                "bf16",
            ],
            default="f32",
        )

    def __init__(self, params: dict):
        self.batch_size = int(params["batch_size"])
        assert self.batch_size > 0, "batch size should be greater than 0"
        self.hidden_size_list = to_int_vector(params["hidden_size_list"])
        layers = len(self.hidden_size_list) - 1
        assert layers >= 1, "hidden_size_list should have at least 2 elements"

        self.has_bias = (
            [False] * layers
            if "has_bias" not in params
            else to_bool_vector(params["has_bias"])
        )
        assert (
            len(self.has_bias) == layers
        ), "has_bias should have the same length as hidden_size_list"

        # todo
        self.has_ln = to_bool_vector(params["has_ln"])
        self.act_type = params["act_type"]
        self.dtype = params["dtype"]


class MLP(Driver):
    def __init__(self, ctx: ir.Context, params: dict, main_entry: str = "main_entry"):
        self.params = MLPParams(params)
        super().__init__(ctx)

    def init_module(self, ctx: ir.Context) -> ir.Module:
        with ctx, Location.unknown():
            layers = len(self.params.hidden_size_list) - 1
            module = Module.create()
            dtype = mlir_type(self.params.dtype, ctx)
            src = RankedTensorType.get(
                [self.params.batch_size, self.params.hidden_size_list[0]], dtype
            )
            weights = []
            bias = []
            for i in range(layers):
                weights.append(
                    RankedTensorType.get(
                        [
                            self.params.hidden_size_list[i],
                            self.params.hidden_size_list[i + 1],
                        ],
                        dtype,
                    )
                )
                if self.params.has_bias[i]:
                    bias.append(
                        RankedTensorType.get(
                            [self.params.hidden_size_list[i + 1]], dtype
                        )
                    )
            result = RankedTensorType.get(
                [
                    self.params.batch_size,
                    self.params.hidden_size_list[-1],
                ],
                dtype,
            )
            with InsertionPoint(module.body):
                f = func.FuncOp(
                    name="mlp",
                    type=FunctionType.get(
                        inputs=[src] + weights + bias, results=[result]
                    ),
                )
                with InsertionPoint(f.add_entry_block()):
                    data = f.entry_block.arguments[0]
                    bias_idx = len(weights) + 1
                    for i in range(layers):
                        weight = f.entry_block.arguments[i + 1]
                        if self.params.has_bias[i]:
                            bias = f.entry_block.arguments[bias_idx]
                            bias_idx += 1
                        else:
                            bias = None
                        data = onednn_graph.MatMulOp(
                            data,
                            weight,
                            bias=bias,
                            transpose_a=False,
                            transpose_b=False,
                        ).result
                    func.ReturnOp([data])
            print(module)
        return module

    def prepare_np_args(self) -> 'list[np.ndarray]':
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_args = []
        for arg in bench_func.arguments:
            np_args.append(make_tensor(arg.type))
        return np_args

    def prepare_np_res(self) -> 'list[np.ndarray]':
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_res = []
        for res in bench_func.type.results:
            np_res.append(make_tensor(res))
        return np_res
