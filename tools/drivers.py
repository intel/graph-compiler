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
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gc_mlir import ir
from gc_mlir.dialects import func, onednn_graph
from utils import (
    get_default_passes,
    get_kernel_func_from_module,
    make_tensor,
    mlir_type,
    to_bool_vector,
    to_int_vector,
)


class Driver(ABC):
    """Abstract class for driver."""

    @staticmethod
    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def handle_args(self, args: argparse.Namespace):
        pass

    def __init__(self, ctx: ir.Context, args: argparse.Namespace):
        self.main_entry = "main_entry"
        self.handle_args(args)
        self.ir_module = self.init_module(ctx)

    @abstractmethod
    def init_module(self, ctx: ir.Context) -> ir.Module:
        pass

    @abstractmethod
    def prepare_np_args(self) -> List[np.ndarray]:
        pass

    def get_passes(self) -> str:
        return get_default_passes()


class LoadMLIR(Driver):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--path", type=str, required=True)
        parser.add_argument("--entry", type=str, default="main_entry")

    def handle_args(self, args: argparse.Namespace):
        self.path = args.path
        self.main_entry = args.entry

    def _get_mlir(self):
        with open(self.path, "r") as file:
            content = file.read()
        return content

    def init_module(self, ctx: ir.Context) -> ir.Module:
        module = ir.Module.parse(self._get_mlir(), ctx)
        return module

    def prepare_np_args(self) -> List[np.ndarray]:
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_args = []
        for arg in bench_func.arguments:
            np_args.append(make_tensor(arg.type))
        return np_args

class MLP(Driver):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        print("mlp add args")
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

    def handle_args(self, args: argparse.Namespace):
        self.batch_size = args.batch_size
        assert self.batch_size > 0, "batch size should be greater than 0"

        self.hidden_size_list = to_int_vector(args.hidden_size_list)
        layers = len(self.hidden_size_list) - 1
        assert layers >= 1, "hidden_size_list should have at least 2 elements"

        self.has_bias = (
            [False] * layers
            if "has_bias" not in args.__dict__
            else to_bool_vector(args.has_bias)
        )

        assert (
            len(self.has_bias) == layers
        ), "has_bias should have the same length as hidden_size_list"

        # TODO
        self.has_ln = to_bool_vector(args.has_ln)
        self.act_type = args.act_type
        self.dtype = args.dtype

    def init_module(self, ctx: ir.Context) -> ir.Module:
        with ctx, ir.Location.unknown():
            layers = len(self.params.hidden_size_list) - 1
            module = ir.Module.create()
            dtype = mlir_type(self.params.dtype, ctx)
            src = ir.RankedTensorType.get(
                [self.params.batch_size, self.params.hidden_size_list[0]], dtype
            )
            weights = []
            bias = []
            for i in range(layers):
                weights.append(
                    ir.RankedTensorType.get(
                        [
                            self.params.hidden_size_list[i],
                            self.params.hidden_size_list[i + 1],
                        ],
                        dtype,
                    )
                )
                if self.params.has_bias[i]:
                    bias.append(
                        ir.RankedTensorType.get(
                            [self.params.hidden_size_list[i + 1]], dtype
                        )
                    )
            result = ir.RankedTensorType.get(
                [
                    self.params.batch_size,
                    self.params.hidden_size_list[-1],
                ],
                dtype,
            )
            with ir.InsertionPoint(module.body):
                f = func.FuncOp(
                    name="mlp",
                    type=ir.FunctionType.get(
                        inputs=[src] + weights + bias, results=[result]
                    ),
                )
                with ir.InsertionPoint(f.add_entry_block()):
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
        return module

    def prepare_np_args(self) -> List[np.ndarray]:
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_args = []
        for arg in bench_func.arguments:
            np_args.append(make_tensor(arg.type))
        return np_args
