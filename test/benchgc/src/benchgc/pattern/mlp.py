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
from gc_mlir.dialects import arith, func, linalg, tensor
from gc_mlir.ir import BF16Type, FloatAttr
from benchgc.mlir.utils import (
    STR_TO_MLIR_TYPE,
    get_kernel_func_from_module,
    make_mlir_ndarray,
    to_bool_list,
    to_int_list,
)


class Pattern(ABC):
    """Abstract class for driver."""

    @staticmethod
    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add arguments to parser"""

    @abstractmethod
    def handle_args(self, args: argparse.Namespace):
        """Get and handle the args"""

    def __init__(self, ctx: ir.Context, args: argparse.Namespace):
        self.main_entry = "main_entry"
        self.handle_args(args)
        self.ir_module = self.init_module(ctx)

    @abstractmethod
    def init_module(self, ctx: ir.Context) -> ir.Module:
        """Create MLIR moudule by args"""

class MLP(Pattern):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--hidden_size_list", type=str, default="")
        parser.add_argument("--has_bias", required=False, type=str)
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

        self.hidden_size_list = to_int_list(args.hidden_size_list)
        layers = len(self.hidden_size_list) - 1
        assert layers >= 1, "hidden_size_list should have at least 2 elements"

        self.has_bias = (
            [False] * layers if args.has_bias is None else to_bool_list(args.has_bias)
        )

        assert (
            len(self.has_bias) == layers
        ), "has_bias should have the same length as hidden_size_list"

        self.act_type = args.act_type
        self.dtype = args.dtype

    def init_module(self, ctx: ir.Context) -> ir.Module:
        with ctx, ir.Location.unknown():
            layers = len(self.hidden_size_list) - 1
            module = ir.Module.create()
            dtype = STR_TO_MLIR_TYPE(self.dtype, ctx)
            src = ir.RankedTensorType.get(
                [self.batch_size, self.hidden_size_list[0]], dtype
            )
            weights = []
            bias = []
            for i in range(layers):
                weights.append(
                    ir.RankedTensorType.get(
                        [
                            self.hidden_size_list[i],
                            self.hidden_size_list[i + 1],
                        ],
                        dtype,
                    )
                )
                if self.has_bias[i]:
                    bias.append(
                        ir.RankedTensorType.get([self.hidden_size_list[i + 1]], dtype)
                    )
            result = ir.RankedTensorType.get(
                [
                    self.batch_size,
                    self.hidden_size_list[-1],
                ],
                dtype,
            )
            with ir.InsertionPoint(module.body):
                f = func.FuncOp(
                    name=self.main_entry,
                    type=ir.FunctionType.get(
                        inputs=[src] + weights + bias, results=[result]
                    ),
                )
                f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                with ir.InsertionPoint(f.add_entry_block()):
                    data = f.entry_block.arguments[0]
                    bias_idx = len(weights) + 1
                    for i in range(layers):
                        weight = f.entry_block.arguments[i + 1]
                        if self.has_bias[i]:
                            bias = f.entry_block.arguments[bias_idx]
                            bias_idx += 1
                        else:
                            bias = None
                        layer_out_shape = [
                            self.batch_size,
                            self.hidden_size_list[i + 1],
                        ]

                        data = linalg.matmul(
                            data, weight, outs=[tensor.EmptyOp(layer_out_shape, dtype)]
                        )
                        if bias:
                            broadcast_bias = linalg.broadcast(
                                bias,
                                outs=[tensor.EmptyOp(layer_out_shape, dtype)],
                                dimensions=[0],
                            )
                            data = linalg.add(
                                data,
                                broadcast_bias,
                                outs=[tensor.EmptyOp(layer_out_shape, dtype)],
                            )

                        if self.act_type == "relu":
                            element = FloatAttr.get(dtype, 0)
                            tensor_type = ir.RankedTensorType.get(
                                layer_out_shape, dtype
                            )
                            attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
                            cst = arith.ConstantOp(tensor_type, attr)
                            data = linalg.max(
                                data, cst, outs=[tensor.EmptyOp(layer_out_shape, dtype)]
                            )
                    func.ReturnOp([data])
        return module

    def prepare_np_args(self, disable_results_to_params: False) -> List[np.ndarray]:
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_args = []
        for arg in bench_func.arguments:
            np_args.append(make_mlir_ndarray(arg.type))

        if not disable_results_to_params:
            for res in bench_func.type.results:
                np_args.append(make_mlir_ndarray(res))
        return np_args
