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
from typing import List

from benchgc.arg.arg import Arg
from benchgc.mlir.util import str_to_mlir_dtype
from benchgc.util import to_bool_list, to_int_list
from gc_mlir import ir
from gc_mlir.dialects import arith, func, linalg, tensor

from .base import Pattern


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
            dtype = str_to_mlir_dtype(ctx, self.dtype)
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
                    zero = arith.ConstantOp(
                        value=ir.FloatAttr.get(dtype, 0.0), result=dtype
                    ).result
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
                        out = linalg.fill(
                            zero, outs=[tensor.EmptyOp(layer_out_shape, dtype)]
                        )
                        data = linalg.matmul(data, weight, outs=[out])
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
                            element = ir.FloatAttr.get(dtype, 0)
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

    def default_fill(
        flags: argparse.Namespace,
        arg: Arg,
        arglist: List[Arg],
    ):
        layers = len(flags.hidden_size_list.strip().split("x"))
        if arg.index == 0:
            # src
            arg.fill_type = "D"
            arg.fill_param = [
                "matmul",
                "src",
                arglist[0].dtype,
                arglist[0].dtype,
                arglist[0].dtype,
                1,
            ]
        elif arg.index <= layers:
            # wei
            arg.fill_type = "D"
            arg.fill_param = [
                "matmul",
                "wei",
                arglist[0].dtype,
                arglist[0].dtype,
                arglist[0].dtype,
                1,
            ]
        else:
            # bias
            arg.fill_type = "N"
            if arg.dtype in ["f32", "bf16", "f16"]:
                arg.fill_param = ["-8", "8"]
            else:
                arg.fill_param = ["0", "8"]
