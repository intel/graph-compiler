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

import os
import sys
import argparse

import numpy as np
from gc_mlir import ir
from gc_mlir.graph_compiler import GraphCompiler
from numpy.testing import assert_allclose

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import ml_dtypes
import torch
from bench import py_timeit_bench
from enhanced_np_to_memref import ranked_memref_to_numpy
from utils import get_mlir_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLIR Correctness Check")
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--tile_m', type=int, default=4)
    parser.add_argument('--tile_n', type=int, default=4)
    parser.add_argument('--tile_k', type=int, default=4)
    parser.add_argument('--dtype',
                        type=str,
                        default="bf16",
                        choices=["bf16", "f32"])
    parser.add_argument('--use_blocked_layout', type=int, default=1)
    args = parser.parse_args()
    with ir.Context() as ctx:
        M = args.M
        N = args.N
        K = args.K
        MBlock = args.tile_m
        NBlock = args.tile_n
        KBlock = args.tile_k
        dtype = args.dtype
        vnni_size = 2 if dtype == "bf16" else 1
        useBlockedLayout = args.use_blocked_layout
        shapeA = [M // MBlock, K //
                  KBlock, MBlock, KBlock] if useBlockedLayout else [M, K]
        shapeB = [N // NBlock, K // KBlock, KBlock // vnni_size, NBlock]
        shapeC = [M // MBlock, N //
                  NBlock, MBlock, NBlock] if useBlockedLayout else [M, N]

        assert M % MBlock == 0
        assert N % NBlock == 0
        assert K % KBlock == 0
        assert KBlock % vnni_size == 0

        if vnni_size > 1:
            shapeB.append(vnni_size)
        Atype = "x".join([str(x) for x in shapeA]) + "x" + dtype
        Btype = "x".join([str(x) for x in shapeB]) + "x" + dtype
        Ctype = "x".join([str(x) for x in shapeC]) + "x" + dtype
        opName = "linalgx.mm4d_vnni" if useBlockedLayout else "linalgx.mm2d_vnni"
        block_start = "{"
        block_end = "}"
        mlir_str = f"""
module {block_start}
    func.func @main_entry(%arg0:tensor<{Atype}>, %arg1:tensor<{Btype}>) ->tensor<{Ctype}> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
        %empty_tensor = tensor.empty() : tensor<{Ctype}>
        %cst_0 = arith.constant 0.000000e+00 : {dtype}
        %arg2 = linalg.fill ins(%cst_0 : {dtype}) outs(%empty_tensor : tensor<{Ctype}>) -> tensor<{Ctype}>
        %0 = {opName} ins(%arg0, %arg1: tensor<{Atype}>, tensor<{Btype}>) outs(%arg2: tensor<{Ctype}>) -> tensor<{Ctype}>
        return %0:tensor<{Ctype}>
    {block_end}
{block_end}     
"""
        print(mlir_str)
        module = ir.Module.parse(mlir_str)
        print(module)

        torch_arg0 = torch.rand(
            [M, K], dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
        torch_arg1 = torch.rand(
            [K, N], dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
        ref_res = torch.matmul(torch_arg0, torch_arg1)

        torch_arg1 = torch_arg1.view(
            [K // KBlock, KBlock // vnni_size, vnni_size, N // NBlock,
             NBlock]).contiguous()
        torch_arg1 = torch_arg1.permute([3, 0, 1, 4, 2]).contiguous()
        torch_arg1 = torch_arg1.view(shapeB).contiguous()
        if useBlockedLayout:
            torch_arg0 = torch_arg0.view(
                [M // MBlock, MBlock, K // KBlock, KBlock]).contiguous()
            ref_res = ref_res.view([M // MBlock, MBlock, N // NBlock,
                                    NBlock]).contiguous()
            torch_arg0 = torch_arg0.permute([0, 2, 1, 3]).contiguous()
            ref_res = ref_res.permute([0, 2, 1, 3]).contiguous()
            torch_arg0 = torch_arg0.view(shapeA).contiguous()
            ref_res = ref_res.view(shapeC).contiguous()

        np_arg0 = torch_arg0.view(
            dtype=torch.uint16 if dtype ==
            "bf16" else torch.float32).numpy().view(
                ml_dtypes.bfloat16 if dtype == "bf16" else ml_dtypes.float32)
        np_arg1 = torch_arg1.view(
            dtype=torch.uint16 if dtype ==
            "bf16" else torch.float32).numpy().view(
                ml_dtypes.bfloat16 if dtype == "bf16" else ml_dtypes.float32)
        gc_res = np.ones(
            shapeC,
            dtype=ml_dtypes.bfloat16 if dtype == "bf16" else ml_dtypes.float32)
        entry = "main_entry"
        mlir_args = get_mlir_args(module, entry, [np_arg0, np_arg1, gc_res])
        passes = "any(gc-cpu-pipeline)"
        shared_libs = [
            os.environ["MLIR_C_RUNNER_UTILS"],
            os.environ["MLIR_RUNNER_UTILS"],
        ]

        # bench
        # _, cost = py_timeit_bench(
        #     module,
        #     "main_entry",
        #     passes,
        #     mlir_args,
        #     shared_libs,
        # )
        # print("cost=", cost)

        # just run
        compiler = GraphCompiler(passes, shared_libs)
        engine = compiler.compile_and_jit(module)
        engine.invoke(entry, *mlir_args)

        # print(gc_res)
        rtol, atol = 1e-5, 1e-5
        if dtype == "f32":
            rtol = 1e-5
            atol = 1e-5
        else:
            torch_dtype = torch.bfloat16
            rtol = 5e-2
            atol = 5e-2
        assert_allclose(
            gc_res.astype(np.float32),
            ref_res.to(torch.float32).numpy(),
            rtol=rtol,
            atol=atol,
        )
