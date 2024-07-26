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


def generate_mlir_bf16_2dx4d(M,
                             N,
                             K,
                             tile_m=32,
                             tile_n=32,
                             tile_k=32,
                             dtype_size=2):
    M_block = (M - 1) // tile_m + 1
    K_block = (K - 1) // tile_k + 1
    N_block = (N - 1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M}x{K}xbf16>, %cst: tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16> ) -> tensor<{M}x{N}xbf16> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<{M}x{N}xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<{M}x{N}xbf16>) -> tensor<{M}x{N}xbf16>
    %2 = linalgx.mm2d_vnni ins(%arg0, %cst : tensor<{M}x{K}xbf16>, tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16>) outs(%1 : tensor<{M}x{N}xbf16>) -> tensor<{M}x{N}xbf16>
    return %2 : tensor<{M}x{N}xbf16>
{block_end}
    '''
    return mlir_code


def generate_mlir_bf16_4dx4d(M,
                             N,
                             K,
                             tile_m=32,
                             tile_n=32,
                             tile_k=32,
                             dtype_size=2):
    M_block = (M - 1) // tile_m + 1
    K_block = (K - 1) // tile_k + 1
    N_block = (N - 1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xbf16>, %cst: tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16> ) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    %2 = linalgx.mm4d_vnni ins(%arg0, %cst : tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xbf16>, tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16>) outs(%1 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    return %2 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
{block_end}
    '''
    return mlir_code


def generate_mlir_f32_4dx4d_generic(M, N, K, tile_m=32, tile_n=32, tile_k=32):
    M_block = (M - 1) // tile_m + 1
    K_block = (K - 1) // tile_k + 1
    N_block = (N - 1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xf32>, %cst: tensor<{N_block}x{K_block}x{tile_k}x{tile_n}xf32> ) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32> attributes {block_start}llvm.emit_c_interface{block_end}
 {block_start}
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
    %2 = linalg.generic {block_start}indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]{block_end} ins(%arg0, %cst : tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xf32>, tensor<{N_block}x{K_block}x{tile_k}x{tile_n}xf32>) outs(%1 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>) {block_start}
        ^bb0(%in: f32, %in_1: f32, %out: f32):
        %3 = arith.mulf %in, %in_1 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
    {block_end} -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
    return %2 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
{block_end}
    '''
    return mlir_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLIR Correctness Check")
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--tile_m', type=int, default=4)
    parser.add_argument('--tile_n', type=int, default=4)
    parser.add_argument('--tile_k', type=int, default=4)
    parser.add_argument(
        '--mode',
        type=str,
        default="bf16_4dx4d",
        choices=["bf16_4dx4d", "f32_4dx4d_generic", "bf16_2dx4d"])
    parser.add_argument('--use_blocked_layout', type=int, default=1)
    args = parser.parse_args()
    with ir.Context() as ctx:
        M = args.M
        N = args.N
        K = args.K
        MBlock = args.tile_m
        NBlock = args.tile_n
        KBlock = args.tile_k
        dtype = args.mode.split("_")[0]
        vnni_size = 2 if dtype == "bf16" else 1
        useBlockedLayout = args.mode.split("_")[1].split("x")[0] == "4d"
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
        mlir_str = ""
        if args.mode == "bf16_4dx4d":
            mlir_str = generate_mlir_bf16_4dx4d(M, N, K, MBlock, NBlock,
                                                KBlock)
        elif args.mode == "f32_4dx4d_generic":
            mlir_str = generate_mlir_f32_4dx4d_generic(M, N, K, MBlock, NBlock,
                                                       KBlock)
        elif args.mode == "bf16_2dx4d":
            mlir_str = generate_mlir_bf16_2dx4d(M, N, K, MBlock, NBlock,
                                                KBlock)
        print(mlir_str)
        # with open("/home/zhicong/code/gc-pipeline/build/test.mlir", "r") as file:
        #     content = file.read()
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
                ml_dtypes.bfloat16 if dtype == "bf16" else np.dtype("float32"))
        np_arg1 = torch_arg1.view(
            dtype=torch.uint16 if dtype ==
            "bf16" else torch.float32).numpy().view(
                ml_dtypes.bfloat16 if dtype == "bf16" else np.dtype("float32"))
        gc_res = np.ones(shapeC,
                         dtype=ml_dtypes.bfloat16
                         if dtype == "bf16" else np.dtype("float32"))
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
        isCloseResult = np.allclose(
            gc_res.astype(np.float32),
            ref_res.to(torch.float32).numpy(),
            rtol=rtol,
            atol=atol,
        )
        if isCloseResult:
            print("Correct")
        else:
            print("Incorrect")
