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

from enum import Flag
import os
import sys
import ml_dtypes
import numpy as np
from gc_mlir import ir
from gc_mlir.graph_compiler import GraphCompiler
from numpy.testing import assert_allclose

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import torch
# from bench import py_timeit_bench
from utils import get_mlir_args

if __name__ == "__main__":
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True

        M = 64
        N = 256
        K = 512
        MBlock = 32
        NBlock = 32
        KBlock = 32
        vnni_size = 2
        shapeA = [M // MBlock, K // KBlock, MBlock, KBlock]
        shapeB = [N // NBlock, K // KBlock, KBlock // vnni_size, NBlock, vnni_size]
        shapeC = [M // MBlock, N // NBlock, MBlock, NBlock] 

        block_start = "{"
        block_end = "}"
        mlir_str = f'''
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {block_start}
  func.func @entry(%arg0: tensor<{M // MBlock}x{K // KBlock}x{MBlock}x{KBlock}xbf16>, %cst: tensor<{N // NBlock}x{K // KBlock}x{KBlock // vnni_size}x{NBlock}x{vnni_size}xbf16>) -> tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16>) -> tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16>
    %2 = linalg.generic {block_start}indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]{block_end} ins(%arg0, %cst : tensor<{M // MBlock}x{K // KBlock}x{MBlock}x{KBlock}xbf16>, tensor<{N // NBlock}x{K // KBlock}x{KBlock // vnni_size}x{NBlock}x{vnni_size}xbf16>) outs(%1 : tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16>) {block_start}
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_1 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    {block_end} -> tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16>
    return %2 : tensor<{M // MBlock}x{N // NBlock}x{MBlock}x{NBlock}xbf16>
  {block_end}
{block_end}
        '''
        print(mlir_str)

        # 4D x 5D, inputs transposed
        module_in = ir.Module.parse(mlir_str)

        # entry(%transposed: tensor<2x16x32x32xbf16>, %transposed_5: tensor<8x16x16x32x2xbf16>) -> tensor<2x8x32x32xbf16>
        torch_arg0 = torch.rand((M, K), dtype=torch.bfloat16)
        torch_arg1 = torch.rand((K, N), dtype=torch.bfloat16)
        ref_res = torch_arg0 @ torch_arg1

        passes = "any(gc-cpu-pipeline)"
        shared_libs = [
            os.environ["MLIR_C_RUNNER_UTILS"],
            os.environ["MLIR_RUNNER_UTILS"],
        ]
        compiler = GraphCompiler(passes)
        ctx.enable_multithreading(False)

        arg0 = torch_arg0.view(shapeA).permute([0, 2, 1, 3]).contiguous() # MK -> MKmk
        np_arg0 = arg0.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        arg1 = torch_arg1.view(shapeB).permute([3, 0, 1, 4, 2]).contiguous() # KN -> NKkn2k
        np_arg1 = arg1.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        gc_res = np.ones(shapeC, dtype=ml_dtypes.bfloat16)

        entry = "entry"
        mlir_args = get_mlir_args(module_in, entry, [np_arg0, np_arg1, gc_res])
        engine_in = compiler.compile_and_jit(module_in, ir_printing=False)
        engine_in.invoke(entry, *mlir_args)
        gc_res = np.reshape(np.transpose(gc_res, (0, 2, 1, 3)), (M, N)) # MNmn -> MN

        assert_allclose(gc_res.astype(np.float32), ref_res.to(torch.float32).numpy(), rtol=1e-5, atol=1e-5)
