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
        vnni_size = 1
        shapeA = [M // MBlock, K // KBlock, MBlock, KBlock]
        shapeB = [N // NBlock, K // KBlock, KBlock, NBlock]
        shapeC = [M // MBlock, N // NBlock, MBlock, NBlock] 

        # 4D x 4D, inputs transposed
        mlir_str = """
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @main_entry(%arg0: tensor<2x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>) -> tensor<2x8x32x32xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x8x32x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%1 : tensor<2x8x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<2x8x32x32xf32>
    return %2 : tensor<2x8x32x32xf32>
  }
}
        """
        module = ir.Module.parse(mlir_str)

        torch_arg0 = torch.rand((M, K), dtype=torch.float32)
        torch_arg1 = torch.rand((K, N), dtype=torch.float32)
        ref_res = torch.matmul(torch_arg0, torch_arg1)

        arg0_0 = torch_arg0.view([M // MBlock, MBlock, K // KBlock, KBlock]).permute([0, 2, 1, 3]).contiguous().numpy().view(np.dtype("float32"))
        arg0_1 = np.transpose(np.reshape(torch_arg0.contiguous().numpy().view(np.dtype("float32")), (M // MBlock, MBlock, K // KBlock, KBlock)), (0, 2, 1, 3)) # MK -> MKmk
        print("arg0_0 arg0_1 close: ", np.allclose(arg0_0, arg0_1, rtol=1e-5, atol=1e-5))

        arg1 = torch_arg1.view([K // KBlock, KBlock, N // NBlock, NBlock]).permute([2, 0, 1, 3]).contiguous().numpy().view(np.dtype("float32"))
        # arg1 = np.transpose(np.reshape(torch_arg1.contiguous().numpy(), (16, 32, 8, 32)), (2, 0, 1, 3)).view(np.dtype("float32")) # KN -> NKkn, 8x16x32x32

        gc_res = np.ones(shapeC, dtype=np.dtype("float32"))

        entry = "main_entry"
        mlir_args = get_mlir_args(module, entry, [arg0_1, arg1, gc_res])

        passes = "any(gc-cpu-pipeline)"
        compiler = GraphCompiler(passes)
        engine_in = compiler.compile_and_jit(module)
        engine_in.invoke(entry, *mlir_args)
        gc_res = np.reshape(np.transpose(gc_res, (0, 2, 1, 3)), (64, 256)) # MNmn -> MN

        print("gc_res ref_res close: ", np.allclose(gc_res, ref_res.to(torch.float32).numpy(), rtol=1e-5, atol=1e-5))
        assert_allclose(gc_res, ref_res.to(torch.float32).numpy(), rtol=1e-5, atol=1e-5)

