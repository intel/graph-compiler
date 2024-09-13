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

        # 4D x 4D, inputs plain, two layers
        mlir_str_4D4D = """
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
module {
  func.func @entry(%arg0: tensor<64x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x1024xf32>, %arg4: tensor<1024xf32>) -> tensor<64x1024xf32> attributes {llvm.emit_c_interface, runtime_const_args_index = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} {
    %0 = tensor.empty() : tensor<2x16x32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0] high[0, 0] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<64x512xf32> to tensor<64x512xf32>
    %expanded = tensor.expand_shape %padded [[0, 1], [2, 3]] output_shape [2, 32, 16, 32] : tensor<64x512xf32> into tensor<2x32x16x32xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<2x32x16x32xf32>) outs(%0 : tensor<2x16x32x32xf32>) permutation = [0, 2, 1, 3]
    %1 = tensor.empty() : tensor<8x16x32x32xf32>
    %padded_0 = tensor.pad %arg1 low[0, 0] high[0, 0] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<512x256xf32> to tensor<512x256xf32>
    %expanded_1 = tensor.expand_shape %padded_0 [[0, 1], [2, 3]] output_shape [16, 32, 8, 32] : tensor<512x256xf32> into tensor<16x32x8x32xf32>
    %transposed_2 = linalg.transpose ins(%expanded_1 : tensor<16x32x8x32xf32>) outs(%1 : tensor<8x16x32x32xf32>) permutation = [2, 0, 1, 3]
    %2 = tensor.empty() : tensor<2x8x32x32xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%transposed, %transposed_2 : tensor<2x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%3 : tensor<2x8x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %14 = arith.mulf %in, %in_8 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<2x8x32x32xf32>
    %expanded_3 = tensor.expand_shape %arg2 [[0, 1]] output_shape [8, 32] : tensor<256xf32> into tensor<8x32xf32>
    %broadcasted = linalg.broadcast ins(%expanded_3 : tensor<8x32xf32>) outs(%2 : tensor<2x8x32x32xf32>) dimensions = [0, 2]
    %5 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted : tensor<2x8x32x32xf32>) outs(%4 : tensor<2x8x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %out : f32
      linalg.yield %14 : f32
    } -> tensor<2x8x32x32xf32>
    %6 = tensor.empty() : tensor<32x8x32x32xf32>
    %expanded_4 = tensor.expand_shape %arg3 [[0, 1], [2, 3]] output_shape [8, 32, 32, 32] : tensor<256x1024xf32> into tensor<8x32x32x32xf32>
    %transposed_5 = linalg.transpose ins(%expanded_4 : tensor<8x32x32x32xf32>) outs(%6 : tensor<32x8x32x32xf32>) permutation = [2, 0, 1, 3]
    %7 = tensor.empty() : tensor<2x32x32x32xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<2x32x32x32xf32>) -> tensor<2x32x32x32xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%5, %transposed_5 : tensor<2x8x32x32xf32>, tensor<32x8x32x32xf32>) outs(%8 : tensor<2x32x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %14 = arith.mulf %in, %in_8 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<2x32x32x32xf32>
    %expanded_6 = tensor.expand_shape %arg4 [[0, 1]] output_shape [32, 32] : tensor<1024xf32> into tensor<32x32xf32>
    %10 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<32x32xf32>) outs(%9 : tensor<2x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %out : f32
      linalg.yield %14 : f32
    } -> tensor<2x32x32x32xf32>
    %11 = tensor.empty() : tensor<2x32x32x32xf32>
    %transposed_7 = linalg.transpose ins(%10 : tensor<2x32x32x32xf32>) outs(%11 : tensor<2x32x32x32xf32>) permutation = [0, 2, 1, 3]
    %collapsed = tensor.collapse_shape %transposed_7 [[0, 1], [2, 3]] : tensor<2x32x32x32xf32> into tensor<64x1024xf32>
    %12 = tensor.empty() : tensor<64x1024xf32>
    %13 = linalg.copy ins(%collapsed : tensor<64x1024xf32>) outs(%12 : tensor<64x1024xf32>) -> tensor<64x1024xf32>
    return %13 : tensor<64x1024xf32>
  }
}
        """

        module_in = ir.Module.parse(mlir_str_4D4D)


        mlir_str_4D4D_out = """
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
module {
  llvm.mlir.global external constant @__num_orig_args(5 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external constant @__compute_args(dense<[5, 0, 5, 6, 7, 8]> : tensor<6xi32>) {addr_space = 0 : i32} : !llvm.array<6 x i32>
  llvm.mlir.global external constant @__fold_args(dense<[8, 1, 2, 3, 4, 5, 6, 7, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
  llvm.mlir.global external constant @__runtime_fold_buffer_ids_(dense<[4, 0, 1, 2, 3]> : tensor<5xi64>) {addr_space = 0 : i32} : !llvm.array<5 x i64>
  func.func @entry(%arg0: tensor<64x512xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<32x8x32x32xf32>, %arg4: tensor<32x32xf32>) -> tensor<64x1024xf32> attributes {llvm.emit_c_interface, runtime_const_args_index = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x16x32x32xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [2, 32, 16, 32] : tensor<64x512xf32> into tensor<2x32x16x32xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<2x32x16x32xf32>) outs(%0 : tensor<2x16x32x32xf32>) permutation = [0, 2, 1, 3]
    %1 = tensor.empty() : tensor<2x8x32x32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%transposed, %arg1 : tensor<2x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%2 : tensor<2x8x32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<2x8x32x32xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<8x32xf32>) outs(%1 : tensor<2x8x32x32xf32>) dimensions = [0, 2]
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted : tensor<2x8x32x32xf32>) outs(%3 : tensor<2x8x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %in, %out : f32
      linalg.yield %12 : f32
    } -> tensor<2x8x32x32xf32>
    %5 = tensor.empty() : tensor<2x32x32x32xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x32xf32>) -> tensor<2x32x32x32xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%4, %arg3 : tensor<2x8x32x32xf32>, tensor<32x8x32x32xf32>) outs(%6 : tensor<2x32x32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.mulf %in, %in_1 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<2x32x32x32xf32>
    %8 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<32x32xf32>) outs(%7 : tensor<2x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %in, %out : f32
      linalg.yield %12 : f32
    } -> tensor<2x32x32x32xf32>
    %9 = tensor.empty() : tensor<2x32x32x32xf32>
    %transposed_0 = linalg.transpose ins(%8 : tensor<2x32x32x32xf32>) outs(%9 : tensor<2x32x32x32xf32>) permutation = [0, 2, 1, 3]
    %collapsed = tensor.collapse_shape %transposed_0 [[0, 1], [2, 3]] : tensor<2x32x32x32xf32> into tensor<64x1024xf32>
    %10 = tensor.empty() : tensor<64x1024xf32>
    %11 = linalg.copy ins(%collapsed : tensor<64x1024xf32>) outs(%10 : tensor<64x1024xf32>) -> tensor<64x1024xf32>
    return %11 : tensor<64x1024xf32>
  }

  func.func @runtime_fold(%arg0: tensor<512x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x1024xf32>, %arg3: tensor<1024xf32>) -> (tensor<8x16x32x32xf32>, tensor<8x32xf32>, tensor<32x8x32x32xf32>, tensor<32x32xf32>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<8x16x32x32xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [16, 32, 8, 32] : tensor<512x256xf32> into tensor<16x32x8x32xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<16x32x8x32xf32>) outs(%0 : tensor<8x16x32x32xf32>) permutation = [2, 0, 1, 3]
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1]] output_shape [8, 32] : tensor<256xf32> into tensor<8x32xf32>
    %1 = tensor.empty() : tensor<32x8x32x32xf32>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1], [2, 3]] output_shape [8, 32, 32, 32] : tensor<256x1024xf32> into tensor<8x32x32x32xf32>
    %transposed_2 = linalg.transpose ins(%expanded_1 : tensor<8x32x32x32xf32>) outs(%1 : tensor<32x8x32x32xf32>) permutation = [2, 0, 1, 3]
    %expanded_3 = tensor.expand_shape %arg3 [[0, 1]] output_shape [32, 32] : tensor<1024xf32> into tensor<32x32xf32>
    return %transposed, %expanded_0, %transposed_2, %expanded_3 : tensor<8x16x32x32xf32>, tensor<8x32xf32>, tensor<32x8x32x32xf32>, tensor<32x32xf32>
  }
}
            """
        module_out = ir.Module.parse(mlir_str_4D4D_out)

        # module_in entry(%arg0: tensor<64x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x1024xf32>, %arg4: tensor<1024xf32>) -> tensor<64x1024xf32>
        torch_arg0 = torch.rand((64, 512), dtype=torch.float32)
        torch_arg1 = torch.rand((512, 256), dtype=torch.float32)
        torch_arg2 = torch.rand((256), dtype=torch.float32)
        torch_arg3 = torch.rand((256, 1024), dtype=torch.float32)
        torch_arg4 = torch.rand((1024), dtype=torch.float32)

        ref_res = (torch_arg0 @ torch_arg1 + torch_arg2) @ torch_arg3 + torch_arg4

        passes = "any(gc-cpu-pipeline)"
        compiler = GraphCompiler(passes)
        ctx.enable_multithreading(False)

        arg0 = torch_arg0.contiguous().numpy()
        arg1 = torch_arg1.contiguous().numpy()
        arg2 = torch_arg2.contiguous().numpy()
        arg3 = torch_arg3.contiguous().numpy()
        arg4 = torch_arg4.contiguous().numpy()
        gc_res = np.zeros((64, 1024), dtype=np.float32)

        entry = "entry"
        mlir_args = get_mlir_args(module_in, entry, [arg0, arg1, arg2, arg3, arg4, gc_res])
        engine_in = compiler.compile_and_jit(module_in, ir_printing=False)
        engine_in.invoke(entry, *mlir_args)

        print("Reference vs GC input IR close: ", np.allclose(gc_res, ref_res.to(torch.float32).numpy(), rtol=1e-5, atol=1e-5))
        assert_allclose(gc_res, ref_res.to(torch.float32).numpy(), rtol=1e-5, atol=1e-5)


        # module_out entry(%arg0: tensor<64x512xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<32x8x32x32xf32>, %arg4: tensor<32x32xf32>) -> tensor<64x1024xf32>
        # module_out runtime_fold(%arg0: tensor<512x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x1024xf32>, %arg3: tensor<1024xf32>) -> (tensor<8x16x32x32xf32>, tensor<8x32xf32>, tensor<32x8x32x32xf32>, tensor<32x32xf32>)
        fold_arg0 = arg1
        fold_arg1 = arg2
        fold_arg2 = arg3
        fold_arg3 = arg4
        fold_res0 = np.zeros((8, 16, 32, 32), dtype=np.float32)
        fold_res1 = np.zeros((8, 32), dtype=np.float32)
        fold_res2 = np.zeros((32, 8, 32, 32), dtype=np.float32)
        fold_res3 = np.zeros((32, 32), dtype=np.float32)

        runtime_fold = "runtime_fold"
        fold_mlir_args = get_mlir_args(module_out, runtime_fold, [fold_arg0, fold_arg1, fold_arg2, fold_arg3, fold_res0, fold_res1, fold_res2, fold_res3])

        gc_res_out = np.zeros((64, 1024), dtype=np.float32)
        entry = "entry"
        entry_mlir_args = get_mlir_args(module_out, entry, [arg0, fold_res0, fold_res1, fold_res2, fold_res3, gc_res_out])

        engine_out = compiler.compile_and_jit(module_out, ir_printing=False)
        engine_out.invoke(runtime_fold, *fold_mlir_args)
        engine_out.invoke(entry, *entry_mlir_args)

        print("GC input IR vs GC output IR close: ", np.allclose(gc_res, gc_res_out, rtol=1e-5, atol=1e-5))
        assert_allclose(gc_res, gc_res_out, rtol=1e-5, atol=1e-5)
