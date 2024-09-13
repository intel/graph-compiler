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
import ml_dtypes

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
        # ctx.enable_multithreading = False
        module_in = ir.Module.parse(
            """
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
module {
  func.func @entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<512x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x1024xbf16>, %arg4: tensor<1024xbf16>) -> tensor<64x1024xbf16> attributes {llvm.emit_c_interface, runtime_const_args_index = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} {
    %0 = tensor.empty() : tensor<2x16x32x32xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %padded = tensor.pad %arg0 low[0, 0] high[0, 0] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %cst : bf16
    } : tensor<64x512xbf16> to tensor<64x512xbf16>
    %expanded = tensor.expand_shape %padded [[0, 1], [2, 3]] output_shape [2, 32, 16, 32] : tensor<64x512xbf16> into tensor<2x32x16x32xbf16>
    %transposed = linalg.transpose ins(%expanded : tensor<2x32x16x32xbf16>) outs(%0 : tensor<2x16x32x32xbf16>) permutation = [0, 2, 1, 3]
    %1 = tensor.empty() : tensor<8x16x32x32xbf16>
    %padded_0 = tensor.pad %arg1 low[0, 0] high[0, 0] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %cst : bf16
    } : tensor<512x256xbf16> to tensor<512x256xbf16>
    %expanded_1 = tensor.expand_shape %padded_0 [[0, 1], [2, 3]] output_shape [16, 32, 8, 32] : tensor<512x256xbf16> into tensor<16x32x8x32xbf16>
    %transposed_2 = linalg.transpose ins(%expanded_1 : tensor<16x32x8x32xbf16>) outs(%1 : tensor<8x16x32x32xbf16>) permutation = [2, 0, 1, 3]
    %2 = tensor.empty() : tensor<8x16x16x32x2xbf16>
    %padded_3 = tensor.pad %transposed_2 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
      tensor.yield %cst : bf16
    } : tensor<8x16x32x32xbf16> to tensor<8x16x32x32xbf16>
    %expanded_4 = tensor.expand_shape %padded_3 [[0], [1], [2, 3], [4]] output_shape [8, 16, 16, 2, 32] : tensor<8x16x32x32xbf16> into tensor<8x16x16x2x32xbf16>
    %transposed_5 = linalg.transpose ins(%expanded_4 : tensor<8x16x16x2x32xbf16>) outs(%2 : tensor<8x16x16x32x2xbf16>) permutation = [0, 1, 2, 4, 3]
    %3 = tensor.empty() : tensor<2x8x32x32xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<2x8x32x32xbf16>) -> tensor<2x8x32x32xbf16>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%transposed, %transposed_5 : tensor<2x16x32x32xbf16>, tensor<8x16x16x32x2xbf16>) outs(%4 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %17 = arith.mulf %in, %in_19 : bf16
      %18 = arith.addf %out, %17 : bf16
      linalg.yield %18 : bf16
    } -> tensor<2x8x32x32xbf16>
    %6 = tensor.empty() : tensor<8x32xbf16>
    %padded_6 = tensor.pad %arg2 low[0] high[0] {
    ^bb0(%arg5: index):
      tensor.yield %cst : bf16
    } : tensor<256xbf16> to tensor<256xbf16>
    %expanded_7 = tensor.expand_shape %padded_6 [[0, 1]] output_shape [8, 32] : tensor<256xbf16> into tensor<8x32xbf16>
    %transposed_8 = linalg.transpose ins(%expanded_7 : tensor<8x32xbf16>) outs(%6 : tensor<8x32xbf16>) permutation = [0, 1]
    %broadcasted = linalg.broadcast ins(%transposed_8 : tensor<8x32xbf16>) outs(%3 : tensor<2x8x32x32xbf16>) dimensions = [0, 2]
    %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted : tensor<2x8x32x32xbf16>) outs(%5 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %17 = arith.addf %in, %out : bf16
      linalg.yield %17 : bf16
    } -> tensor<2x8x32x32xbf16>
    %8 = tensor.empty() : tensor<32x8x32x32xbf16>
    %padded_9 = tensor.pad %arg3 low[0, 0] high[0, 0] {
    ^bb0(%arg5: index, %arg6: index):
      tensor.yield %cst : bf16
    } : tensor<256x1024xbf16> to tensor<256x1024xbf16>
    %expanded_10 = tensor.expand_shape %padded_9 [[0, 1], [2, 3]] output_shape [8, 32, 32, 32] : tensor<256x1024xbf16> into tensor<8x32x32x32xbf16>
    %transposed_11 = linalg.transpose ins(%expanded_10 : tensor<8x32x32x32xbf16>) outs(%8 : tensor<32x8x32x32xbf16>) permutation = [2, 0, 1, 3]
    %9 = tensor.empty() : tensor<32x8x16x32x2xbf16>
    %padded_12 = tensor.pad %transposed_11 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
      tensor.yield %cst : bf16
    } : tensor<32x8x32x32xbf16> to tensor<32x8x32x32xbf16>
    %expanded_13 = tensor.expand_shape %padded_12 [[0], [1], [2, 3], [4]] output_shape [32, 8, 16, 2, 32] : tensor<32x8x32x32xbf16> into tensor<32x8x16x2x32xbf16>
    %transposed_14 = linalg.transpose ins(%expanded_13 : tensor<32x8x16x2x32xbf16>) outs(%9 : tensor<32x8x16x32x2xbf16>) permutation = [0, 1, 2, 4, 3]
    %10 = tensor.empty() : tensor<2x32x32x32xbf16>
    %11 = linalg.fill ins(%cst : bf16) outs(%10 : tensor<2x32x32x32xbf16>) -> tensor<2x32x32x32xbf16>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%7, %transposed_14 : tensor<2x8x32x32xbf16>, tensor<32x8x16x32x2xbf16>) outs(%11 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_19: bf16, %out: bf16):
      %17 = arith.mulf %in, %in_19 : bf16
      %18 = arith.addf %out, %17 : bf16
      linalg.yield %18 : bf16
    } -> tensor<2x32x32x32xbf16>
    %13 = tensor.empty() : tensor<32x32xbf16>
    %padded_15 = tensor.pad %arg4 low[0] high[0] {
    ^bb0(%arg5: index):
      tensor.yield %cst : bf16
    } : tensor<1024xbf16> to tensor<1024xbf16>
    %expanded_16 = tensor.expand_shape %padded_15 [[0, 1]] output_shape [32, 32] : tensor<1024xbf16> into tensor<32x32xbf16>
    %transposed_17 = linalg.transpose ins(%expanded_16 : tensor<32x32xbf16>) outs(%13 : tensor<32x32xbf16>) permutation = [0, 1]
    %14 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_17 : tensor<32x32xbf16>) outs(%12 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %17 = arith.addf %in, %out : bf16
      linalg.yield %17 : bf16
    } -> tensor<2x32x32x32xbf16>
    %15 = tensor.empty() : tensor<64x1024xbf16>
    %transposed_18 = linalg.transpose ins(%14 : tensor<2x32x32x32xbf16>) outs(%10 : tensor<2x32x32x32xbf16>) permutation = [0, 2, 1, 3]
    %collapsed = tensor.collapse_shape %transposed_18 [[0, 1], [2, 3]] : tensor<2x32x32x32xbf16> into tensor<64x1024xbf16>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0] [64, 1024] [1, 1] : tensor<64x1024xbf16> to tensor<64x1024xbf16>
    %16 = linalg.copy ins(%extracted_slice : tensor<64x1024xbf16>) outs(%15 : tensor<64x1024xbf16>) -> tensor<64x1024xbf16>
    return %16 : tensor<64x1024xbf16>
  }
}
            """
        )
        module_out = ir.Module.parse(
            """
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
module {
  llvm.mlir.global external constant @__num_orig_args(5 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external constant @__compute_args(dense<[5, 0, 5, 6, 7, 8]> : tensor<6xi32>) {addr_space = 0 : i32} : !llvm.array<6 x i32>
  llvm.mlir.global external constant @__fold_args(dense<[8, 1, 2, 3, 4, 5, 6, 7, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
  llvm.mlir.global external constant @__runtime_fold_buffer_ids_(dense<[4, 0, 1, 2, 3]> : tensor<5xi64>) {addr_space = 0 : i32} : !llvm.array<5 x i64>
  func.func @entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<8x16x16x32x2xbf16>, %arg2: tensor<8x32xbf16>, %arg3: tensor<32x8x16x32x2xbf16>, %arg4: tensor<32x32xbf16>) -> tensor<64x1024xbf16> attributes {llvm.emit_c_interface, runtime_const_args_index = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<2x16x32x32xbf16>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [2, 32, 16, 32] : tensor<64x512xbf16> into tensor<2x32x16x32xbf16>
    %transposed = linalg.transpose ins(%expanded : tensor<2x32x16x32xbf16>) outs(%0 : tensor<2x16x32x32xbf16>) permutation = [0, 2, 1, 3]
    %1 = tensor.empty() : tensor<2x8x32x32xbf16>
    %2 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<2x8x32x32xbf16>) -> tensor<2x8x32x32xbf16>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%transposed, %arg1 : tensor<2x16x32x32xbf16>, tensor<8x16x16x32x2xbf16>) outs(%2 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %11 = arith.mulf %in, %in_1 : bf16
      %12 = arith.addf %out, %11 : bf16
      linalg.yield %12 : bf16
    } -> tensor<2x8x32x32xbf16>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<8x32xbf16>) outs(%1 : tensor<2x8x32x32xbf16>) dimensions = [0, 2]
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted : tensor<2x8x32x32xbf16>) outs(%3 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %11 = arith.addf %in, %out : bf16
      linalg.yield %11 : bf16
    } -> tensor<2x8x32x32xbf16>
    %5 = tensor.empty() : tensor<2x32x32x32xbf16>
    %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<2x32x32x32xbf16>) -> tensor<2x32x32x32xbf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%4, %arg3 : tensor<2x8x32x32xbf16>, tensor<32x8x16x32x2xbf16>) outs(%6 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %11 = arith.mulf %in, %in_1 : bf16
      %12 = arith.addf %out, %11 : bf16
      linalg.yield %12 : bf16
    } -> tensor<2x32x32x32xbf16>
    %8 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<32x32xbf16>) outs(%7 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %11 = arith.addf %in, %out : bf16
      linalg.yield %11 : bf16
    } -> tensor<2x32x32x32xbf16>
    %9 = tensor.empty() : tensor<64x1024xbf16>
    %transposed_0 = linalg.transpose ins(%8 : tensor<2x32x32x32xbf16>) outs(%5 : tensor<2x32x32x32xbf16>) permutation = [0, 2, 1, 3]
    %collapsed = tensor.collapse_shape %transposed_0 [[0, 1], [2, 3]] : tensor<2x32x32x32xbf16> into tensor<64x1024xbf16>
    %10 = linalg.copy ins(%collapsed : tensor<64x1024xbf16>) outs(%9 : tensor<64x1024xbf16>) -> tensor<64x1024xbf16>
    return %10 : tensor<64x1024xbf16>
  }
  func.func @runtime_fold(%arg0: tensor<512x256xbf16>, %arg1: tensor<256xbf16>, %arg2: tensor<256x1024xbf16>, %arg3: tensor<1024xbf16>) -> (tensor<8x16x16x32x2xbf16>, tensor<8x32xbf16>, tensor<32x8x16x32x2xbf16>, tensor<32x32xbf16>) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<8x16x32x32xbf16>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [16, 32, 8, 32] : tensor<512x256xbf16> into tensor<16x32x8x32xbf16>
    %transposed = linalg.transpose ins(%expanded : tensor<16x32x8x32xbf16>) outs(%0 : tensor<8x16x32x32xbf16>) permutation = [2, 0, 1, 3]
    %1 = tensor.empty() : tensor<8x16x16x32x2xbf16>
    %expanded_0 = tensor.expand_shape %transposed [[0], [1], [2, 3], [4]] output_shape [8, 16, 16, 2, 32] : tensor<8x16x32x32xbf16> into tensor<8x16x16x2x32xbf16>
    %transposed_1 = linalg.transpose ins(%expanded_0 : tensor<8x16x16x2x32xbf16>) outs(%1 : tensor<8x16x16x32x2xbf16>) permutation = [0, 1, 2, 4, 3]
    %expanded_2 = tensor.expand_shape %arg1 [[0, 1]] output_shape [8, 32] : tensor<256xbf16> into tensor<8x32xbf16>
    %2 = tensor.empty() : tensor<32x8x32x32xbf16>
    %expanded_3 = tensor.expand_shape %arg2 [[0, 1], [2, 3]] output_shape [8, 32, 32, 32] : tensor<256x1024xbf16> into tensor<8x32x32x32xbf16>
    %transposed_4 = linalg.transpose ins(%expanded_3 : tensor<8x32x32x32xbf16>) outs(%2 : tensor<32x8x32x32xbf16>) permutation = [2, 0, 1, 3]
    %3 = tensor.empty() : tensor<32x8x16x32x2xbf16>
    %expanded_5 = tensor.expand_shape %transposed_4 [[0], [1], [2, 3], [4]] output_shape [32, 8, 16, 2, 32] : tensor<32x8x32x32xbf16> into tensor<32x8x16x2x32xbf16>
    %transposed_6 = linalg.transpose ins(%expanded_5 : tensor<32x8x16x2x32xbf16>) outs(%3 : tensor<32x8x16x32x2xbf16>) permutation = [0, 1, 2, 4, 3]
    %expanded_7 = tensor.expand_shape %arg3 [[0, 1]] output_shape [32, 32] : tensor<1024xbf16> into tensor<32x32xbf16>
    return %transposed_1, %expanded_2, %transposed_6, %expanded_7 : tensor<8x16x16x32x2xbf16>, tensor<8x32xbf16>, tensor<32x8x16x32x2xbf16>, tensor<32x32xbf16>
  }
}
            """
        )

        # module_in entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<512x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x1024xbf16>, %arg4: tensor<1024xbf16>) -> tensor<64x1024xbf16>
        torch_arg0 = torch.rand((64, 512), dtype=torch.bfloat16)
        torch_arg1 = torch.rand((512, 256), dtype=torch.bfloat16)
        torch_arg2 = torch.rand((256), dtype=torch.bfloat16)
        torch_arg3 = torch.rand((256, 1024), dtype=torch.bfloat16)
        torch_arg4 = torch.rand((1024), dtype=torch.bfloat16)

        ref_res = (torch_arg0 @ torch_arg1 + torch_arg2) @ torch_arg3 + torch_arg4

        passes = "any(gc-cpu-pipeline)"
        compiler = GraphCompiler(passes)
        ctx.enable_multithreading(False)

        arg0 = torch_arg0.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        arg1 = torch_arg1.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        arg2 = torch_arg2.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        arg3 = torch_arg3.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        arg4 = torch_arg4.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        gc_res = np.ones((64, 1024), dtype=ml_dtypes.bfloat16)

        entry = "entry"
        mlir_args = get_mlir_args(module_in, entry, [arg0, arg1, arg2, arg3, arg4, gc_res])
        engine_in = compiler.compile_and_jit(module_in, ir_printing=True)
        engine_in.invoke(entry, *mlir_args)

        assert_allclose(gc_res.astype(np.float32), ref_res.to(torch.float32).numpy(), rtol=1e-5, atol=1e-5)


        # module_out entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<8x16x16x32x2xbf16>, %arg2: tensor<8x32xbf16>, %arg3: tensor<32x8x16x32x2xbf16>, %arg4: tensor<32x32xbf16>) -> tensor<64x1024xbf16>
        # module_out runtime_fold(%arg0: tensor<512x256xbf16>, %arg1: tensor<256xbf16>, %arg2: tensor<256x1024xbf16>, %arg3: tensor<1024xbf16>) -> (tensor<8x16x16x32x2xbf16>, tensor<8x32xbf16>, tensor<32x8x16x32x2xbf16>, tensor<32x32xbf16>)
        fold_arg0 = arg1
        fold_arg1 = arg2
        fold_arg2 = arg3
        fold_arg3 = arg4
        fold_res0 = np.zeros((8, 16, 16, 32, 2), dtype=ml_dtypes.bfloat16)
        fold_res1 = np.zeros((8, 32), dtype=ml_dtypes.bfloat16)
        fold_res2 = np.zeros((32, 8, 16, 32, 2), dtype=ml_dtypes.bfloat16)
        fold_res3 = np.zeros((32, 32), dtype=ml_dtypes.bfloat16)

        runtime_fold = "runtime_fold"
        fold_mlir_args = get_mlir_args(module_out, runtime_fold, [fold_arg0, fold_arg1, fold_arg2, fold_arg3, fold_res0, fold_res1, fold_res2, fold_res3])

        gc_res_out = np.zeros((64, 1024), dtype=ml_dtypes.bfloat16)
        entry = "entry"
        mlir_args = get_mlir_args(module_out, entry, [arg0, fold_res0, fold_res1, fold_res2, fold_res3, gc_res_out])

        engine_out = compiler.compile_and_jit(module_out, ir_printing=True)
        engine_out.invoke(runtime_fold, *fold_mlir_args)
        engine_out.invoke(entry, *mlir_args)

        assert_allclose(gc_res.astype(np.float32), gc_res_out.astype(np.float32), rtol=1e-5, atol=1e-5)
        
