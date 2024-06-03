// RUN: gc-opt --split-input-file -pass-pipeline="builtin.module(constant-subgraph-analysis,constant-tensor-folding)" %s | FileCheck %s

// CHECK-LABEL: func.func @entry
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // COM: A two-layer mlp. arg0: input feature. arg1: weight of #1 linear. arg2: bias of #1 linear.
  // COM: arg3: weight of #2 linear. arg4: bias of #2 linear.
  func.func @entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<512x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x1024xbf16>, %arg4: tensor<1024xbf16>) -> tensor<64x1024xbf16> attributes {llvm.emit_c_interface, onednn_graph.const_args = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} {
    %1 = tensor.empty() : tensor<2x16x32x32xbf16>
    %packed_arg0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<64x512xbf16> -> tensor<2x16x32x32xbf16>
    %2 = tensor.empty() : tensor<8x16x32x32xbf16>
    %packed_arg1 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<512x256xbf16> -> tensor<8x16x32x32xbf16>
    %3 = tensor.empty() : tensor<8x16x16x32x2xbf16>
    %packed_packed_arg1 = tensor.pack %packed_arg1 inner_dims_pos = [2] inner_tiles = [2] into %3 : tensor<8x16x32x32xbf16> -> tensor<8x16x16x32x2xbf16>
    %4 = tensor.empty() : tensor<2x8x32x32xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %5 = linalg.fill ins(%cst_0 : bf16) outs(%4 : tensor<2x8x32x32xbf16>) -> tensor<2x8x32x32xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%packed_arg0, %packed_packed_arg1 : tensor<2x16x32x32xbf16>, tensor<8x16x16x32x2xbf16>) outs(%5 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %44 = arith.mulf %in, %in_0 : bf16
      %55 = arith.addf %out, %44 : bf16
      linalg.yield %55 : bf16
    } -> tensor<2x8x32x32xbf16>
    %15 = tensor.empty() : tensor<8x32xbf16>
    %packed_arg2 = tensor.pack %arg2 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %15 : tensor<256xbf16> -> tensor<8x32xbf16>
    %bc_arg2_init = tensor.empty() : tensor<2x8x32x32xbf16>
    %bc_arg2 = linalg.broadcast ins(%packed_arg2 : tensor<8x32xbf16>) outs(%bc_arg2_init : tensor<2x8x32x32xbf16>) dimensions = [0, 2]
    %extf32 = arith.extf %bc_arg2 : tensor<2x8x32x32xbf16> to tensor<2x8x32x32xf32>
    %cst_2 = arith.constant 2.000000e+00 : f32
    %extf32_mul2_init = tensor.empty() : tensor<2x8x32x32xf32>
    %extf32_mul2 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extf32 : tensor<2x8x32x32xf32>) outs(%extf32_mul2_init : tensor<2x8x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %cst_2 : f32
      linalg.yield %8 : f32
    } -> tensor<2x8x32x32xf32>
    %truncbf16 = arith.truncf %extf32_mul2 : tensor<2x8x32x32xf32> to tensor<2x8x32x32xbf16>
    %7 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%truncbf16 : tensor<2x8x32x32xbf16>) outs(%6 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %45 = arith.addf %in, %out : bf16
      linalg.yield %45 : bf16
    } -> tensor<2x8x32x32xbf16>
    %8 = tensor.empty() : tensor<32x8x32x32xbf16>
    %packed_arg3 = tensor.pack %arg3 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %8 : tensor<256x1024xbf16> -> tensor<32x8x32x32xbf16>
    %9 = tensor.empty() : tensor<32x8x16x32x2xbf16>
    %packed_packed_arg3 = tensor.pack %packed_arg3 inner_dims_pos = [2] inner_tiles = [2] into %9 : tensor<32x8x32x32xbf16> -> tensor<32x8x16x32x2xbf16>
    %10 = tensor.empty() : tensor<2x32x32x32xbf16>
    %11 = linalg.fill ins(%cst_0 : bf16) outs(%10 : tensor<2x32x32x32xbf16>) -> tensor<2x32x32x32xbf16>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%7, %packed_packed_arg3 : tensor<2x8x32x32xbf16>, tensor<32x8x16x32x2xbf16>) outs(%11 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %46 = arith.mulf %in, %in_0 : bf16
      %56 = arith.addf %out, %46 : bf16
      linalg.yield %56 : bf16
    } -> tensor<2x32x32x32xbf16>
    %16 = tensor.empty() : tensor<32x32xbf16>
    %packed_arg4 = tensor.pack %arg4 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %16 : tensor<1024xbf16> -> tensor<32x32xbf16>
    %13 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%packed_arg4 : tensor<32x32xbf16>) outs(%12 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %47 = arith.addf %in, %out : bf16
      linalg.yield %47 : bf16
    } -> tensor<2x32x32x32xbf16>
    %14 = tensor.empty() : tensor<64x1024xbf16>
    %unpack = tensor.unpack %13 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %14 : tensor<2x32x32x32xbf16> -> tensor<64x1024xbf16>
    return %unpack : tensor<64x1024xbf16>
  }
}
// CHECK: linalg.broadcast
// CHECK: func.func @fold
// CHECK: arith.extf
// CHECK: arith.truncf

// COM: expected output:
// COM: module {
// COM:   llvm.mlir.global external constant @__num_orig_num_args(5 : i32) {addr_space = 0 : i32} : i32
// COM:   llvm.mlir.global external constant @__compute_args(dense<[5, 0, 5, 6, 7, 8]> : tensor<6xi32>) {addr_space = 0 : i32} : !llvm.array<6 x i32>
// COM:   llvm.mlir.global external constant @__fold_args(dense<[8, 1, 2, 3, 4, 5, 6, 7, 8]> : tensor<9xi32>) {addr_space = 0 : i32} : !llvm.array<9 x i32>
// COM:   llvm.mlir.global external constant @__fold_buffer_ids(dense<[4, 0, 1, 2, 3]> : tensor<5xi64>) {addr_space = 0 : i32} : !llvm.array<5 x i64>
// COM:   func.func @entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<8x16x16x32x2xbf16>, %arg2: tensor<8x32xbf16>, %arg3: tensor<32x8x16x32x2xbf16>, %arg4: tensor<32x32xbf16>) -> tensor<64x1024xbf16> attributes {llvm.emit_c_interface, onednn_graph.const_args = [1 : i32, 2 : i32, 3 : i32, 4 : i32]}
// COM:   func.func @fold(%arg0: tensor<512x256xbf16>, %arg1: tensor<256xbf16>, %arg2: tensor<256x1024xbf16>, %arg3: tensor<1024xbf16>) -> (tensor<8x16x16x32x2xbf16>, tensor<8x32xbf16>, tensor<32x8x16x32x2xbf16>, tensor<32x32xbf16>) attributes {llvm.emit_c_interface}
