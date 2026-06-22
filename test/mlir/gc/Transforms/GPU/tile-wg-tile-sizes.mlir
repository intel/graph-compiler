// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-attention --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_add
// CHECK: scf.forall
// CHECK:   linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM:]], [[#WGN:]]>
// CHECK:   scf.for
// CHECK:     linalg.matmul {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]], [[#WGK:]]>
// CHECK:   linalg.add {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]]>
func.func @matmul_add(%arg0: tensor<1024x512xf16>, %arg1: tensor<512x1024xf16>) -> tensor<1024x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1024x1024xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x512xf16>, tensor<512x1024xf16>) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %3 = linalg.add ins(%2, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  return %3 : tensor<1024x1024xf16>
}

// -----

// CHECK-LABEL: func.func @transpose_matmul_add
// CHECK: scf.forall
// CHECK:   linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM:]], [[#WGN:]]>
// CHECK:   scf.for
// CHECK:     linalg.transpose {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGK:]], [[#WGN]]>
// CHECK:     linalg.matmul {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]], [[#WGK]]>
// CHECK:   linalg.add {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]]>
func.func @transpose_matmul_add(%arg0: tensor<1024x512xf16>, %argB: tensor<1024x512xf16>) -> tensor<1024x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1024x1024xf16>
  %bt_init = tensor.empty() : tensor<512x1024xf16>
  %bt = linalg.transpose ins(%argB : tensor<1024x512xf16>) outs(%bt_init : tensor<512x1024xf16>) permutation = [1, 0]
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %2 = linalg.matmul ins(%arg0, %bt : tensor<1024x512xf16>, tensor<512x1024xf16>) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %3 = linalg.add ins(%2, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  return %3 : tensor<1024x1024xf16>
}

// -----

// CHECK-LABEL: func.func @attention
// CHECK: scf.forall
// CHECK:   linalgx.attention {{.*}}gc.tiling.wg_tile_sizes = array<i64: 1, 1, 128, 0, 0, 0>
#mapQ = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#mapK = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#mapV = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#mapS = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#mapO = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
func.func @attention(%q: tensor<2x16x1024x64xf16>, %k: tensor<2x16x1024x64xf16>, %v: tensor<2x16x1024x64xf16>) -> tensor<2x16x1024x64xf16> {
  %scale = arith.constant 1.250000e-01 : f16
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<2x16x1024x64xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<2x16x1024x64xf16>) -> tensor<2x16x1024x64xf16>
  %2 = linalgx.attention
       { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO] }
       ins(%q, %k, %v, %scale : tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, f16)
       outs(%1 : tensor<2x16x1024x64xf16>) -> tensor<2x16x1024x64xf16>
  return %2 : tensor<2x16x1024x64xf16>
}
