// RUN: gc-opt %s --split-input-file --propagate-layout-on-named-ops | FileCheck %s

// CHECK-LABEL: @single_matmul_f32
func.func @single_matmul_f32(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  return %2 : tensor<128x32xf32>
}
// CHECK-COUNT-3: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-COUNT-1: tensor.unpack

// CHECK-LABEL: @single_matmul_bf16
func.func @single_matmul_bf16(%arg0: tensor<128x64xbf16>, %arg1: tensor<64x32xbf16>) -> tensor<128x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xbf16>, tensor<64x32xbf16>) outs(%0 : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  return %2 : tensor<128x32xbf16>
}
// CHECK-COUNT-4: tensor.pack
// CHECK-COUNT-1: linalgx.mm4d_vnni
// CHECK-COUNT-1: tensor.unpack

// CHECK-LABEL: @single_batch_matmul_bf16
func.func @single_batch_matmul_bf16(%arg0: tensor<64x128x64xbf16>, %arg1: tensor<64x64x32xbf16>) -> tensor<64x128x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<64x128x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<64x128x32xbf16>) -> tensor<64x128x32xbf16>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x128x64xbf16>, tensor<64x64x32xbf16>) outs(%0 : tensor<64x128x32xbf16>) -> tensor<64x128x32xbf16>
  return %2 : tensor<64x128x32xbf16>
}
// CHECK-COUNT-4: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-COUNT-1: tensor.unpack

func.func @pack_vnni_mmt4d(%arg0: tensor<4x2x32x32xbf16>, %arg1: tensor<1x2x32x32xbf16>) -> tensor<4x1x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<4x1x32x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4x1x32x32xbf16>) -> tensor<4x1x32x32xbf16>
  %2 = linalg.mmt4d ins(%arg0, %arg1 : tensor<4x2x32x32xbf16>, tensor<1x2x32x32xbf16>) outs(%0 : tensor<4x1x32x32xbf16>) -> tensor<4x1x32x32xbf16>
  return %2 : tensor<4x1x32x32xbf16>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalgx.mm4d_vnni

func.func @pack_vnni_batchmmt4d(%arg0: tensor<4x4x2x32x32xbf16>, %arg1: tensor<4x1x2x32x32xbf16>) -> tensor<4x4x1x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<4x4x1x32x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4x4x1x32x32xbf16>) -> tensor<4x4x1x32x32xbf16>
  %2 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<4x4x2x32x32xbf16>, tensor<4x1x2x32x32xbf16>) outs(%0 : tensor<4x4x1x32x32xbf16>) -> tensor<4x4x1x32x32xbf16>
  return %2 : tensor<4x4x1x32x32xbf16>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic

