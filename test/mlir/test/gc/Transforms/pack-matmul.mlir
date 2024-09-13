// RUN: gc-opt %s --split-input-file --propagate-layout-on-named-ops --post-process-pack-unpack | FileCheck %s

// CHECK-LABEL: @single_matmul_f32
func.func @single_matmul_f32(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  return %2 : tensor<128x32xf32>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: tensor.unpack

// CHECK-LABEL: @single_matmul_bf16
func.func @single_matmul_bf16(%arg0: tensor<128x64xbf16>, %arg1: tensor<64x32xbf16>) -> tensor<128x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xbf16>, tensor<64x32xbf16>) outs(%0 : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  return %2 : tensor<128x32xbf16>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: tensor.unpack
