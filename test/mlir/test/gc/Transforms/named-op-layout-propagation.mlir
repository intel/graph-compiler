// RUN: gc-opt %s --split-input-file --propagate-layout-on-named-ops | FileCheck %s

// CHECK-LABEL: @matmul_add
func.func @matmul_add(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) outs(%1 : tensor<128x32xf32>) -> tensor<128x32xf32>
  %3 = linalg.broadcast ins(%arg2 : tensor<32xf32>) outs(%0 : tensor<128x32xf32>) dimensions = [0]
  %4 = linalg.add ins(%2, %3 : tensor<128x32xf32>, tensor<128x32xf32>) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  return %4 : tensor<128x32xf32>
}
