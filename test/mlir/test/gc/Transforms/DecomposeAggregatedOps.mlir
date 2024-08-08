// RUN: gc-opt %s -decompose-aggregated-ops | FileCheck %s

// CHECK-LABEL: softmax
func.func @softmax(%arg0: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  %0 = tensor.empty() : tensor<2x2x2x2xf32>
  // CHECK-NOT: linalg.softmax
  // CHECK-COUNT-4: linalg.generic
  %1 = linalg.softmax dimension(3)
    ins(%arg0 : tensor<2x2x2x2xf32>) outs(%0 : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %1 : tensor<2x2x2x2xf32>
}
