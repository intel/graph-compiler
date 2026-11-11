// RUN: gc-opt %s | FileCheck %s

// CHECK: linalg.fill {gc.fusion_group = #gc.fusion_group<9223372036854775807>}
func.func @fusion_group_attr() -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<4xf32>
  %filled = linalg.fill {gc.fusion_group = #gc.fusion_group<9223372036854775807>}
      ins(%zero : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %filled : tensor<4xf32>
}
