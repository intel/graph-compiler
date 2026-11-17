// RUN: gc-opt %s | FileCheck %s

// CHECK-LABEL: func.func @fusion_group_attr
// CHECK: linalg.fill {gc.fusion_group = #gc.fusion_group<0, tile_sizes = [32]>}
func.func @fusion_group_attr() -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<4xf32>
  %filled = linalg.fill {gc.fusion_group = #gc.fusion_group<0, tile_sizes = [32]>}
      ins(%zero : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %filled : tensor<4xf32>
}

// CHECK-LABEL: func.func @fusion_group_tile_sizes
// CHECK: linalg.fill {gc.fusion_group = #gc.fusion_group<1, tile_sizes = [32, 64]>}
func.func @fusion_group_tile_sizes() -> tensor<4x8xf32> {
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<4x8xf32>
  %filled = linalg.fill {gc.fusion_group = #gc.fusion_group<1, tile_sizes = [32, 64]>}
      ins(%zero : f32) outs(%out : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %filled : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @fusion_group_empty_tile_sizes
// CHECK: linalg.fill {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
func.func @fusion_group_empty_tile_sizes() -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %out = tensor.empty() : tensor<4xf32>
  %filled = linalg.fill {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
      ins(%zero : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %filled : tensor<4xf32>
}
