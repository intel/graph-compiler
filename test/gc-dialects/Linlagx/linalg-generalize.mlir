// RUN: gc-opt %s --split-input-file --linalg-generalize-named-ops | FileCheck %s


// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[$MAP6:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP7:.+]] = affine_map<(d0, d1) -> (0, d1)>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: mlp
// CHECK-SAME: %[[ARG0:.+]]: tensor<8192x16384xf32>, %[[ARG1:.+]]: tensor<8192xf32>, %[[ARG2:.+]]: tensor<1x128x128xf32>
func.func @mlp(%arg0: tensor<8192x16384xf32>, %arg1: tensor<8192xf32>, %arg2: tensor<1x128x128xf32>) -> tensor<1x8192xf32> {
  // CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg2
  // CHECK-SAME{literal}: [[0], [1, 2]] : tensor<1x128x128xf32> into tensor<1x16384xf32>
  %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2]] : tensor<1x128x128xf32> into tensor<1x16384xf32>
  
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16384x8192xf32>
  %0 = tensor.empty() : tensor<16384x8192xf32>
  
  // CHECK: %[[TRANSPOSED:.+]] = linalg.generic
  // CHECK-SAME: {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} 
  // CHECK-SAME: ins(%arg0 : tensor<8192x16384xf32>) outs(%[[EMPTY]] : tensor<16384x8192xf32>) {
  // CHECK: ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   linalg.yield %[[IN]] : f32
  // CHECK: } -> tensor<16384x8192xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<8192x16384xf32>) outs(%0 : tensor<16384x8192xf32>) permutation = [1, 0] 
  
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x8192xf32>
  %1 = tensor.empty() : tensor<1x8192xf32>
  
  // CHECK: %[[FILLED:.+]] = linalg.generic
  // CHECK-SAME: {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME: ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<1x8192xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   linalg.yield %[[IN]] : f32
  // CHECK: } -> tensor<1x8192xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x8192xf32>) -> tensor<1x8192xf32>

  // CHECK: %[[MMRESULT:.+]] = linalg.generic
  // CHECK-SAME: {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]}
  // CHECK-SAME: ins(%[[COLLAPSE]], %[[TRANSPOSED]] : tensor<1x16384xf32>, tensor<16384x8192xf32>) outs(%[[FILLED]] : tensor<1x8192xf32>)
  // CHECK: ^bb0(%[[IN0:.+]]: f32, %[[IN1:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   %[[MUL:.+]] = arith.mulf %[[IN0]], %[[IN1]] : f32
  // CHECK:   %[[ADD:.+]] = arith.addf %[[OUT]], %[[MUL]] : f32
  // CHECK:   linalg.yield %[[ADD]] : f32
  // CHECK: } -> tensor<1x8192xf32>
  %3 = linalg.matmul ins(%collapsed, %transposed : tensor<1x16384xf32>, tensor<16384x8192xf32>) outs(%2 : tensor<1x8192xf32>) -> tensor<1x8192xf32>
  
  // CHECK: %[[BIAS:.+]] = linalg.generic
  // CHECK-SAME: {indexing_maps = [#map6, #map7, #map1], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME: ins(%arg1, %[[MMRESULT]] : tensor<8192xf32>, tensor<1x8192xf32>) outs(%[[EMPTY]] : tensor<1x8192xf32>) {
  // CHECK: ^bb0(%[[IN0:.+]]: f32, %[[IN1:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   %[[ADD:.+]] = arith.addf %[[IN0]], %[[IN1]] : f32
  // CHECK:   linalg.yield %[[ADD]] : f32
  // CHECK: } -> tensor<1x8192xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1, %3 : tensor<8192xf32>, tensor<1x8192xf32>) outs(%1 : tensor<1x8192xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.addf %in, %in_0 : f32
    linalg.yield %5 : f32
  } -> tensor<1x8192xf32>

  // CHECK: return %[[BIAS]] : tensor<1x8192xf32>
  return %4 : tensor<1x8192xf32>
}
// -----
