// RUN: gc-opt %s -decompose-tensor-operation --split-input-file | FileCheck %s

/// CHECK-LABEL: @gather_single_gather_dim
func.func @gather_single_gather_dim(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x1xindex>) -> tensor<2x3x2x2x2xf32> {
  /// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x2x2x2xf32>
  /// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG1:.*]] : tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x2x2x2xf32>)
  %1 = tensor.gather %arg0[%arg1] gather_dims([1]) : (tensor<2x2x2x2xf32>, tensor<2x3x1xindex>) -> tensor<2x3x2x2x2xf32>
  return %1 : tensor<2x3x2x2x2xf32>
}

// -----

/// CHECK-LABEL: @gather_single_gather_dim_no_shrink
func.func @gather_single_gather_dim_no_shrink(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x1xindex>) -> tensor<2x3x2x1x2x2xf32> {
  /// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x2x1x2x2xf32>
  /// CHECK: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG1:.*]] : tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x2x1x2x2xf32>)
  %1 = tensor.gather %arg0[%arg1] gather_dims([1]) : (tensor<2x2x2x2xf32>, tensor<2x3x1xindex>) -> tensor<2x3x2x1x2x2xf32>
  return %1 : tensor<2x3x2x1x2x2xf32>
}

// -----

/// CHECK-LABEL: @gather_multiple_gather_dim
func.func @gather_multiple_gather_dim(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x2xindex>) -> tensor<2x3x2x2xf32> {
  /// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x2x2xf32>
  /// CHECK: %[[EXTRACTSLICE1:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 0] [2, 3, 1] [1, 1, 1] : tensor<2x3x2xindex> to tensor<2x3x1xindex>
  /// CHECK: %[[EXTRACTSLICE2:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 1] [2, 3, 1] [1, 1, 1] : tensor<2x3x2xindex> to tensor<2x3x1xindex>
  /// CHECK: linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXTRACTSLICE1:.*]], %[[EXTRACTSLICE2:.*]] : tensor<2x3x1xindex>, tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x2x2xf32>)
  %1 = tensor.gather %arg0[%arg1] gather_dims([1, 2]) : (tensor<2x2x2x2xf32>, tensor<2x3x2xindex>) -> tensor<2x3x2x2xf32>
  return %1 : tensor<2x3x2x2xf32>
}

// -----

/// CHECK-LABEL: @gather_multiple_gather_dim_no_shrink
func.func @gather_multiple_gather_dim_no_shrink(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x3x2xindex>) -> tensor<2x3x2x1x1x2xf32> {
  /// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x2x1x1x2xf32>
  /// CHECK: %[[EXTRACTSLICE1:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 0] [2, 3, 1] [1, 1, 1] : tensor<2x3x2xindex> to tensor<2x3x1xindex>
  /// CHECK: %[[EXTRACTSLICE2:.*]] = tensor.extract_slice %[[ARG1:.*]][0, 0, 1] [2, 3, 1] [1, 1, 1] : tensor<2x3x2xindex> to tensor<2x3x1xindex>
  /// CHECK: linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[EXTRACTSLICE1:.*]], %[[EXTRACTSLICE2:.*]] : tensor<2x3x1xindex>, tensor<2x3x1xindex>) outs(%[[EMPTY:.*]] : tensor<2x3x2x1x1x2xf32>)
  %1 = tensor.gather %arg0[%arg1] gather_dims([1, 2]) : (tensor<2x2x2x2xf32>, tensor<2x3x2xindex>) -> tensor<2x3x2x1x1x2xf32>
  return %1 : tensor<2x3x2x1x1x2xf32>
}
