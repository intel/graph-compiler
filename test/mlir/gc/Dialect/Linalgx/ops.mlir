// RUN: gc-opt %s --verify-roundtrip --split-input-file | FileCheck %s

// Grouped matmul syntax and serialization, without transformation passes.
// Keep the original single-branch syntax, including its unparenthesized result.
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#weights = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#out = affine_map<(e, m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @grouped_single
// CHECK-NEXT:    %[[RESULT:.+]] = linalgx.grouped_matmul
// CHECK-SAME:      ins(%arg0, %arg1, %arg2 : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
// CHECK-SAME:      outs(%arg3 : tensor<?x512xf32>) -> tensor<?x512xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x512xf32>
// CHECK-NEXT:  }
func.func @grouped_single(%a: tensor<?x256xf16>, %w: tensor<8x512x256xf16>,
                          %offs: tensor<8xi32>, %init: tensor<?x512xf32>)
    -> tensor<?x512xf32> {
  %r = linalgx.grouped_matmul
      <{indexing_maps = [#lhs, #weights, #offsets, #out]}>
      ins(%a, %w, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
      outs(%init : tensor<?x512xf32>) -> tensor<?x512xf32>
  return %r : tensor<?x512xf32>
}

// -----

// Multiple branches print (lhs, weights) pairs followed by shared offsets.
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#weights = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#out = affine_map<(e, m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @grouped_shared_lhs
// CHECK-NEXT:    %[[RESULT:.+]]:2 = linalgx.grouped_matmul
// CHECK-SAME:      ins(%arg0, %arg1, %arg0, %arg2, %arg3
// CHECK-SAME:      outs(%arg4, %arg5 : tensor<?x512xf32>, tensor<?x512xf32>)
// CHECK-SAME:      -> (tensor<?x512xf32>, tensor<?x512xf32>)
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<?x512xf32>, tensor<?x512xf32>
// CHECK-NEXT:  }
func.func @grouped_shared_lhs(%a: tensor<?x256xf16>, %w0: tensor<8x512x256xf16>,
                              %w1: tensor<8x512x256xf16>, %offs: tensor<8xi32>,
                              %init0: tensor<?x512xf32>, %init1: tensor<?x512xf32>)
    -> (tensor<?x512xf32>, tensor<?x512xf32>) {
  %r:2 = linalgx.grouped_matmul
      <{indexing_maps = [#lhs, #lhs, #weights, #weights, #offsets, #out, #out]}>
      ins(%a, %w0, %a, %w1, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>,
          tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
      outs(%init0, %init1 : tensor<?x512xf32>, tensor<?x512xf32>)
      -> (tensor<?x512xf32>, tensor<?x512xf32>)
  return %r#0, %r#1 : tensor<?x512xf32>, tensor<?x512xf32>
}

// -----

// The optional scalar bound has no indexing map or entry in the ins type list.
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#weights = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#out = affine_map<(e, m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @grouped_shared_weights
// CHECK-NEXT:    %[[RESULT:.+]]:2 = linalgx.grouped_matmul
// CHECK-SAME:      ins(%arg0, %arg2, %arg1, %arg2, %arg3
// CHECK-SAME:      max_group_size(%arg4)
// CHECK-SAME:      outs(%arg5, %arg6 : tensor<?x512xf32>, tensor<?x512xf32>)
// CHECK-SAME:      -> (tensor<?x512xf32>, tensor<?x512xf32>)
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<?x512xf32>, tensor<?x512xf32>
// CHECK-NEXT:  }
func.func @grouped_shared_weights(%a0: tensor<?x256xf16>, %a1: tensor<?x256xf16>,
                                  %w: tensor<8x512x256xf16>, %offs: tensor<8xi64>,
                                  %bound: index, %init0: tensor<?x512xf32>,
                                  %init1: tensor<?x512xf32>)
    -> (tensor<?x512xf32>, tensor<?x512xf32>) {
  %r:2 = linalgx.grouped_matmul
      <{indexing_maps = [#lhs, #lhs, #weights, #weights, #offsets, #out, #out]}>
      ins(%a0, %w, %a1, %w, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>,
          tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi64>)
      max_group_size(%bound)
      outs(%init0, %init1 : tensor<?x512xf32>, tensor<?x512xf32>)
      -> (tensor<?x512xf32>, tensor<?x512xf32>)
  return %r#0, %r#1 : tensor<?x512xf32>, tensor<?x512xf32>
}
