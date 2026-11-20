// RUN: gc-opt %s --plan-tile-and-fuse --split-input-file --verify-diagnostics

// Reject partially annotated input with an existing anchor group.
func.func @preannotated_anchor(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %first = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  // expected-error @+1 {{plan-tile-and-fuse requires input without gc.fusion_group attributes}}
  %result = linalg.matmul
    {gc.fusion_group = #gc.fusion_group<7, tile_sizes = [4, 4, 4]>}
    ins(%first, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Reject an existing group on a non-anchor tensor operation.
func.func @preannotated_tensor_op(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<2x2x4xf32> {
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  // expected-error @+1 {{plan-tile-and-fuse requires input without gc.fusion_group attributes}}
  %result = tensor.expand_shape %mm [[0, 1], [2]] output_shape [2, 2, 4]
    {gc.fusion_group = #gc.fusion_group<11, tile_sizes = [2, 2, 4]>}
    : tensor<4x4xf32> into tensor<2x2x4xf32>
  return %result : tensor<2x2x4xf32>
}

// -----

// The precondition also covers the function itself.
// expected-error @+1 {{plan-tile-and-fuse requires input without gc.fusion_group attributes}}
func.func @preannotated_function()
    attributes {gc.fusion_group = #gc.fusion_group<3, tile_sizes = []>} {
  return
}

// -----

// Reject annotations on nested operations even without tensor semantics.
func.func @preannotated_nested(%cond: i1, %x: f32, %y: f32) -> f32 {
  %result = scf.if %cond -> (f32) {
    // expected-error @+1 {{plan-tile-and-fuse requires input without gc.fusion_group attributes}}
    %sum = arith.addf %x, %y
      {gc.fusion_group = #gc.fusion_group<5, tile_sizes = []>} : f32
    scf.yield %sum : f32
  } else {
    scf.yield %x : f32
  }
  return %result : f32
}
