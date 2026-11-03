// RUN: gc-opt %s --split-input-file --verify-diagnostics

#l = affine_map<(e, m, n, k) -> (m, k)>
#w = affine_map<(e, m, n, k) -> (e, n, k)>
#o = affine_map<(e, m, n, k) -> (e)>
#r = affine_map<(e, m, n, k) -> (m, n)>
func.func @no_shared_matrix(%a0: tensor<5x64xf16>, %a1: tensor<5x64xf16>,
                            %w0: tensor<3x128x64xf16>, %w1: tensor<3x128x64xf16>,
                            %offs: tensor<3xi32>, %c0: tensor<5x128xf32>, %c1: tensor<5x128xf32>) {
  // expected-error @+1 {{fused branches must share their lhs or weights value}}
  %r:2 = linalgx.grouped_matmul <{indexing_maps = [#l, #l, #w, #w, #o, #r, #r]}>
      ins(%a0, %w0, %a1, %w1, %offs : tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<3xi32>)
      outs(%c0, %c1 : tensor<5x128xf32>, tensor<5x128xf32>) -> (tensor<5x128xf32>, tensor<5x128xf32>)
  return
}

// -----

#l = affine_map<(e, m, n, k) -> (m, k)>
#w = affine_map<(e, m, n, k) -> (e, n, k)>
#o = affine_map<(e, m, n, k) -> (e)>
#r = affine_map<(e, m, n, k) -> (m, n)>
func.func @different_widths(%a: tensor<5x64xf16>, %w0: tensor<3x128x64xf16>,
                            %w1: tensor<3x256x64xf16>, %offs: tensor<3xi32>,
                            %c0: tensor<5x128xf32>, %c1: tensor<5x256xf32>) {
  // expected-error @+1 {{fused branches must have a common iteration domain}}
  %r:2 = linalgx.grouped_matmul <{indexing_maps = [#l, #l, #w, #w, #o, #r, #r]}>
      ins(%a, %w0, %a, %w1, %offs : tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<5x64xf16>, tensor<3x256x64xf16>, tensor<3xi32>)
      outs(%c0, %c1 : tensor<5x128xf32>, tensor<5x256xf32>) -> (tensor<5x128xf32>, tensor<5x256xf32>)
  return
}

// -----

#l = affine_map<(e, m, n, k) -> (m, k)>
#w = affine_map<(e, m, n, k) -> (e, n, k)>
#o = affine_map<(e, m, n, k) -> (e)>
#r = affine_map<(e, m, n, k) -> (m, n)>
func.func @different_types(%a: tensor<5x64xf16>, %w0: tensor<3x128x64xf16>,
                           %w1: tensor<3x128x64xf32>, %offs: tensor<3xi32>,
                           %c0: tensor<5x128xf32>, %c1: tensor<5x128xf32>) {
  // expected-error @+1 {{fused branches must have matching element types}}
  %r:2 = linalgx.grouped_matmul <{indexing_maps = [#l, #l, #w, #w, #o, #r, #r]}>
      ins(%a, %w0, %a, %w1, %offs : tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<5x64xf16>, tensor<3x128x64xf32>, tensor<3xi32>)
      outs(%c0, %c1 : tensor<5x128xf32>, tensor<5x128xf32>) -> (tensor<5x128xf32>, tensor<5x128xf32>)
  return
}

// -----

#l = affine_map<(e, m, n, k) -> (m, k)>
#w = affine_map<(e, m, n, k) -> (e, n, k)>
#o = affine_map<(e, m, n, k) -> (e)>
#r = affine_map<(e, m, n, k) -> (m, n)>
func.func @different_branch_counts(%a: tensor<5x64xf16>, %w0: tensor<3x128x64xf16>,
                                   %offs: tensor<3xi32>, %c0: tensor<5x128xf32>, %c1: tensor<5x128xf32>) {
  // expected-error @+1 {{expects equally sized, non-empty lhs, weights and outputs lists}}
  %r:2 = linalgx.grouped_matmul <{indexing_maps = [#l, #w, #o, #r, #r]}>
      ins(%a, %w0, %offs : tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<3xi32>)
      outs(%c0, %c1 : tensor<5x128xf32>, tensor<5x128xf32>) -> (tensor<5x128xf32>, tensor<5x128xf32>)
  return
}

// -----

#l = affine_map<(e, m, n, k) -> (m, k)>
#w = affine_map<(e, m, n, k) -> (e, n, k)>
#o = affine_map<(e, m, n, k) -> (e)>
#r = affine_map<(e, m, n, k) -> (m, n)>
func.func @wrong_map_count(%a: tensor<5x64xf16>, %w0: tensor<3x128x64xf16>,
                           %w1: tensor<3x128x64xf16>, %offs: tensor<3xi32>,
                           %c0: tensor<5x128xf32>, %c1: tensor<5x128xf32>) {
  // expected-error @+1 {{expects one indexing map per shaped operand}}
  %r:2 = linalgx.grouped_matmul <{indexing_maps = [#l, #w, #o, #r]}>
      ins(%a, %w0, %a, %w1, %offs : tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<5x64xf16>, tensor<3x128x64xf16>, tensor<3xi32>)
      outs(%c0, %c1 : tensor<5x128xf32>, tensor<5x128xf32>) -> (tensor<5x128xf32>, tensor<5x128xf32>)
  return
}
