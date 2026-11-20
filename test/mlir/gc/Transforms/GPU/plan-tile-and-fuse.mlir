// RUN: gc-opt %s --plan-tile-and-fuse --split-input-file | FileCheck %s

// Fuse the single-user fill and the consumer with the anchor. The empty has
// two distinct users, even after both join the group, and stays separate.
// CHECK-LABEL: func.func @matmul_with_neighbors(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[BIAS:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty()
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      : tensor<4x4xf32>
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[MM:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = linalg.elementwise <add>
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[MM]], %[[BIAS]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @matmul_with_neighbors(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %bias: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %init = linalg.fill
    ins(%zero : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.elementwise <add>
    ins(%mm, %bias : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Fuse tensor reshapes in both directions. Backward expansion reaches the
// empty through the fill on a later iteration; all members share the anchor ID.
// CHECK-LABEL: func.func @reshapes_around_matmul(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<2x4x8xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<8x16xf32>) -> tensor<2x4x16xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[LHS:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0, 1], [2]]
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      : tensor<2x4x8xf32> into tensor<8x8xf32>
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty()
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      : tensor<8x16xf32>
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT:    %[[MM:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[LHS]], %[[B]] : tensor<8x8xf32>, tensor<8x16xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = tensor.expand_shape %[[MM]] {{\[}}[0, 1], [2]]
// CHECK-SAME:      output_shape [2, 4, 16]
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      : tensor<8x16xf32> into tensor<2x4x16xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<2x4x16xf32>
// CHECK-NEXT:  }
func.func @reshapes_around_matmul(
    %a: tensor<2x4x8xf32>, %b: tensor<8x16xf32>) -> tensor<2x4x16xf32> {
  %zero = arith.constant 0.0 : f32
  %lhs = tensor.collapse_shape %a [[0, 1], [2]]
    : tensor<2x4x8xf32> into tensor<8x8xf32>
  %empty = tensor.empty() : tensor<8x16xf32>
  %init = linalg.fill
    ins(%zero : f32) outs(%empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  %mm = linalg.matmul
    ins(%lhs, %b : tensor<8x8xf32>, tensor<8x16xf32>)
    outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
  %result = tensor.expand_shape %mm [[0, 1], [2]] output_shape [2, 4, 16]
    : tensor<8x16xf32> into tensor<2x4x16xf32>
  return %result : tensor<2x4x16xf32>
}

// -----

// Preserve the grouped matmul's dynamic shapes, indexing maps, and hint.
#gl = affine_map<(e, m, n, k) -> (m, k)>
#gw = affine_map<(e, m, n, k) -> (e, n, k)>
#go = affine_map<(e, m, n, k) -> (e)>
#gr = affine_map<(e, m, n, k) -> (m, n)>

// CHECK-DAG: #[[GL:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[GW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[GO:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK-DAG: #[[GR:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-LABEL: func.func @grouped_matmul(
// CHECK-SAME:    %[[LHS:[^:]+]]: tensor<?x4xf32>,
// CHECK-SAME:    %[[WEIGHTS:[^:]+]]: tensor<2x4x4xf32>,
// CHECK-SAME:    %[[OFFSETS:[^:]+]]: tensor<2xi32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<?x4xf32>,
// CHECK-SAME:    %[[MAX:[^:]+]]: index) -> tensor<?x4xf32> {
// CHECK-NEXT:    %[[RESULT:.+]] = linalgx.grouped_matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      <indexing_maps = [#[[GL]], #[[GW]], #[[GO]], #[[GR]]]>
// CHECK-SAME:      ins(%[[LHS]], %[[WEIGHTS]], %[[OFFSETS]] :
// CHECK-SAME:        tensor<?x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>)
// CHECK-SAME:      max_group_size(%[[MAX]])
// CHECK-SAME:      outs(%[[OUT]] : tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x4xf32>
// CHECK-NEXT:  }
func.func @grouped_matmul(
    %lhs: tensor<?x4xf32>, %weights: tensor<2x4x4xf32>,
    %offsets: tensor<2xi32>, %out: tensor<?x4xf32>, %max: index)
    -> tensor<?x4xf32> {
  %result = linalgx.grouped_matmul
    <{indexing_maps = [#gl, #gw, #go, #gr]}>
    ins(%lhs, %weights, %offsets :
      tensor<?x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>)
    max_group_size(%max)
    outs(%out : tensor<?x4xf32>) -> tensor<?x4xf32>
  return %result : tensor<?x4xf32>
}

// -----

// An attention anchor receives a group without being decomposed or tiled.
#q = affine_map<(m, k1, k2, n) -> (m, k1)>
#k = affine_map<(m, k1, k2, n) -> (k2, k1)>
#v = affine_map<(m, k1, k2, n) -> (k2, n)>
#s = affine_map<(m, k1, k2, n) -> ()>
#o = affine_map<(m, k1, k2, n) -> (m, n)>

// CHECK-DAG: #[[Q:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-DAG: #[[K:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK-DAG: #[[V:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[S:.+]] = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK-DAG: #[[O:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK-LABEL: func.func @attention(
// CHECK-SAME:    %[[QUERY:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[KEY:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[VALUE:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[SCALE:.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:    %[[RESULT:.+]] = linalgx.attention
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      <indexing_maps = [#[[Q]], #[[K]], #[[V]], #[[S]], #[[O]]]>
// CHECK-SAME:      ins(%[[QUERY]], %[[KEY]], %[[VALUE]], %[[SCALE]] :
// CHECK-SAME:        tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, f32)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @attention(
    %query: tensor<4x4xf32>, %key: tensor<4x4xf32>,
    %value: tensor<4x4xf32>, %out: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %scale = arith.constant 0.5 : f32
  %result = linalgx.attention
    <{indexing_maps = [#q, #k, #v, #s, #o]}>
    ins(%query, %key, %value, %scale :
      tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, f32)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Connected anchors of different kinds still receive separate groups, in IR
// order. The fill before them is assigned only after all three anchors.
#gl = affine_map<(e, m, n, k) -> (m, k)>
#gw = affine_map<(e, m, n, k) -> (e, n, k)>
#go = affine_map<(e, m, n, k) -> (e)>
#gr = affine_map<(e, m, n, k) -> (m, n)>
#q = affine_map<(m, k1, k2, n) -> (m, k1)>
#k = affine_map<(m, k1, k2, n) -> (k2, k1)>
#v = affine_map<(m, k1, k2, n) -> (k2, n)>
#s = affine_map<(m, k1, k2, n) -> ()>
#o = affine_map<(m, k1, k2, n) -> (m, n)>
#mask = affine_map<(m, k1, k2, n) -> (m, k2)>

// CHECK-DAG: #[[GL:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[GW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[GO:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK-DAG: #[[GR:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[Q:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-DAG: #[[K:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK-DAG: #[[V:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[S:.+]] = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK-DAG: #[[O:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK-DAG: #[[M:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-LABEL: func.func @multiple_anchors(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[WEIGHTS:[^:]+]]: tensor<2x4x4xf32>,
// CHECK-SAME:    %[[OFFSETS:[^:]+]]: tensor<2xi32>,
// CHECK-SAME:    %[[MASK:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[SCALE:.+]] = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty()
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<3, tile_sizes = []>}
// CHECK-SAME:      : tensor<4x4xf32>
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<4, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[MM:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[GMM:.+]] = linalgx.grouped_matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      <indexing_maps = [#[[GL]], #[[GW]], #[[GO]], #[[GR]]]>
// CHECK-SAME:      ins(%[[MM]], %[[WEIGHTS]], %[[OFFSETS]] :
// CHECK-SAME:        tensor<4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = linalgx.attention
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
// CHECK-SAME:      <indexing_maps = [#[[Q]], #[[K]], #[[V]], #[[S]], #[[M]], #[[O]]]>
// CHECK-SAME:      ins(%[[GMM]], %[[B]], %[[A]], %[[SCALE]], %[[MASK]] :
// CHECK-SAME:        tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
// CHECK-SAME:        f32, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @multiple_anchors(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>,
    %weights: tensor<2x4x4xf32>, %offsets: tensor<2xi32>,
    %mask: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 0.5 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %init = linalg.fill
    ins(%zero : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %gmm = linalgx.grouped_matmul
    <{indexing_maps = [#gl, #gw, #go, #gr]}>
    ins(%mm, %weights, %offsets :
      tensor<4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>)
    outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalgx.attention
    <{indexing_maps = [#q, #k, #v, #s, #mask, #o]}>
    ins(%gmm, %b, %a, %scale, %mask :
      tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, f32, tensor<4x4xf32>)
    outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// The same contraction predicate as the tiler recognizes generic matmuls.
// CHECK-DAG: #[[ML:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MR:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[MO:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @generic_matmul(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      {indexing_maps = [#[[ML]], #[[MR]], #[[MO]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>)
// CHECK-SAME:      attrs = {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>,
// CHECK-SAME:               test.marker = "keep"} {
// CHECK-NEXT:    ^bb0(%[[X:[^:]+]]: f32, %[[Y:[^:]+]]: f32, %[[ACC:[^:]+]]: f32):
// CHECK-NEXT:      %[[PRODUCT:.+]] = arith.mulf %[[X]], %[[Y]] : f32
// CHECK-NEXT:      %[[SUM:.+]] = arith.addf %[[PRODUCT]], %[[ACC]] : f32
// CHECK-NEXT:      linalg.yield %[[SUM]] : f32
// CHECK-NEXT:    } -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @generic_matmul(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill
    ins(%zero : f32) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%init : tensor<4x4xf32>)
    attrs = {test.marker = "keep"} {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %product = arith.mulf %x, %y : f32
    %sum = arith.addf %product, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Without anchors, named and generic Linalg operations get singleton IDs in
// IR order. Arithmetic inside the generic is left unannotated.
// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @no_anchors(
// CHECK-SAME:    %[[INPUT:[^:]+]]: tensor<4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      {indexing_maps = [#[[IDENTITY]], #[[IDENTITY]]],
// CHECK-SAME:       iterator_types = ["parallel"]}
// CHECK-SAME:      ins(%[[INPUT]] : tensor<4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4xf32>)
// CHECK-SAME:      attrs = {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>} {
// CHECK-NEXT:    ^bb0(%[[X:[^:]+]]: f32, %[[ACC:[^:]+]]: f32):
// CHECK-NEXT:      %[[SUM:.+]] = arith.addf %[[X]], %[[ACC]] : f32
// CHECK-NEXT:      linalg.yield %[[SUM]] : f32
// CHECK-NEXT:    } -> tensor<4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xf32>
// CHECK-NEXT:  }
func.func @no_anchors(%input: tensor<4xf32>, %out: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill
    ins(%zero : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%input : tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %x, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// Tensor operations without anchors are grouped, including scalar results
// from tensor.dim and tensor.extract. Scalar arithmetic is left unchanged.
// CHECK-LABEL: func.func @tensor_ops_without_anchors(
// CHECK-SAME:    %[[INPUT:[^:]+]]: tensor<?xf32>,
// CHECK-SAME:    %[[SIZE:[^:]+]]: index) -> (tensor<4xf32>, index, f32) {
// CHECK-NEXT:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty(%[[SIZE]])
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      : tensor<?xf32>
// CHECK-NEXT:    %[[CAST:.+]] = tensor.cast %[[EMPTY]]
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      : tensor<?xf32> to tensor<4xf32>
// CHECK-NEXT:    %[[DIM:.+]] = tensor.dim
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
// CHECK-SAME:      %[[INPUT]], %[[C0]] : tensor<?xf32>
// CHECK-NEXT:    %[[ELEMENT:.+]] = tensor.extract %[[INPUT]][%[[C0]]]
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<3, tile_sizes = []>}
// CHECK-SAME:      : tensor<?xf32>
// CHECK-NEXT:    return %[[CAST]], %[[DIM]], %[[ELEMENT]] : tensor<4xf32>, index, f32
// CHECK-NEXT:  }
func.func @tensor_ops_without_anchors(%input: tensor<?xf32>, %size: index)
    -> (tensor<4xf32>, index, f32) {
  %c0 = arith.constant 0 : index
  %empty = tensor.empty(%size) : tensor<?xf32>
  %cast = tensor.cast %empty : tensor<?xf32> to tensor<4xf32>
  %dim = tensor.dim %input, %c0 : tensor<?xf32>
  %element = tensor.extract %input[%c0] : tensor<?xf32>
  return %cast, %dim, %element : tensor<4xf32>, index, f32
}

// -----

// Tensor-valued control flow and its nested operations receive distinct groups.
// CHECK-LABEL: func.func @nested_regions(
// CHECK-SAME:    %[[COND:[^:]+]]: i1,
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = scf.if %[[COND]] -> (tensor<4x4xf32>) {
// CHECK-NEXT:      %[[MM:.+]] = linalg.matmul
// CHECK-SAME:        {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:        ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:        outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      scf.yield %[[MM]] : tensor<4x4xf32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[FILL:.+]] = linalg.fill
// CHECK-SAME:        {gc.fusion_group = #gc.fusion_group<3, tile_sizes = []>}
// CHECK-SAME:        ins(%[[ZERO]] : f32)
// CHECK-SAME:        outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      scf.yield %[[FILL]] : tensor<4x4xf32>
// CHECK-NEXT:    } {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @nested_regions(
    %cond: i1, %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill
    ins(%zero : f32) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = scf.if %cond -> (tensor<4x4xf32>) {
    %mm = linalg.matmul
      ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %mm : tensor<4x4xf32>
  } else {
    %fill = linalg.fill
      ins(%zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.yield %fill : tensor<4x4xf32>
  }
  return %result : tensor<4x4xf32>
}

// -----

// Operations with buffer semantics are left unchanged by the epilogue.
// CHECK-LABEL: func.func @buffer_fill(
// CHECK-SAME:    %[[VALUE:[^:]+]]: f32, %[[OUT:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT:    linalg.fill ins(%[[VALUE]] : f32) outs(%[[OUT]] : memref<4xf32>)
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @buffer_fill(%value: f32, %out: memref<4xf32>) {
  linalg.fill ins(%value : f32) outs(%out : memref<4xf32>)
  return
}

// -----

// Functions with no target operations are unchanged.
// CHECK-LABEL: func.func @no_linalg(
// CHECK-SAME:    %[[X:[^:]+]]: f32, %[[Y:[^:]+]]: f32) -> f32 {
// CHECK-NEXT:    %[[SUM:.+]] = arith.addf %[[X]], %[[Y]] : f32
// CHECK-NEXT:    return %[[SUM]] : f32
// CHECK-NEXT:  }
func.func @no_linalg(%x: f32, %y: f32) -> f32 {
  %sum = arith.addf %x, %y : f32
  return %sum : f32
}

// -----

// IDs restart for each function even when both are in the same module.
// CHECK-LABEL: func.func @first_function(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @first_function(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill
    ins(%zero : f32) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @second_function(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[RESULT:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @second_function(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %result = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Repeated operands count as one producer user. The earlier anchor claims the
// middle operation as a consumer before the later anchor can claim it as a
// producer. The tail cannot join group 0: its input from group 1 depends on
// group 0. It can join group 1 with an independent input from group 0 instead.
// CHECK-LABEL: func.func @repeated_uses_and_ownership(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>)
// CHECK-SAME:    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
// CHECK-NEXT:    %[[LHS:.+]] = linalg.elementwise <add>
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[FIRST:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[LHS]], %[[LHS]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[MIDDLE:.+]] = linalg.elementwise <add>
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[FIRST]], %[[FIRST]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[SECOND:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      ins(%[[MIDDLE]], %[[FIRST]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[TAIL:.+]] = linalg.elementwise <add>
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      ins(%[[SECOND]], %[[FIRST]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %[[FIRST]], %[[TAIL]] : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK-NEXT:  }
func.func @repeated_uses_and_ownership(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %lhs = linalg.elementwise <add>
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %first = linalg.matmul
    ins(%lhs, %lhs : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %middle = linalg.elementwise <add>
    ins(%first, %first : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %second = linalg.matmul
    ins(%middle, %first : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %tail = linalg.elementwise <add>
    ins(%second, %first : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %first, %tail : tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// A group whose inputs are block arguments and whose result is unused has no
// producer or consumer candidates. The driver still converges and annotates it.
// CHECK-LABEL: func.func @anchor_without_candidates(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[OUT:[^:]+]]: tensor<4x4xf32>) {
// CHECK-NEXT:    %{{.+}} = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @anchor_without_candidates(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>) {
  %unused = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

// -----

// Adding the consumer exposes an independent producer chain on subsequent
// iterations. A disconnected, unused tensor operation remains a singleton.
// CHECK-LABEL: func.func @producers_exposed_by_consumer(
// CHECK: tensor.cast {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: tensor.cast {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: tensor.cast {{.*}}gc.fusion_group<1, tile_sizes = []>
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: linalg.elementwise <add> {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: return
func.func @producers_exposed_by_consumer(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %bias: tensor<4x4xf32>,
    %out: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %seed = tensor.cast %bias : tensor<4x4xf32> to tensor<4x4xf32>
  %independent = tensor.cast %seed : tensor<4x4xf32> to tensor<4x4xf32>
  %disconnected = tensor.cast %b : tensor<4x4xf32> to tensor<4x4xf32>
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.elementwise <add>
    ins(%mm, %independent : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Consumers can have multiple users, including returns. Forward expansion
// reaches the second cast on a later iteration and never absorbs the return.
// CHECK-LABEL: func.func @consumer_multiple_users(
// CHECK: %[[MM:.+]] = linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: %[[FIRST:.+]] = tensor.cast %[[MM]]
// CHECK-SAME: gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: %[[SECOND:.+]] = tensor.cast %[[FIRST]]
// CHECK-SAME: gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: return %[[MM]], %[[FIRST]], %[[SECOND]]
func.func @consumer_multiple_users(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %first = tensor.cast %mm : tensor<4x4xf32> to tensor<4x4xf32>
  %second = tensor.cast %first : tensor<4x4xf32> to tensor<4x4xf32>
  return %mm, %first, %second : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// Distinct users are counted across every result of a producer. Two results
// used by the same anchor still have exactly one user, so the producer joins.
// CHECK-LABEL: func.func @multi_result_single_user(
// CHECK: %[[PAIR:.+]]:2 = linalg.generic
// CHECK-SAME: gc.fusion_group<0, tile_sizes = []>
// CHECK: linalg.yield
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-SAME: ins(%[[PAIR]]#0, %[[PAIR]]#1
func.func @multi_result_single_user(
    %a: tensor<4x4xf32>, %out: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %pair:2 = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%a : tensor<4x4xf32>)
    outs(%out, %out : tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%x: f32, %u: f32, %v: f32):
    linalg.yield %x, %x : f32, f32
  } -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %mm = linalg.matmul
    ins(%pair#0, %pair#1 : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %mm : tensor<4x4xf32>
}

// -----

// An external return of the other result prevents producer fusion. Nothing is
// cloned, and the producer remains a singleton.
// CHECK-LABEL: func.func @multi_result_external_user(
// CHECK: %[[PAIR:.+]]:2 = linalg.generic
// CHECK-SAME: gc.fusion_group<1, tile_sizes = []>
// CHECK: %[[MM:.+]] = linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: return %[[MM]], %[[PAIR]]#1
func.func @multi_result_external_user(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %pair:2 = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%a : tensor<4x4xf32>)
    outs(%out, %out : tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%x: f32, %u: f32, %v: f32):
    linalg.yield %x, %x : f32, f32
  } -> (tensor<4x4xf32>, tensor<4x4xf32>)
  %mm = linalg.matmul
    ins(%pair#0, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %mm, %pair#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// Neither producer nor consumer fusion crosses block boundaries.
// CHECK-LABEL: func.func @cross_block_neighbors(
// CHECK: tensor.cast {{.*}}gc.fusion_group<1, tile_sizes = []>
// CHECK-NEXT: cf.br
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: cf.br
// CHECK: tensor.cast {{.*}}gc.fusion_group<2, tile_sizes = []>
// CHECK-NEXT: return
func.func @cross_block_neighbors(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %producer = tensor.cast %a : tensor<4x4xf32> to tensor<4x4xf32>
  cf.br ^anchor
^anchor:
  %mm = linalg.matmul
    ins(%producer, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  cf.br ^consumer
^consumer:
  %result = tensor.cast %mm : tensor<4x4xf32> to tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// A call with unknown effects cannot join a group, in either direction.
// CHECK-LABEL: func.func @effectful_neighbors(
// CHECK: call @tensor_effect{{.*}}gc.fusion_group<1, tile_sizes = []>
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: call @tensor_effect{{.*}}gc.fusion_group<2, tile_sizes = []>
// CHECK-NEXT: return
func.func @effectful_neighbors(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %producer = func.call @tensor_effect(%a)
    : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm = linalg.matmul
    ins(%producer, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %consumer = func.call @tensor_effect(%mm)
    : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %consumer : tensor<4x4xf32>
}
func.func private @tensor_effect(tensor<4x4xf32>) -> tensor<4x4xf32>

// -----

// Inspect Linalg payloads as well as the outer operation's effects.
// CHECK-LABEL: func.func @effectful_linalg_payload(
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: linalg.generic
// CHECK-SAME: gc.fusion_group<1, tile_sizes = []>
// CHECK: func.call @scalar_effect
// CHECK-NEXT: linalg.yield
func.func @effectful_linalg_payload(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm : tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %value = func.call @scalar_effect(%x) : (f32) -> f32
    linalg.yield %value : f32
  } -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
func.func private @scalar_effect(f32) -> f32

// -----

// Reject a Linalg region capture that is absent from the operation's operands.
// CHECK-LABEL: func.func @unrepresented_capture(
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: linalg.generic
// CHECK-SAME: gc.fusion_group<1, tile_sizes = []>
func.func @unrepresented_capture(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>, %extra: f32)
    -> tensor<4x4xf32> {
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm : tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %value = arith.addf %x, %extra : f32
    linalg.yield %value : f32
  } -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// An independent input defined by a non-fusible operation is allowed. The
// consumer joins the anchor; the effectful producer remains outside.
// CHECK-LABEL: func.func @independent_nonfusible_input(
// CHECK: call @independent_effect{{.*}}gc.fusion_group<1, tile_sizes = []>
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: linalg.elementwise <add> {{.*}}gc.fusion_group<0, tile_sizes = []>
func.func @independent_nonfusible_input(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %external = func.call @independent_effect(%b)
    : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.elementwise <add>
    ins(%mm, %external : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
func.func private @independent_effect(tensor<4x4xf32>) -> tensor<4x4xf32>

// -----

// An alternate path from the anchor through a non-fusible call would create
// group -> call -> group if the consumer joined. Reject it without an error.
// CHECK-LABEL: func.func @cycle_through_nonfusible_operation(
// CHECK: linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK: call @cycle_effect{{.*}}gc.fusion_group<1, tile_sizes = []>
// CHECK: linalg.elementwise <add> {{.*}}gc.fusion_group<2, tile_sizes = []>
func.func @cycle_through_nonfusible_operation(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %external = func.call @cycle_effect(%mm)
    : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.elementwise <add>
    ins(%mm, %external : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
func.func private @cycle_effect(tensor<4x4xf32>) -> tensor<4x4xf32>

// -----

// The alternate path can be carried by a region capture instead of an explicit
// operand. The if and its yielded values also remain outside the anchor group.
// CHECK-LABEL: func.func @cycle_through_region_capture(
// CHECK: %[[MM:.+]] = linalg.matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: scf.if
// CHECK-NEXT: scf.yield %[[MM]]
// CHECK: gc.fusion_group<1, tile_sizes = []>
// CHECK-NEXT: linalg.elementwise <add> {{.*}}gc.fusion_group<2, tile_sizes = []>
func.func @cycle_through_region_capture(
    %cond: i1, %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %mm = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %external = scf.if %cond -> (tensor<4x4xf32>) {
    scf.yield %mm : tensor<4x4xf32>
  } else {
    scf.yield %b : tensor<4x4xf32>
  }
  %result = linalg.elementwise <add>
    ins(%mm, %external : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// -----

// Seeded contraction anchors can capture another anchor's result in their
// payload. That dependency must remain visible through the owned group's
// inputs: absorbing the joint consumer into group 0 would create a cycle
// through group 1. The unsupported captured payload also prevents group 1
// from absorbing the consumer, so it remains a singleton.
// CHECK-LABEL: func.func @cycle_through_seeded_anchor_capture(
// CHECK: %[[FIRST:.+]] = linalg.matmul
// CHECK-SAME: gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: %[[SECOND:.+]] = linalg.generic
// CHECK-SAME: gc.fusion_group<1, tile_sizes = []>
// CHECK: tensor.extract %[[FIRST]]
// CHECK-SAME: gc.fusion_group<2, tile_sizes = []>
// CHECK: %[[CONSUMER:.+]] = linalg.elementwise <add>
// CHECK-SAME: gc.fusion_group<3, tile_sizes = []>
// CHECK-SAME: ins(%[[FIRST]], %[[SECOND]]
// CHECK-NEXT: return %[[CONSUMER]]
func.func @cycle_through_seeded_anchor_capture(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %out: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %first = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %second = linalg.generic {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %c0 = arith.constant 0 : index
    %captured = tensor.extract %first[%c0, %c0] : tensor<4x4xf32>
    %product = arith.mulf %x, %y : f32
    %sum = arith.addf %product, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<4x4xf32>
  %consumer = linalg.elementwise <add>
    ins(%first, %second : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %consumer : tensor<4x4xf32>
}

// -----

// Supported Linalgx groups participate in expansion. Tensor operations that
// produce scalars are supported, too, while the scalar constant stays outside.
#gl = affine_map<(e, m, n, k) -> (m, k)>
#gw = affine_map<(e, m, n, k) -> (e, n, k)>
#go = affine_map<(e, m, n, k) -> (e)>
#gr = affine_map<(e, m, n, k) -> (m, n)>
// CHECK-LABEL: func.func @grouped_matmul_neighbors(
// CHECK: arith.constant 0 : index
// CHECK-NEXT: tensor.dim {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: linalgx.grouped_matmul {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: tensor.cast {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: return
func.func @grouped_matmul_neighbors(
    %lhs: tensor<?x4xf32>, %weights: tensor<2x4x4xf32>,
    %offsets: tensor<2xi32>, %out: tensor<?x4xf32>) -> tensor<8x4xf32> {
  %c0 = arith.constant 0 : index
  %max = tensor.dim %lhs, %c0 : tensor<?x4xf32>
  %gmm = linalgx.grouped_matmul
    <{indexing_maps = [#gl, #gw, #go, #gr]}>
    ins(%lhs, %weights, %offsets :
      tensor<?x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>)
    max_group_size(%max)
    outs(%out : tensor<?x4xf32>) -> tensor<?x4xf32>
  %result = tensor.cast %gmm : tensor<?x4xf32> to tensor<8x4xf32>
  return %result : tensor<8x4xf32>
}

// -----

#q = affine_map<(m, k1, k2, n) -> (m, k1)>
#k = affine_map<(m, k1, k2, n) -> (k2, k1)>
#v = affine_map<(m, k1, k2, n) -> (k2, n)>
#s = affine_map<(m, k1, k2, n) -> ()>
#o = affine_map<(m, k1, k2, n) -> (m, n)>
// CHECK-LABEL: func.func @attention_neighbors(
// CHECK: tensor.cast {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: arith.constant
// CHECK-NEXT: linalgx.attention {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: linalg.elementwise <add> {{.*}}gc.fusion_group<0, tile_sizes = []>
// CHECK-NEXT: return
func.func @attention_neighbors(
    %query: tensor<4x4xf32>, %key: tensor<4x4xf32>, %value: tensor<4x4xf32>,
    %out: tensor<4x4xf32>, %bias: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %input = tensor.cast %query : tensor<4x4xf32> to tensor<4x4xf32>
  %scale = arith.constant 0.5 : f32
  %attention = linalgx.attention
    <{indexing_maps = [#q, #k, #v, #s, #o]}>
    ins(%input, %key, %value, %scale :
      tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, f32)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.elementwise <add>
    ins(%attention, %bias : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
