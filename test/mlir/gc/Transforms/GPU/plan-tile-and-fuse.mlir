// RUN: gc-opt %s --plan-tile-and-fuse --split-input-file | FileCheck %s

// Assign the anchor first even though its producer appears earlier in the IR.
// Both producer and consumer remain in separate groups.
// CHECK-LABEL: func.func @matmul_with_neighbors(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<4x4xf32>,
// CHECK-SAME:    %[[BIAS:[^:]+]]: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty()
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      : tensor<4x4xf32>
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[MM:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = linalg.elementwise <add>
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<3, tile_sizes = []>}
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

// Reshapes wrapping a matmul remain separate groups. The anchor is assigned
// first, followed by collapse_shape, empty, fill, and expand_shape in IR order.
// CHECK-LABEL: func.func @reshapes_around_matmul(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<2x4x8xf32>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<8x16xf32>) -> tensor<2x4x16xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[LHS:.+]] = tensor.collapse_shape %[[A]] {{\[}}[0, 1], [2]]
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
// CHECK-SAME:      : tensor<2x4x8xf32> into tensor<8x8xf32>
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty()
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<2, tile_sizes = []>}
// CHECK-SAME:      : tensor<8x16xf32>
// CHECK-NEXT:    %[[INIT:.+]] = linalg.fill
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<3, tile_sizes = []>}
// CHECK-SAME:      ins(%[[ZERO]] : f32)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT:    %[[MM:.+]] = linalg.matmul
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<0, tile_sizes = []>}
// CHECK-SAME:      ins(%[[LHS]], %[[B]] : tensor<8x8xf32>, tensor<8x16xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT:    %[[RESULT:.+]] = tensor.expand_shape %[[MM]] {{\[}}[0, 1], [2]]
// CHECK-SAME:      output_shape [2, 4, 16]
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<4, tile_sizes = []>}
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
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
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
// CHECK-SAME:      {gc.fusion_group = #gc.fusion_group<1, tile_sizes = []>}
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
