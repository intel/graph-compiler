// RUN: gc-opt %s --simplify-indexing-maps --split-input-file | FileCheck %s

// A single indexed value is broadcast across d1 in [0, 64), allowing the
// floordiv map to become an identity map.
// CHECK-DAG: #[[ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @one_group_per_tile
// CHECK-SAME: %[[W:[^:]+]]: tensor<32x64xi4>, %[[S:[^:]+]]: tensor<32x1xf16>, %[[Z:[^:]+]]: tensor<32x1xi4>
// CHECK-DAG: %[[SB:.+]] = linalg.broadcast ins(%[[S]] : tensor<32x1xf16>) outs(%{{.*}} : tensor<32x1x64xf16>) dimensions = [2]
// CHECK-DAG: %[[SC:.+]] = tensor.collapse_shape %[[SB]] {{\[}}[0], [1, 2]] : tensor<32x1x64xf16> into tensor<32x64xf16>
// CHECK-DAG: %[[ZB:.+]] = linalg.broadcast ins(%[[Z]] : tensor<32x1xi4>) outs(%{{.*}} : tensor<32x1x64xi4>) dimensions = [2]
// CHECK-DAG: %[[ZC:.+]] = tensor.collapse_shape %[[ZB]] {{\[}}[0], [1, 2]] : tensor<32x1x64xi4> into tensor<32x64xi4>
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]], #[[ID]]]
// CHECK-SAME: ins(%[[W]], %[[SC]], %[[ZC]] :
func.func @one_group_per_tile(%w: tensor<32x64xi4>, %s: tensor<32x1xf16>,
                              %z: tensor<32x1xi4>, %o: tensor<32x64xf16>) -> tensor<32x64xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 64)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 64)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%w, %s, %z : tensor<32x64xi4>, tensor<32x1xf16>, tensor<32x1xi4>) outs(%o : tensor<32x64xf16>) {
  ^bb0(%in: i4, %in_s: f16, %in_z: i4, %out: f16):
    %0 = arith.extui %in : i4 to i32
    %1 = arith.extui %in_z : i4 to i32
    %2 = arith.subi %0, %1 : i32
    %3 = arith.sitofp %2 : i32 to f16
    %4 = arith.mulf %3, %in_s : f16
    linalg.yield %4 : f16
  } -> tensor<32x64xf16>
  return %r : tensor<32x64xf16>
}

// -----

// Trip count 32 <= divisor 64 but the operand dim is 4, not 1 (one value per
// group): the operand does not match the tile, the map must stay untouched.
// CHECK-DAG: #[[ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[GRP:.+]] = affine_map<(d0, d1) -> (d0, d1 floordiv 64)>
// CHECK-LABEL: func.func @small_trip_count
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[GRP]], #[[ID]]]
func.func @small_trip_count(%w: tensor<32x32xf16>, %s: tensor<32x4xf16>,
                            %o: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 64)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%w, %s : tensor<32x32xf16>, tensor<32x4xf16>) outs(%o : tensor<32x32xf16>) {
  ^bb0(%in: f16, %in_s: f16, %out: f16):
    %0 = arith.mulf %in, %in_s : f16
    linalg.yield %0 : f16
  } -> tensor<32x32xf16>
  return %r : tensor<32x32xf16>
}

// -----

// For d1 in [0, 64), d1 floordiv 32 selects two values. Broadcasting the
// 32x2 operand to 32x2x32 and collapsing to 32x64 permits identity indexing.
// CHECK-DAG: #[[ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @two_groups_per_tile
// CHECK-SAME: %[[W:[^:]+]]: tensor<32x64xf16>, %[[S:[^:]+]]: tensor<32x2xf16>
// CHECK: %[[E:.+]] = tensor.empty() : tensor<32x2x32xf16>
// CHECK: %[[B:.+]] = linalg.broadcast ins(%[[S]] : tensor<32x2xf16>) outs(%[[E]] : tensor<32x2x32xf16>) dimensions = [2]
// CHECK: %[[C:.+]] = tensor.collapse_shape %[[B]] {{\[}}[0], [1, 2]] : tensor<32x2x32xf16> into tensor<32x64xf16>
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]]
// CHECK-SAME: ins(%[[W]], %[[C]] : tensor<32x64xf16>, tensor<32x64xf16>)
func.func @two_groups_per_tile(%w: tensor<32x64xf16>, %s: tensor<32x2xf16>,
                               %o: tensor<32x64xf16>) -> tensor<32x64xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 32)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%w, %s : tensor<32x64xf16>, tensor<32x2xf16>) outs(%o : tensor<32x64xf16>) {
  ^bb0(%in: f16, %in_s: f16, %out: f16):
    %0 = arith.mulf %in, %in_s : f16
    linalg.yield %0 : f16
  } -> tensor<32x64xf16>
  return %r : tensor<32x64xf16>
}

// -----

// Dynamic trip count: nothing is known, the map must stay untouched.
// CHECK-DAG: #[[ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[GRP:.+]] = affine_map<(d0, d1) -> (d0, d1 floordiv 64)>
// CHECK-LABEL: func.func @dynamic_trip_count
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[GRP]], #[[ID]]]
func.func @dynamic_trip_count(%w: tensor<32x?xf16>, %s: tensor<32x?xf16>,
                              %o: tensor<32x?xf16>) -> tensor<32x?xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 64)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%w, %s : tensor<32x?xf16>, tensor<32x?xf16>) outs(%o : tensor<32x?xf16>) {
  ^bb0(%in: f16, %in_s: f16, %out: f16):
    %0 = arith.mulf %in, %in_s : f16
    linalg.yield %0 : f16
  } -> tensor<32x?xf16>
  return %r : tensor<32x?xf16>
}

// -----

// Group dim in the middle of a rank-3 operand, four groups per tile: the new
// dim is inserted right after the group dim and collapsed back into it.
// CHECK-DAG: #[[ID:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @middle_group_dim
// CHECK: tensor.empty() : tensor<2x4x16x8xf16>
// CHECK: linalg.broadcast ins(%{{.*}} : tensor<2x4x8xf16>) outs(%{{.*}} : tensor<2x4x16x8xf16>) dimensions = [2]
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2], [3]] : tensor<2x4x16x8xf16> into tensor<2x64x8xf16>
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]]
func.func @middle_group_dim(%w: tensor<2x64x8xf16>, %s: tensor<2x4x8xf16>,
                            %o: tensor<2x64x8xf16>) -> tensor<2x64x8xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%w, %s : tensor<2x64x8xf16>, tensor<2x4x8xf16>) outs(%o : tensor<2x64x8xf16>) {
  ^bb0(%in: f16, %in_s: f16, %out: f16):
    %0 = arith.mulf %in, %in_s : f16
    linalg.yield %0 : f16
  } -> tensor<2x64x8xf16>
  return %r : tensor<2x64x8xf16>
}

// -----

// Trip count 48 is not divisible by 32, so uniform broadcast expansion
// cannot represent the final partial group. The map must stay untouched.
// CHECK-DAG: #[[ID:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[GRP:.+]] = affine_map<(d0, d1) -> (d0, d1 floordiv 32)>
// CHECK-LABEL: func.func @partial_group
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[GRP]], #[[ID]]]
func.func @partial_group(%w: tensor<32x48xf16>, %s: tensor<32x2xf16>,
                         %o: tensor<32x48xf16>) -> tensor<32x48xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 32)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%w, %s : tensor<32x48xf16>, tensor<32x2xf16>) outs(%o : tensor<32x48xf16>) {
  ^bb0(%in: f16, %in_s: f16, %out: f16):
    %0 = arith.mulf %in, %in_s : f16
    linalg.yield %0 : f16
  } -> tensor<32x48xf16>
  return %r : tensor<32x48xf16>
}
