// RUN: gc-opt %s --drop-unit-dims --split-input-file | FileCheck %s
// RUN: gc-opt %s --drop-unit-dims --split-input-file | FileCheck %s --check-prefix=RANKS

// RANKS-LABEL: func.func @dequantize(
// RANKS-NOT: vector<1x
// RANKS-NOT: vector<8x1x
// RANKS: return
// CHECK-LABEL: func.func @dequantize(
// CHECK: vector.transfer_read {{.*}}vector<8xf16>
// CHECK: vector.transfer_read {{.*}}vector<8xi4>
// CHECK: vector.transfer_read {{.*}}vector<8x8xi4>
// CHECK: arith.extui {{.*}}vector<8x8xi4> to vector<8x8xi32>
// CHECK: arith.extui {{.*}}vector<8x8xi4> to vector<8x8xi32>
// CHECK: arith.subi {{.*}}vector<8x8xi32>
// CHECK: arith.sitofp {{.*}}vector<8x8xi32> to vector<8x8xf16>
// CHECK: arith.mulf {{.*}}vector<8x8xf16>
// CHECK: return {{.*}}vector<8x8xf16>
func.func @dequantize(%scales: memref<1x1x8xf16>, %zeros: memref<1x1x8xi4>, %weights: memref<1x8x8xi4>) -> vector<8x8xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %zero = arith.constant 0 : i4
  %scale = vector.transfer_read %scales[%c0, %c0, %c0], %padding {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : memref<1x1x8xf16>, vector<1x8x1xf16>
  %scale_broadcast = vector.broadcast %scale : vector<1x8x1xf16> to vector<8x1x8x1xf16>
  %scale_transpose = vector.transpose %scale_broadcast, [1, 2, 3, 0] : vector<8x1x8x1xf16> to vector<1x8x1x8xf16>
  %offset = vector.transfer_read %zeros[%c0, %c0, %c0], %zero {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : memref<1x1x8xi4>, vector<1x8x1xi4>
  %offset_broadcast = vector.broadcast %offset : vector<1x8x1xi4> to vector<8x1x8x1xi4>
  %offset_transpose = vector.transpose %offset_broadcast, [1, 2, 3, 0] : vector<8x1x8x1xi4> to vector<1x8x1x8xi4>
  %weight = vector.transfer_read %weights[%c0, %c0, %c0], %zero {in_bounds = [true, true, true]} : memref<1x8x8xi4>, vector<1x8x8xi4>
  %scale_tile = vector.shape_cast %scale_transpose : vector<1x8x1x8xf16> to vector<1x8x8xf16>
  %offset_tile = vector.shape_cast %offset_transpose : vector<1x8x1x8xi4> to vector<1x8x8xi4>
  %weight_ext = arith.extui %weight : vector<1x8x8xi4> to vector<1x8x8xi32>
  %offset_ext = arith.extui %offset_tile : vector<1x8x8xi4> to vector<1x8x8xi32>
  %subtract = arith.subi %weight_ext, %offset_ext : vector<1x8x8xi32>
  %convert = arith.sitofp %subtract : vector<1x8x8xi32> to vector<1x8x8xf16>
  %result = arith.mulf %convert, %scale_tile : vector<1x8x8xf16>
  %tile = vector.shape_cast %result : vector<1x8x8xf16> to vector<8x8xf16>
  return %tile : vector<8x8xf16>
}

// -----

// CHECK-LABEL: func.func @expanded_source_axis(
// CHECK: %[[BCAST:.*]] = vector.broadcast %arg0 : vector<1x1x4xf32> to vector<3x8x4xf32>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[BCAST]] : vector<3x8x4xf32> to vector<3x1x8x4xf32>
// CHECK-NEXT: return %[[RESULT]]
func.func @expanded_source_axis(%source: vector<1x1x4xf32>) -> vector<3x1x8x4xf32> {
  %result = vector.broadcast %source : vector<1x1x4xf32> to vector<3x1x8x4xf32>
  return %result : vector<3x1x8x4xf32>
}

// -----

// CHECK-LABEL: func.func @scalable_broadcast(
// CHECK: %[[BCAST:.*]] = vector.broadcast %arg0 : vector<1x[1]xf32> to vector<4x[1]xf32>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[BCAST]] : vector<4x[1]xf32> to vector<4x1x[1]xf32>
// CHECK-NEXT: return %[[RESULT]]
func.func @scalable_broadcast(%source: vector<1x[1]xf32>) -> vector<4x1x[1]xf32> {
  %result = vector.broadcast %source : vector<1x[1]xf32> to vector<4x1x[1]xf32>
  return %result : vector<4x1x[1]xf32>
}