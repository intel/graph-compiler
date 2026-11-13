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

// -----

// CHECK-LABEL: func.func @middle_unit_transfer(
// CHECK: %[[READ:.*]] = vector.transfer_read {{.*}}vector<1x32x1x64xf16>
// CHECK-NEXT: %[[RESULT:.*]] = vector.shape_cast %[[READ]] : vector<1x32x1x64xf16> to vector<32x64xf16>
// CHECK-NEXT: return %[[RESULT]] : vector<32x64xf16>
func.func @middle_unit_transfer(%source: memref<1x1024x24x64xf16>, %tile: index, %batch: index) -> vector<32x64xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %read = vector.transfer_read %source[%c0, %tile, %batch, %c0], %padding {in_bounds = [true, true, true, true]} : memref<1x1024x24x64xf16>, vector<1x32x1x64xf16>
  %transpose = vector.transpose %read, [0, 2, 1, 3] : vector<1x32x1x64xf16> to vector<1x1x32x64xf16>
  %result = vector.shape_cast %transpose : vector<1x1x32x64xf16> to vector<32x64xf16>
  return %result : vector<32x64xf16>
}

// -----

// CHECK: #[[SWAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @preserve_transposed_transfer(
// CHECK: %[[READ:.*]] = vector.transfer_read {{.*}}permutation_map = #[[SWAP]]{{.*}}vector<16x8xf16>
// CHECK-NEXT: return %[[READ]] : vector<16x8xf16>
func.func @preserve_transposed_transfer(%source: memref<8x16xf16>) -> vector<16x8xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %read = vector.transfer_read %source[%c0, %c0], %padding {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<8x16xf16>, vector<16x8xf16>
  return %read : vector<16x8xf16>
}

// -----

// CHECK: #[[METADATA:.*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-LABEL: func.func @strided_metadata_read(
// CHECK: %[[READ:.*]] = vector.transfer_read %arg0[%arg1, %arg2, %arg3]{{.*}}permutation_map = #[[METADATA]]{{.*}}vector<64xi4>
// CHECK-NEXT: return %[[READ]] : vector<64xi4>
func.func @strided_metadata_read(%source: memref<256x512x32xi4, strided<[16384, 1, 512]>>, %expert: index, %channel: index, %group: index) -> vector<64xi4> {
  %padding = arith.constant 0 : i4
  %read = vector.transfer_read %source[%expert, %channel, %group], %padding {in_bounds = [true, true, true]} : memref<256x512x32xi4, strided<[16384, 1, 512]>>, vector<1x64x1xi4>
  %result = vector.shape_cast %read : vector<1x64x1xi4> to vector<64xi4>
  return %result : vector<64xi4>
}

// -----

// CHECK-LABEL: func.func @masked_read(
// CHECK: %[[READ:.*]] = vector.transfer_read {{.*}}, %arg4 {{.*}}vector<1x1x64xi4>
// CHECK-NEXT: return %[[READ]] : vector<1x1x64xi4>
func.func @masked_read(%source: memref<256x32x512xi4>, %expert: index, %group: index, %channel: index, %mask: vector<1x1x64xi1>) -> vector<1x1x64xi4> {
  %padding = arith.constant 0 : i4
  %read = vector.transfer_read %source[%expert, %group, %channel], %padding, %mask {in_bounds = [true, true, true]} : memref<256x32x512xi4>, vector<1x1x64xi4>
  return %read : vector<1x1x64xi4>
}

// -----

// CHECK-LABEL: func.func @contiguous_projected_read(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[VIEW:.*]] = memref.subview %arg0[%arg1, 0, 0, %[[ZERO]]] [1, 24, 128, 1] [1, 1, 1, 1]
// CHECK-SAME: to memref<24x128xf16, strided<[128, 1], offset: ?>>
// CHECK: %[[READ:.*]] = vector.transfer_read %[[VIEW]][%arg2, %arg3]{{.*}}vector<8x64xf16>
// CHECK-NEXT: return %[[READ]] : vector<8x64xf16>
func.func @contiguous_projected_read(%source: memref<4x24x128x1xf16>, %batch: index, %row: index, %column: index) -> vector<8x64xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %read = vector.transfer_read %source[%batch, %row, %column, %c0], %padding {in_bounds = [true, true, true, true]} : memref<4x24x128x1xf16>, vector<1x8x64x1xf16>
  %result = vector.shape_cast %read : vector<1x8x64x1xf16> to vector<8x64xf16>
  return %result : vector<8x64xf16>
}

// -----

// CHECK-LABEL: func.func @computed_store(
// CHECK: %[[VALUE:.*]] = vector.shape_cast %arg0 : vector<1x1x8x64xf16> to vector<8x64xf16>
// CHECK: vector.transfer_write %[[VALUE]]{{.*}}vector<8x64xf16>, memref<4x24x128x64xf16>
func.func @computed_store(%value: vector<1x1x8x64xf16>, %target: memref<4x24x128x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %target[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x64xf16>, memref<4x24x128x64xf16>
  return
}

// -----

// CHECK-LABEL: func.func @copy_store(
// CHECK: vector.transfer_read {{.*}}vector<1x8x1x64xf16>
// CHECK: vector.transfer_write {{.*}}vector<1x1x8x64xf16>, memref<4x24x128x64xf16>
func.func @copy_store(%source: memref<4x128x24x64xf16>, %target: memref<4x24x128x64xf16>) {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %read = vector.transfer_read %source[%c0, %c0, %c0, %c0], %padding {in_bounds = [true, true, true, true]} : memref<4x128x24x64xf16>, vector<1x8x1x64xf16>
  %transpose = vector.transpose %read, [0, 2, 1, 3] : vector<1x8x1x64xf16> to vector<1x1x8x64xf16>
  vector.transfer_write %transpose, %target[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x64xf16>, memref<4x24x128x64xf16>
  return
}

// -----

// CHECK-LABEL: func.func @out_of_bounds_read(
// CHECK: %[[READ:.*]] = vector.transfer_read {{.*}}in_bounds = [false, true, true]{{.*}}vector<1x1x64xi4>
// CHECK-NEXT: return %[[READ]] : vector<1x1x64xi4>
func.func @out_of_bounds_read(%source: memref<256x32x512xi4>, %expert: index, %group: index, %channel: index) -> vector<1x1x64xi4> {
  %padding = arith.constant 0 : i4
  %read = vector.transfer_read %source[%expert, %group, %channel], %padding {in_bounds = [false, true, true]} : memref<256x32x512xi4>, vector<1x1x64xi4>
  return %read : vector<1x1x64xi4>
}