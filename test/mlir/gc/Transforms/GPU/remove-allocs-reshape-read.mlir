// RUN: gc-opt %s --remove-allocs --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @rank_reduced_read(
// CHECK-NOT: memref.alloc
// CHECK-NOT: vector.transfer
// CHECK: %[[CAST:.*]] = vector.shape_cast %arg0 : vector<1x64x64xf16> to vector<64x64xf16>
// CHECK-NEXT: return %[[CAST]] : vector<64x64xf16>
func.func @rank_reduced_read(%tile: vector<1x64x64xf16>) -> vector<64x64xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x64x64xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x64xf16>, memref<1x64x64xf16>
  %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding {in_bounds = [true, true]} : memref<1x64x64xf16>, vector<64x64xf16>
  memref.dealloc %alloc : memref<1x64x64xf16>
  return %read : vector<64x64xf16>
}

// -----

// CHECK-LABEL: func.func @loop_rank_reduced_read(
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NOT: vector.transfer
// CHECK: vector.shape_cast %arg0 : vector<1x4x4xf16> to vector<4x4xf16>
// CHECK: arith.addf
// CHECK-NEXT: scf.yield
func.func @loop_rank_reduced_read(%tile: vector<1x4x4xf16>, %count: index, %init: vector<4x4xf16>) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x4x4xf16>
  %result = scf.for %step = %c0 to %count step %c1 iter_args(%acc = %init) -> vector<4x4xf16> {
    vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x4x4xf16>
    %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding {in_bounds = [true, true]} : memref<1x4x4xf16>, vector<4x4xf16>
    %sum = arith.addf %acc, %read : vector<4x4xf16>
    scf.yield %sum : vector<4x4xf16>
  }
  memref.dealloc %alloc : memref<1x4x4xf16>
  return %result : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func.func @collapsed_read(
// CHECK-NOT: memref.alloc
// CHECK-NOT: vector.transfer
// CHECK-NOT: memref.collapse_shape
// CHECK: %[[CAST:.*]] = vector.shape_cast %arg0 : vector<1x4x4xf16> to vector<4x4xf16>
// CHECK-NEXT: return %[[CAST]] : vector<4x4xf16>
func.func @collapsed_read(%tile: vector<1x4x4xf16>) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x4x4xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x4x4xf16>
  %view = memref.collapse_shape %alloc [[0, 1], [2]] : memref<1x4x4xf16> into memref<4x4xf16>
  %read = vector.transfer_read %view[%c0, %c0], %padding {in_bounds = [true, true]} : memref<4x4xf16>, vector<4x4xf16>
  memref.dealloc %alloc : memref<1x4x4xf16>
  return %read : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func.func @expanded_read(
// CHECK-NOT: memref.alloc
// CHECK-NOT: vector.transfer
// CHECK-NOT: memref.expand_shape
// CHECK: %[[CAST:.*]] = vector.shape_cast %arg0 : vector<4x4xf16> to vector<1x4x4xf16>
// CHECK-NEXT: return %[[CAST]] : vector<1x4x4xf16>
func.func @expanded_read(%tile: vector<4x4xf16>) -> vector<1x4x4xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<4x4xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf16>, memref<4x4xf16>
  %view = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 4, 4] : memref<4x4xf16> into memref<1x4x4xf16>
  %read = vector.transfer_read %view[%c0, %c0, %c0], %padding {in_bounds = [true, true, true]} : memref<1x4x4xf16>, vector<1x4x4xf16>
  memref.dealloc %alloc : memref<4x4xf16>
  return %read : vector<1x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @masked_read(
// CHECK: memref.alloc
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read {{.*}}, %arg1
func.func @masked_read(%tile: vector<1x4x4xf16>, %mask: vector<4x4xi1>) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x4x4xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x4x4xf16>
  %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding, %mask {in_bounds = [true, true]} : memref<1x4x4xf16>, vector<4x4xf16>
  memref.dealloc %alloc : memref<1x4x4xf16>
  return %read : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func.func @permuted_read(
// CHECK: memref.alloc
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
func.func @permuted_read(%tile: vector<1x4x4xf16>) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x4x4xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x4x4xf16>
  %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<1x4x4xf16>, vector<4x4xf16>
  memref.dealloc %alloc : memref<1x4x4xf16>
  return %read : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func.func @partial_write(
// CHECK: memref.alloc
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
func.func @partial_write(%tile: vector<1x4x4xf16>) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x8x8xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x8x8xf16>
  %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding {in_bounds = [true, true]} : memref<1x8x8xf16>, vector<4x4xf16>
  memref.dealloc %alloc : memref<1x8x8xf16>
  return %read : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func.func @conditional_write(
// CHECK: memref.alloc
// CHECK: scf.if
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
func.func @conditional_write(%tile: vector<1x4x4xf16>, %condition: i1) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x4x4xf16>
  scf.if %condition {
    vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x4x4xf16>
  }
  %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding {in_bounds = [true, true]} : memref<1x4x4xf16>, vector<4x4xf16>
  memref.dealloc %alloc : memref<1x4x4xf16>
  return %read : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func.func @multiple_reads(
// CHECK: memref.alloc
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
func.func @multiple_reads(%tile: vector<1x4x4xf16>, %mask: vector<4x4xi1>) -> (vector<4x4xf16>, vector<4x4xf16>) {
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f16
  %alloc = memref.alloc() : memref<1x4x4xf16>
  vector.transfer_write %tile, %alloc[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x4xf16>, memref<1x4x4xf16>
  %read = vector.transfer_read %alloc[%c0, %c0, %c0], %padding {in_bounds = [true, true]} : memref<1x4x4xf16>, vector<4x4xf16>
  %masked = vector.transfer_read %alloc[%c0, %c0, %c0], %padding, %mask {in_bounds = [true, true]} : memref<1x4x4xf16>, vector<4x4xf16>
  memref.dealloc %alloc : memref<1x4x4xf16>
  return %read, %masked : vector<4x4xf16>, vector<4x4xf16>
}