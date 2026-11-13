// RUN: gc-opt %s --normalize-memref-layouts --split-input-file | FileCheck %s

// A column-major matrix argument is read through a row-major view that is
// created once, next to the argument. The read of the [n, k] tile becomes a
// transposing read of the [k, n] tile.

// CHECK-LABEL: @column_major_arg
// CHECK-SAME:    %[[M:.*]]: memref<512x2048xf16, strided<[1, 512]>>
// CHECK-SAME:    %[[N:.*]]: index, %[[K:.*]]: index
// CHECK:       %[[V:.*]] = memref.transpose %[[M]] (d0, d1) -> (d1, d0)
// CHECK-SAME:    : memref<512x2048xf16, strided<[1, 512]>> to memref<2048x512xf16, strided<[512, 1]>>
// CHECK-NOT:   memref.transpose
// CHECK:       vector.transfer_read %[[V]][%[[K]], %[[N]]], %{{.*}} {in_bounds = [true, true], permutation_map = #[[$SWAP:.*]]} : memref<2048x512xf16, strided<[512, 1]>>, vector<64x64xf16>
// CHECK:       vector.transfer_read %[[V]][%[[N]], %[[K]]], %{{.*}} {in_bounds = [true, true], permutation_map = #[[$SWAP]]} : memref<2048x512xf16, strided<[512, 1]>>, vector<64x64xf16>
// CHECK:       return
func.func @column_major_arg(%m: memref<512x2048xf16, strided<[1, 512]>>, %n: index, %k: index) -> (vector<64x64xf16>, vector<64x64xf16>) {
  %pad = arith.constant 0.0 : f16
  %a = vector.transfer_read %m[%n, %k], %pad {in_bounds = [true, true]} : memref<512x2048xf16, strided<[1, 512]>>, vector<64x64xf16>
  %b = vector.transfer_read %m[%k, %n], %pad {in_bounds = [true, true]} : memref<512x2048xf16, strided<[1, 512]>>, vector<64x64xf16>
  return %a, %b : vector<64x64xf16>, vector<64x64xf16>
}

// -----

// A read that already transposes becomes a plain read and a write is
// redirected the same way. A dynamic stride is fine: the unit-stride dim just
// moves innermost. in_bounds is indexed by vector dim and is kept; the mask is
// shaped in memref-dim order and is transposed with the dims.

// CHECK-DAG: #[[$SWAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @transposed_read_and_write
// CHECK-SAME:    %[[M:.*]]: memref<?x?xf32, strided<[1, ?], offset: ?>>
// CHECK-SAME:    %[[MASK:[a-z0-9]*]]: vector<8x16xi1>
// CHECK:       %[[T:.*]] = memref.transpose %[[M]] (d0, d1) -> (d1, d0)
// CHECK-SAME:    to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:       %[[TMASK:.*]] = vector.transpose %[[MASK]], [1, 0] : vector<8x16xi1> to vector<16x8xi1>
// CHECK:       vector.transfer_read %[[T]][%[[K:.*]], %[[N:.*]]], %{{.*}}, %[[TMASK]] {in_bounds = [true, false]} : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<16x8xf32>
// CHECK:       vector.transfer_write %{{.*}}, %[[T]][%[[K]], %[[N]]] {in_bounds = [false, true], permutation_map = #[[$SWAP]]} : vector<8x16xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
#swap = affine_map<(d0, d1) -> (d1, d0)>
func.func @transposed_read_and_write(%m: memref<?x?xf32, strided<[1, ?], offset: ?>>, %n: index, %k: index, %v: vector<8x16xf32>, %mask: vector<8x16xi1>) -> vector<16x8xf32> {
  %pad = arith.constant 0.0 : f32
  %a = vector.transfer_read %m[%n, %k], %pad, %mask {in_bounds = [true, false], permutation_map = #swap} : memref<?x?xf32, strided<[1, ?], offset: ?>>, vector<16x8xf32>
  vector.transfer_write %v, %m[%n, %k] {in_bounds = [false, true]} : vector<8x16xf32>, memref<?x?xf32, strided<[1, ?], offset: ?>>
  return %a : vector<16x8xf32>
}

// -----

// Wrapper masks must be transposed outside the single-operation mask region.

// CHECK-DAG: #[[$SWAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @wrapper_mask_read_and_write
// CHECK-SAME: %[[M:.*]]: memref<16x32xf32, strided<[1, 16]>>
// CHECK-SAME: %[[MASK:[a-z0-9]*]]: vector<8x16xi1>, %[[V:[a-z0-9]*]]: vector<8x16xf32>
// CHECK: %[[VIEW:.*]] = memref.transpose %[[M]]
// CHECK: %[[RMASK:.*]] = vector.transpose %[[MASK]], [1, 0] : vector<8x16xi1> to vector<16x8xi1>
// CHECK-NEXT: %[[R:.*]] = vector.mask %[[RMASK]] { vector.transfer_read %[[VIEW]][%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false], permutation_map = #[[$SWAP]]} : memref<32x16xf32, strided<[16, 1]>>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
// CHECK-NEXT: %[[WMASK:.*]] = vector.transpose %[[MASK]], [1, 0] : vector<8x16xi1> to vector<16x8xi1>
// CHECK-NEXT: vector.mask %[[WMASK]] { vector.transfer_write %[[V]], %[[VIEW]][%{{.*}}, %{{.*}}] {in_bounds = [false, true], permutation_map = #[[$SWAP]]} : vector<8x16xf32>, memref<32x16xf32, strided<[16, 1]>> } : vector<16x8xi1>
// CHECK-NEXT: return %[[R]]
func.func @wrapper_mask_read_and_write(%m: memref<16x32xf32, strided<[1, 16]>>, %mask: vector<8x16xi1>, %v: vector<8x16xf32>) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.mask %mask { vector.transfer_read %m[%c0, %c0], %pad {in_bounds = [true, false]} : memref<16x32xf32, strided<[1, 16]>>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
  vector.mask %mask { vector.transfer_write %v, %m[%c0, %c0] {in_bounds = [false, true]} : vector<8x16xf32>, memref<16x32xf32, strided<[1, 16]>> } : vector<8x16xi1>
  return %r : vector<8x16xf32>
}

// -----

// Square masks still need a transpose even though their type does not change.
// Both wrapper masks and explicit transfer masks must be updated. vector.mask
// rejects transfers that have a mask operand, so each form is a separate op.

// CHECK-LABEL: @square_wrapper_and_transfer_masks
// CHECK-SAME: %[[M:.*]]: memref<16x32xf32, strided<[1, 16]>>
// CHECK-SAME: %[[OUTER:[a-z0-9]*]]: vector<8x8xi1>, %[[INNER:[a-z0-9]*]]: vector<8x8xi1>
// CHECK: %[[VIEW:.*]] = memref.transpose %[[M]]
// CHECK: %[[OUTER_T:.*]] = vector.transpose %[[OUTER]], [1, 0] : vector<8x8xi1> to vector<8x8xi1>
// CHECK-NEXT: %[[R0:.*]] = vector.mask %[[OUTER_T]] { vector.transfer_read %[[VIEW]][%{{.*}}, %{{.*}}], %{{.*}} {{.*}} } : vector<8x8xi1> -> vector<8x8xf32>
// CHECK-NEXT: %[[INNER_T:.*]] = vector.transpose %[[INNER]], [1, 0] : vector<8x8xi1> to vector<8x8xi1>
// CHECK-NEXT: %[[R1:.*]] = vector.transfer_read %[[VIEW]][%{{.*}}, %{{.*}}], %{{.*}}, %[[INNER_T]] {{.*}} : memref<32x16xf32, strided<[16, 1]>>, vector<8x8xf32>
// CHECK-NEXT: return %[[R0]], %[[R1]]
func.func @square_wrapper_and_transfer_masks(%m: memref<16x32xf32, strided<[1, 16]>>, %outer: vector<8x8xi1>, %inner: vector<8x8xi1>) -> (vector<8x8xf32>, vector<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r0 = vector.mask %outer { vector.transfer_read %m[%c0, %c0], %pad {in_bounds = [true, true]} : memref<16x32xf32, strided<[1, 16]>>, vector<8x8xf32> } : vector<8x8xi1> -> vector<8x8xf32>
  %r1 = vector.transfer_read %m[%c0, %c0], %pad, %inner {in_bounds = [true, true]} : memref<16x32xf32, strided<[1, 16]>>, vector<8x8xf32>
  return %r0, %r1 : vector<8x8xf32>, vector<8x8xf32>
}

// -----

// A projected transfer only reads one memref dimension, so its mask order
// remains unchanged when that dimension moves.

// CHECK-LABEL: @projected_wrapper_mask
// CHECK-SAME: %[[M:.*]]: memref<16x32xf32, strided<[1, 16]>>
// CHECK-SAME: %[[MASK:[a-z0-9]*]]: vector<8xi1>
// CHECK: %[[VIEW:.*]] = memref.transpose %[[M]]
// CHECK-NOT: vector.transpose
// CHECK: vector.mask %[[MASK]] { vector.transfer_read %[[VIEW]]
// CHECK-NOT: vector.transpose
// CHECK: return
#project = affine_map<(d0, d1) -> (d0)>
func.func @projected_wrapper_mask(%m: memref<16x32xf32, strided<[1, 16]>>, %mask: vector<8xi1>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.mask %mask { vector.transfer_read %m[%c0, %c0], %pad {in_bounds = [true], permutation_map = #project} : memref<16x32xf32, strided<[1, 16]>>, vector<8xf32> } : vector<8xi1> -> vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A batched column-major weight: the batch dim keeps the largest stride, the
// two matrix dims are swapped. Broadcast results of the map are untouched.

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$BCAST:.*]] = affine_map<(d0, d1, d2) -> (d2, 0)>
// CHECK-LABEL: @batched
// CHECK:       %[[V:.*]] = memref.transpose %{{.*}} (d0, d1, d2) -> (d0, d2, d1)
// CHECK-SAME:    : memref<4x512x2048xf16, strided<[1048576, 1, 512]>> to memref<4x2048x512xf16, strided<[1048576, 512, 1]>>
// CHECK:       vector.transfer_read %[[V]][%[[E:.*]], %[[K:.*]], %[[N:.*]]], %{{.*}} {in_bounds = [true, true], permutation_map = #[[$MAP]]} : memref<4x2048x512xf16, strided<[1048576, 512, 1]>>, vector<64x64xf16>
// CHECK:       vector.transfer_read %[[V]][%[[E]], %[[K]], %[[N]]], %{{.*}} {in_bounds = [true, true], permutation_map = #[[$BCAST]]} : memref<4x2048x512xf16, strided<[1048576, 512, 1]>>, vector<64x64xf16>
#nk = affine_map<(d0, d1, d2) -> (d1, d2)>
#bcast = affine_map<(d0, d1, d2) -> (d1, 0)>
func.func @batched(%w: memref<4x512x2048xf16, strided<[1048576, 1, 512]>>, %e: index, %n: index, %k: index) -> (vector<64x64xf16>, vector<64x64xf16>) {
  %pad = arith.constant 0.0 : f16
  %a = vector.transfer_read %w[%e, %n, %k], %pad {in_bounds = [true, true], permutation_map = #nk} : memref<4x512x2048xf16, strided<[1048576, 1, 512]>>, vector<64x64xf16>
  %b = vector.transfer_read %w[%e, %n, %k], %pad {in_bounds = [true, true], permutation_map = #bcast} : memref<4x512x2048xf16, strided<[1048576, 1, 512]>>, vector<64x64xf16>
  return %a, %b : vector<64x64xf16>, vector<64x64xf16>
}

// -----

// Row-major memrefs, with or without an explicit layout, and memrefs with no
// unit stride are left alone.

// CHECK-LABEL: @untouched
// CHECK-NOT:   memref.transpose
func.func @untouched(%a: memref<512x2048xf16>, %b: memref<?x64xf16, strided<[2048, 1], offset: ?>>, %c: memref<512x2048xf16, strided<[2, 1024]>>, %i: index) -> (vector<64x64xf16>, vector<32x64xf16>, vector<64x64xf16>) {
  %pad = arith.constant 0.0 : f16
  %0 = vector.transfer_read %a[%i, %i], %pad {in_bounds = [true, true]} : memref<512x2048xf16>, vector<64x64xf16>
  %1 = vector.transfer_read %b[%i, %i], %pad {in_bounds = [false, true]} : memref<?x64xf16, strided<[2048, 1], offset: ?>>, vector<32x64xf16>
  %2 = vector.transfer_read %c[%i, %i], %pad {in_bounds = [true, true]} : memref<512x2048xf16, strided<[2, 1024]>>, vector<64x64xf16>
  return %0, %1, %2 : vector<64x64xf16>, vector<32x64xf16>, vector<64x64xf16>
}
