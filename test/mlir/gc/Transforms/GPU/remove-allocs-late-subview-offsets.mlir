// RUN: gc-opt %s --remove-allocs --split-input-file | FileCheck %s

// The copy target is a subview whose offsets are computed after the alloc.
// Removing the alloc must move the whole index computation, not just the
// subview op, otherwise the result violates SSA dominance.

// CHECK-LABEL: func.func @alloc_copy_to_subview_with_late_offsets
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OFF:.+]] = affine.apply
// CHECK:         %[[VIEW:.+]] = memref.subview %arg0[0, %[[OFF]]]
// CHECK:         vector.transfer_write %arg2, %[[VIEW]]
// CHECK-NOT:     memref.copy
func.func @alloc_copy_to_subview_with_late_offsets(%arg0: memref<8x64xf16>, %arg1: index, %arg2: vector<1x64xf16>) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() alignment = 64 : memref<1x64xf16>
  vector.transfer_write %arg2, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16>
  %off = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg1]
  %subview = memref.subview %arg0[0, %off] [1, 64] [1, 1] : memref<8x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  memref.copy %alloc, %subview : memref<1x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  return
}

// -----

// The offset computation is not movable (it reads memory), so the alloc must be
// kept instead of producing invalid IR.

// CHECK-LABEL: func.func @alloc_copy_to_subview_with_unmovable_offsets
// CHECK:         memref.alloc
// CHECK:         memref.copy
func.func @alloc_copy_to_subview_with_unmovable_offsets(%arg0: memref<8x64xf16>, %arg1: memref<index>, %arg2: vector<1x64xf16>) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() alignment = 64 : memref<1x64xf16>
  vector.transfer_write %arg2, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16>
  %off = memref.load %arg1[] : memref<index>
  %subview = memref.subview %arg0[0, %off] [1, 64] [1, 1] : memref<8x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  memref.copy %alloc, %subview : memref<1x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
  return
}

// -----

// An unsupported view chain must fail before moving the late offset
// computation. Returning failure after moving it would violate the rewrite
// pattern contract.

// CHECK-LABEL: func.func @alloc_copy_through_unsupported_reshape
// CHECK:         %[[ALLOC:.+]] = memref.alloc
// CHECK:         vector.transfer_write %arg2, %[[ALLOC]]
// CHECK:         %[[RESHAPE:.+]] = memref.reshape %[[ALLOC]]
// CHECK:         %[[OFF:.+]] = affine.apply
// CHECK:         %[[VIEW:.+]] = memref.subview %arg0[%[[OFF]], 0]
// CHECK:         memref.copy %[[RESHAPE]], %[[VIEW]]
func.func @alloc_copy_through_unsupported_reshape(%arg0: memref<64x64xf16>, %arg1: index, %arg2: vector<1x64xf16>, %shape: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() alignment = 64 : memref<1x64xf16>
  vector.transfer_write %arg2, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16>
  %reshape = memref.reshape %alloc(%shape) : (memref<1x64xf16>, memref<2xindex>) -> memref<64x1xf16>
  %off = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%arg1]
  %subview = memref.subview %arg0[%off, 0] [64, 1] [1, 1] : memref<64x64xf16> to memref<64x1xf16, strided<[64, 1], offset: ?>>
  memref.copy %reshape, %subview : memref<64x1xf16> to memref<64x1xf16, strided<[64, 1], offset: ?>>
  return
}
