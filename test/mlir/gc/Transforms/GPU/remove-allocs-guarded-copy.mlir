// RUN: gc-opt %s --remove-allocs --split-input-file | FileCheck %s

// The else branch yields the destination, making its copy a no-op. Move the
// copy into the then branch and fold the alloc into a masked destination write.

// CHECK-LABEL: func.func @guarded_alloc_copy
// CHECK-NOT:     memref.alloc
// CHECK:         %[[DST:.+]] = memref.subview %arg0[%arg1, 0] [%arg2, 64]
// CHECK:         scf.if %arg3 {
// CHECK-NEXT:      vector.transfer_write %arg4, %[[DST]]{{.*}} {in_bounds = [false, true]}
// CHECK-NEXT:    }
// CHECK-NOT:     memref.copy
func.func @guarded_alloc_copy(%arg0: memref<?x64xf16>, %arg1: index, %arg2: index, %arg3: i1, %arg4: vector<32x64xf16>) {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[%arg1, 0] [%arg2, 64] [1, 1] : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  %alloc = memref.alloc() alignment = 64 : memref<32x64xf16>
  %0 = scf.if %arg3 -> (memref<?x64xf16, strided<[?, ?], offset: ?>>) {
    vector.transfer_write %arg4, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16>
    %subview_0 = memref.subview %alloc[0, 0] [%arg2, 64] [1, 1] : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    %cast = memref.cast %subview_0 : memref<?x64xf16, strided<[64, 1]>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<?x64xf16, strided<[?, ?], offset: ?>>
  } else {
    %cast = memref.cast %subview : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<?x64xf16, strided<[?, ?], offset: ?>>
  }
  memref.copy %0, %subview : memref<?x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  return
}

// -----

// The else branch yields a different buffer, so its copy is not a no-op and
// cannot be removed by moving the copy into the then branch.

// CHECK-LABEL: func.func @guarded_copy_from_other_buffer
// CHECK:         memref.alloc
// CHECK:         %[[RES:.+]] = scf.if
// CHECK:         memref.copy %[[RES]]
func.func @guarded_copy_from_other_buffer(%arg0: memref<?x64xf16>, %arg1: index, %arg2: index, %arg3: i1, %arg4: vector<32x64xf16>, %other: memref<?x64xf16, strided<[64, 1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[%arg1, 0] [%arg2, 64] [1, 1] : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  %alloc = memref.alloc() alignment = 64 : memref<32x64xf16>
  %0 = scf.if %arg3 -> (memref<?x64xf16, strided<[?, ?], offset: ?>>) {
    vector.transfer_write %arg4, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16>
    %subview_0 = memref.subview %alloc[0, 0] [%arg2, 64] [1, 1] : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    %cast = memref.cast %subview_0 : memref<?x64xf16, strided<[64, 1]>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<?x64xf16, strided<[?, ?], offset: ?>>
  } else {
    %cast = memref.cast %other : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<?x64xf16, strided<[?, ?], offset: ?>>
  }
  memref.copy %0, %subview : memref<?x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  return
}

// -----

// Moving the copy into the guard would place it before the intervening store
// and change the final destination value. Preserve the original ordering.

// CHECK-LABEL: func.func @guarded_copy_after_store
// CHECK:         %[[RES:.+]] = scf.if
// CHECK:         memref.store
// CHECK-NEXT:    memref.copy %[[RES]]
func.func @guarded_copy_after_store(%dst: memref<32x64xf16>, %cond: i1, %v: vector<32x64xf16>, %x: f16) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() alignment = 64 : memref<32x64xf16>
  %0 = scf.if %cond -> (memref<32x64xf16>) {
    vector.transfer_write %v, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16>
    scf.yield %alloc : memref<32x64xf16>
  } else {
    scf.yield %dst : memref<32x64xf16>
  }
  memref.store %x, %dst[%c0, %c0] : memref<32x64xf16>
  memref.copy %0, %dst : memref<32x64xf16> to memref<32x64xf16>
  return
}

// -----

// The destination cast follows the guard and cannot be used inside it. The
// moved copy must use the underlying buffer, which dominates the then branch.

// CHECK-LABEL: func.func @guarded_copy_into_late_cast
// CHECK-NOT:     memref.alloc
// CHECK:         scf.if %arg1 {
// CHECK-NEXT:      memref.store %arg2, %arg0
// CHECK-NEXT:    }
// CHECK-NOT:     memref.copy
func.func @guarded_copy_into_late_cast(%dst: memref<?x64xf16>, %cond: i1, %x: f16) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() alignment = 64 : memref<32x64xf16>
  %0 = scf.if %cond -> (memref<?x64xf16>) {
    memref.store %x, %alloc[%c0, %c0] : memref<32x64xf16>
    %cast = memref.cast %alloc : memref<32x64xf16> to memref<?x64xf16>
    scf.yield %cast : memref<?x64xf16>
  } else {
    scf.yield %dst : memref<?x64xf16>
  }
  %cast = memref.cast %dst : memref<?x64xf16> to memref<32x64xf16>
  memref.copy %0, %cast : memref<?x64xf16> to memref<32x64xf16>
  return
}

// -----

// The else branch recomputes the destination as an equivalent subview. Hoist
// it ahead of the guard and use it as the moved copy's target.

// CHECK-LABEL: func.func @guarded_copy_else_subview
// CHECK-NOT:     memref.alloc
// CHECK:         %[[DST:.+]] = memref.subview %arg0[%arg1, 0] [%arg2, 64]
// CHECK:         scf.if %arg3 {
// CHECK-NEXT:      vector.transfer_write %arg4, %[[DST]]{{.*}} {in_bounds = [false, true]}
// CHECK-NEXT:    }
// CHECK-NOT:     memref.copy
func.func @guarded_copy_else_subview(%arg0: memref<?x64xf16>, %arg1: index, %arg2: index, %arg3: i1, %arg4: vector<32x64xf16>) {
  %c0 = arith.constant 0 : index
  %0 = scf.if %arg3 -> (memref<?x64xf16, strided<[?, ?], offset: ?>>) {
    %alloc = memref.alloc() alignment = 64 : memref<32x64xf16>
    vector.transfer_write %arg4, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16>
    %subview_0 = memref.subview %alloc[0, 0] [%arg2, 64] [1, 1] : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    %cast = memref.cast %subview_0 : memref<?x64xf16, strided<[64, 1]>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<?x64xf16, strided<[?, ?], offset: ?>>
  } else {
    %subview_1 = memref.subview %arg0[%arg1, 0] [%arg2, 64] [1, 1] : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    %cast = memref.cast %subview_1 : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<?x64xf16, strided<[?, ?], offset: ?>>
  }
  %subview = memref.subview %arg0[%arg1, 0] [%arg2, 64] [1, 1] : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  memref.copy %0, %subview : memref<?x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  return
}

// -----

// A copy between two views of the same buffer is a no-op.

// CHECK-LABEL: func.func @copy_same_buffer
// CHECK-NOT:     memref.copy
func.func @copy_same_buffer(%arg0: memref<?x64xf16>, %arg1: index, %arg2: index) {
  %a = memref.subview %arg0[%arg1, 0] [%arg2, 64] [1, 1] : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  %b = memref.subview %arg0[%arg1, 0] [%arg2, 64] [1, 1] : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  memref.copy %a, %b : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  return
}

// -----

// Each branch of the guard yields the destination's own subview. The two
// copies are no-ops even though their sources are scf.if results.
// CHECK-LABEL: func.func @multi_result_guarded_same_buffer
// CHECK:         scf.if
// CHECK:           memref.store
// CHECK-NOT:     memref.copy
// CHECK:         return
func.func @multi_result_guarded_same_buffer(
    %dst0: memref<?x64xf32>, %dst1: memref<?x64xf32>,
    %row: index, %rows: index, %cond: i1, %value: f32) {
  %c0 = arith.constant 0 : index
  %r:2 = scf.if %cond ->
      (memref<?x64xf32, strided<[64, 1], offset: ?>>,
       memref<?x64xf32, strided<[64, 1], offset: ?>>) {
    %a = memref.subview %dst0[%row, 0] [%rows, 64] [1, 1]
        : memref<?x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
    %b = memref.subview %dst1[%row, 0] [%rows, 64] [1, 1]
        : memref<?x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
    memref.store %value, %a[%c0, %c0]
        : memref<?x64xf32, strided<[64, 1], offset: ?>>
    scf.yield %a, %b
        : memref<?x64xf32, strided<[64, 1], offset: ?>>,
          memref<?x64xf32, strided<[64, 1], offset: ?>>
  } else {
    %a = memref.subview %dst0[%row, 0] [%rows, 64] [1, 1]
        : memref<?x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
    %b = memref.subview %dst1[%row, 0] [%rows, 64] [1, 1]
        : memref<?x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
    scf.yield %a, %b
        : memref<?x64xf32, strided<[64, 1], offset: ?>>,
          memref<?x64xf32, strided<[64, 1], offset: ?>>
  }
  %a = memref.subview %dst0[%row, 0] [%rows, 64] [1, 1]
      : memref<?x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
  %b = memref.subview %dst1[%row, 0] [%rows, 64] [1, 1]
      : memref<?x64xf32> to memref<?x64xf32, strided<[64, 1], offset: ?>>
  memref.copy %r#0, %a
      : memref<?x64xf32, strided<[64, 1], offset: ?>>
        to memref<?x64xf32, strided<[64, 1], offset: ?>>
  memref.copy %r#1, %b
      : memref<?x64xf32, strided<[64, 1], offset: ?>>
        to memref<?x64xf32, strided<[64, 1], offset: ?>>
  return
}

// -----

// Both guarded results come from temporary buffers in the then branch and
// from the destinations in the else branch. Sink both copies into the guard,
// then redirect their writes to the two destination views.
// CHECK-LABEL: func.func @multi_result_guarded_alloc_copies
// CHECK-NOT:     memref.alloc
// CHECK:         scf.if
// CHECK-DAG:       vector.transfer_write %arg5
// CHECK-DAG:       vector.transfer_write %arg6
// CHECK-NOT:     memref.copy
// CHECK:         return
func.func @multi_result_guarded_alloc_copies(
    %dst0: memref<?x64xf16>, %dst1: memref<?x64xf16>,
    %row: index, %rows: index, %cond: i1,
    %v0: vector<32x64xf16>, %v1: vector<32x64xf16>) {
  %c0 = arith.constant 0 : index
  %r:2 = scf.if %cond ->
      (memref<?x64xf16, strided<[?, ?], offset: ?>>,
       memref<?x64xf16, strided<[?, ?], offset: ?>>) {
    %alloc0 = memref.alloc() alignment = 64 : memref<32x64xf16>
    %alloc1 = memref.alloc() alignment = 64 : memref<32x64xf16>
    vector.transfer_write %v0, %alloc0[%c0, %c0] {in_bounds = [true, true]}
        : vector<32x64xf16>, memref<32x64xf16>
    vector.transfer_write %v1, %alloc1[%c0, %c0] {in_bounds = [true, true]}
        : vector<32x64xf16>, memref<32x64xf16>
    %a = memref.subview %alloc0[0, 0] [%rows, 64] [1, 1]
        : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    %b = memref.subview %alloc1[0, 0] [%rows, 64] [1, 1]
        : memref<32x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    %cast_a = memref.cast %a
        : memref<?x64xf16, strided<[64, 1]>>
          to memref<?x64xf16, strided<[?, ?], offset: ?>>
    %cast_b = memref.cast %b
        : memref<?x64xf16, strided<[64, 1]>>
          to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast_a, %cast_b
        : memref<?x64xf16, strided<[?, ?], offset: ?>>,
          memref<?x64xf16, strided<[?, ?], offset: ?>>
  } else {
    %a = memref.subview %dst0[%row, 0] [%rows, 64] [1, 1]
        : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    %b = memref.subview %dst1[%row, 0] [%rows, 64] [1, 1]
        : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    %cast_a = memref.cast %a
        : memref<?x64xf16, strided<[64, 1], offset: ?>>
          to memref<?x64xf16, strided<[?, ?], offset: ?>>
    %cast_b = memref.cast %b
        : memref<?x64xf16, strided<[64, 1], offset: ?>>
          to memref<?x64xf16, strided<[?, ?], offset: ?>>
    scf.yield %cast_a, %cast_b
        : memref<?x64xf16, strided<[?, ?], offset: ?>>,
          memref<?x64xf16, strided<[?, ?], offset: ?>>
  }
  %target0 = memref.subview %dst0[%row, 0] [%rows, 64] [1, 1]
      : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  memref.copy %r#0, %target0
      : memref<?x64xf16, strided<[?, ?], offset: ?>>
        to memref<?x64xf16, strided<[64, 1], offset: ?>>
  %target1 = memref.subview %dst1[%row, 0] [%rows, 64] [1, 1]
      : memref<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
  memref.copy %r#1, %target1
      : memref<?x64xf16, strided<[?, ?], offset: ?>>
        to memref<?x64xf16, strided<[64, 1], offset: ?>>
  return
}
