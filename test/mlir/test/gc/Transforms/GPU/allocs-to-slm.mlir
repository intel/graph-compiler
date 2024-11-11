// RUN: gc-opt %s --allocs-to-slm | FileCheck %s

func.func @entry() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // Memory space wasn't assigned as it's allocated outside of gpu.launch block
  // CHECK: %[[NEW_MEMREF_0:.*]] = memref.alloc() : memref<16x32xf16>
  %0 = memref.alloc() : memref<16x32xf16>
  // Capture thread-id variables
  // CHECK: gpu.launch blocks(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]]) in (%[[ARG6:.+]] = %c2, %[[ARG7:.+]] = %c2, %[[ARG8:.+]] = %c1) threads
  // CHECK-SAME: (%[[THREAD_X:.+]], %[[THREAD_Y:.+]], %[[ARG5:.+]]) in
  // CHECK-SAME: (%[[ARG9:.+]] = %c2, %[[ARG10:.+]] = %c4, %[[ARG11:.+]] = %c1) {
  gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %c2, %sz_by = %c2, %sz_bz = %c1)
             threads(%tx, %ty, %tz) in (%sz_tx = %c2, %sz_ty = %c4, %sz_tz = %c1) {
    // Memory space was changed as it's explicitly specifided
    // CHECK: %[[NEW_MEMREF_1:.*]] = memref.alloc() : memref<16x32xf16, 1>
    %1 = memref.alloc() : memref<16x32xf16, 1>
    // Added 'shared' memory space
    // CHECK: %[[NEW_MEMREF_2:.*]] = memref.alloc() : memref<32x128xf16, 3>
    // CHECK: %[[OFF_X:.*]] = arith.muli %[[THREAD_X]], %c16 : index
    // CHECK: %[[OFF_Y:.*]] = arith.muli %[[THREAD_Y]], %c32 : index
    // CHECK: %[[NEW_MEMREF_3:.*]] = memref.subview %[[NEW_MEMREF_2]][%[[OFF_X]], %[[OFF_Y]]] [16, 32] [1, 1]
    // CHECK-SAME: memref<32x128xf16, 3> to memref<16x32xf16, strided<[128, 1], offset: ?>, 3>
    %2 = memref.alloc() : memref<16x32xf16>

    // CHECK: linalg.add ins(%[[NEW_MEMREF_1]], %[[NEW_MEMREF_3]] :
    // CHECK-SAME: memref<16x32xf16, 1>, memref<16x32xf16, strided<[128, 1], offset: ?>, 3>) outs(%[[NEW_MEMREF_0]] : memref<16x32xf16>)
    linalg.add ins(%1, %2 :memref<16x32xf16, 1>, memref<16x32xf16>) outs(%0 : memref<16x32xf16>)
    // CHECK: memref.dealloc %[[NEW_MEMREF_1]] : memref<16x32xf16, 1>
    // Verify that there are no deallocs for SLM
    // CHECK-NOT: memref.dealloc %[[NEW_MEMREF_2]] .*
    // CHECK-NOT: memref.dealloc %[[NEW_MEMREF_3]] .*
    memref.dealloc %1 : memref<16x32xf16, 1>
    memref.dealloc %2 : memref<16x32xf16>
    gpu.terminator
  }
  return
}
