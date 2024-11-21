// RUN: gc-opt %s --allocs-to-slm | FileCheck %s

// Computex thread offset for SLM: (Xthread_idx * Yblock_sz * Zblock_sz + Ythread_idx * Zblock_sz + Zthread_idx) * Xchunk_size
// CHECK: #map = affine_map<(d0, d1, d2) -> ((d0 * 12 + d1 * 4 + d2) * 2)>

func.func @entry() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  // Memory space wasn't assigned as it's allocated outside of gpu.launch block
  // CHECK: %[[NEW_MEMREF_0:.*]] = memref.alloc() : memref<2x3x16x32xf16>
  %0 = memref.alloc() : memref<2x3x16x32xf16>
  // Capture thread-id variables
  // CHECK: gpu.launch blocks(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]]) in (%[[ARG6:.+]] = %c2, %[[ARG7:.+]] = %c2, %[[ARG8:.+]] = %c1) threads
  // CHECK-SAME: (%[[THREAD_X:.+]], %[[THREAD_Y:.+]], %[[THREAD_Z:.+]]) in
  // CHECK-SAME: (%[[ARG9:.+]] = %c2, %[[ARG10:.+]] = %c3, %[[ARG11:.+]] = %c4) {
  gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %c2, %sz_by = %c2, %sz_bz = %c1)
             threads(%tx, %ty, %tz) in (%sz_tx = %c2, %sz_ty = %c3, %sz_tz = %c4) {
    // Memory space was changed as it's explicitly specifided
    // CHECK: %[[NEW_MEMREF_1:.*]] = memref.alloc() : memref<2x3x16x32xf16, 1>
    %1 = memref.alloc() : memref<2x3x16x32xf16, 1>
    // Added 'shared' memory space and allocated SLM for each thread (2 * 3 * 4 = 24; 24 * 2 = 48)
    // CHECK: %[[NEW_MEMREF_2:.*]] = memref.alloc() : memref<48x3x16x32xf16, 3>
    // CHECK: %[[OFF_X:.*]] = affine.apply #map(%[[THREAD_X]], %[[THREAD_Y]], %[[THREAD_Z]])
    // CHECK: %[[NEW_MEMREF_3:.*]] = memref.subview %[[NEW_MEMREF_2]][%[[OFF_X]], 0, 0, 0] [2, 3, 16, 32] [1, 1, 1, 1]
    // CHECK-SAME: memref<48x3x16x32xf16, 3> to memref<2x3x16x32xf16, strided<[1536, 512, 32, 1], offset: ?>, 3>
    %2 = memref.alloc() : memref<2x3x16x32xf16>

    // CHECK: linalg.add ins(%[[NEW_MEMREF_1]], %[[NEW_MEMREF_3]] :
    // CHECK-SAME: memref<2x3x16x32xf16, 1>, memref<2x3x16x32xf16, strided<[1536, 512, 32, 1], offset: ?>, 3>) outs(%[[NEW_MEMREF_0]] : memref<2x3x16x32xf16>)
    linalg.add ins(%1, %2 :memref<2x3x16x32xf16, 1>, memref<2x3x16x32xf16>) outs(%0 : memref<2x3x16x32xf16>)
    // CHECK: memref.dealloc %[[NEW_MEMREF_1]] : memref<2x3x16x32xf16, 1>
    // Verify that there are no deallocs for SLM
    // CHECK-NOT: memref.dealloc %[[NEW_MEMREF_2]] .*
    // CHECK-NOT: memref.dealloc %[[NEW_MEMREF_3]] .*
    memref.dealloc %1 : memref<2x3x16x32xf16, 1>
    memref.dealloc %2 : memref<2x3x16x32xf16>
    gpu.terminator
  }
  return
}
