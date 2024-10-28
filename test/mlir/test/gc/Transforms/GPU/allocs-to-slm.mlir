// RUN: gc-opt %s --allocs-to-slm | FileCheck %s

func.func @entry() {
  %c1 = arith.constant 1 : index

  // Memory space wasn't assigned as it's allocated outside of gpu.launch block
  // CHECK: %[[NEW_MEMREF_0:.*]] = memref.alloc() : memref<16x16xf16>
  %0 = memref.alloc() : memref<16x16xf16>
  gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %c1, %sz_by = %c1, %sz_bz = %c1)
             threads(%tx, %ty, %tz) in (%sz_tx = %c1, %sz_ty = %c1, %sz_tz = %c1) {
    // Memory space was changed as it's explicitly specifided
    // CHECK: %[[NEW_MEMREF_1:.*]] = memref.alloc() : memref<16x16xf16, 1>
    %1 = memref.alloc() : memref<16x16xf16, 1>
    // Added 'shared' memory space
    // CHECK: %[[NEW_MEMREF_2:.*]] = memref.alloc() : memref<16x16xf16, 3>
    %2 = memref.alloc() : memref<16x16xf16>

    // CHECK: linalg.add ins(%[[NEW_MEMREF_1]], %[[NEW_MEMREF_2]] : memref<16x16xf16, 1>, memref<16x16xf16, 3>) outs(%[[NEW_MEMREF_0]] : memref<16x16xf16>)
    linalg.add ins(%1, %2 :memref<16x16xf16, 1>, memref<16x16xf16>) outs(%0 : memref<16x16xf16>)
    // CHECK: memref.dealloc %[[NEW_MEMREF_1]] : memref<16x16xf16, 1>
    // CHECK: memref.dealloc %[[NEW_MEMREF_2]] : memref<16x16xf16, 3>
    memref.dealloc %1 : memref<16x16xf16, 1>
    memref.dealloc %2 : memref<16x16xf16>
    gpu.terminator
  }
  return
}
