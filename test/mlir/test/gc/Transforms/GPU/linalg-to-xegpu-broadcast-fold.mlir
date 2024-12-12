// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @broadcast_eliminate_2d
func.func @broadcast_eliminate_2d() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[MEMREF_0:.*]] = memref.alloc() : memref<7x128xf16>
  %0 = memref.alloc() : memref<7x128xf16>
  // CHECK: %[[MEMREF_2:.*]] = memref.alloc() : memref<1x1x7x128xf16>
  %2 = memref.alloc() : memref<1x1x7x128xf16>
  // CHECK: %[[MEMREF_3:.*]] = memref.alloc() : memref<1x1x7x128xf16>
  %3 = memref.alloc() : memref<1x1x7x128xf16>

  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg11 = %c2, %arg12 = %c4, %arg13 = %c1) threads(%arg6, %arg7, %arg8) in (%arg14 = %c4, %arg15 = %c1, %arg16 = %c1) {
    // CHECK-NOT: memref.alloc() : memref<4x1x7x128xf16, 3>
    %slm_base = memref.alloc() : memref<4x1x7x128xf16, 3>
    %1 = memref.subview %slm_base[%arg6, 0, 0, 0] [1, 1, 7, 128] [1, 1, 1, 1] : memref<4x1x7x128xf16, 3> to memref<1x1x7x128xf16, strided<[896, 896, 128, 1], offset: ?>, 3>
  
    // CHECK-NOT: linalg.broadcast
    linalg.broadcast ins(%0 : memref<7x128xf16>) outs(%1 : memref<1x1x7x128xf16, strided<[896, 896, 128, 1], offset: ?>, 3>) dimensions = [0, 1]
    // CHECK: xegpu.create_nd_tdesc %[[MEMREF_0]]
    linalg.add ins(%1, %2 : memref<1x1x7x128xf16, strided<[896, 896, 128, 1], offset: ?>, 3>, memref<1x1x7x128xf16>) outs(%3 : memref<1x1x7x128xf16>)
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @broadcast_eliminate
func.func @broadcast_eliminate_3d() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[MEMREF_0:.*]] = memref.alloc() : memref<1x7x128xf16>
  %0 = memref.alloc() : memref<1x7x128xf16>
  // CHECK: %[[MEMREF_2:.*]] = memref.alloc() : memref<1x1x7x128xf16>
  %2 = memref.alloc() : memref<1x1x7x128xf16>
  // CHECK: %[[MEMREF_3:.*]] = memref.alloc() : memref<1x1x7x128xf16>
  %3 = memref.alloc() : memref<1x1x7x128xf16>

  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg11 = %c2, %arg12 = %c4, %arg13 = %c1) threads(%arg6, %arg7, %arg8) in (%arg14 = %c4, %arg15 = %c1, %arg16 = %c1) {
    // CHECK-NOT: memref.alloc() : memref<4x1x7x128xf16, 3>
    %slm_base = memref.alloc() : memref<4x1x7x128xf16, 3>
    %1 = memref.subview %slm_base[%arg6, 0, 0, 0] [1, 1, 7, 128] [1, 1, 1, 1] : memref<4x1x7x128xf16, 3> to memref<1x1x7x128xf16, strided<[896, 896, 128, 1], offset: ?>, 3>
  
    // CHECK-NOT: linalg.broadcast
    linalg.broadcast ins(%0 : memref<1x7x128xf16>) outs(%1 : memref<1x1x7x128xf16, strided<[896, 896, 128, 1], offset: ?>, 3>) dimensions = [0]
    // Squeezing the %0 before passing to 'linalg.add'
    // CHECK: %[[MEMREF0_SQUEEZ:.+]] = memref.subview %[[MEMREF_0]][0, 0, 0] [1, 7, 128] [1, 1, 1] :
    // CHECK-SAME: memref<1x7x128xf16> to memref<7x128xf16, strided<[128, 1]>>
    // CHECK: xegpu.create_nd_tdesc %[[MEMREF0_SQUEEZ]]
    linalg.add ins(%1, %2 : memref<1x1x7x128xf16, strided<[896, 896, 128, 1], offset: ?>, 3>, memref<1x1x7x128xf16>) outs(%3 : memref<1x1x7x128xf16>)
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @complex_broadcast
func.func @complex_broadcast_3d() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[MEMREF_0:.*]] = memref.alloc() : memref<7x128xf16>
  %0 = memref.alloc() : memref<7x128xf16>
  // CHECK: %[[MEMREF_1:.*]] = memref.alloc() : memref<7x7x128xf16>
  %1 = memref.alloc() : memref<7x7x128xf16>
  // CHECK: %[[MEMREF_2:.*]] = memref.alloc() : memref<7x7x128xf16>
  %2 = memref.alloc() : memref<7x7x128xf16>
  // CHECK: %[[MEMREF_3:.*]] = memref.alloc() : memref<7x7x128xf16>
  %3 = memref.alloc() : memref<7x7x128xf16>

  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg11 = %c2, %arg12 = %c4, %arg13 = %c1) threads(%arg6, %arg7, %arg8) in (%arg14 = %c4, %arg15 = %c1, %arg16 = %c1) {
    // This broadcast can't be replaced by a single memref.subview. Can't remove it
    // CHECK: linalg.broadcast
    linalg.broadcast ins(%0 : memref<7x128xf16>) outs(%1 : memref<7x7x128xf16>) dimensions = [0]
    linalg.add ins(%1, %2 : memref<7x7x128xf16>, memref<7x7x128xf16>) outs(%3 : memref<7x7x128xf16>)
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @single_broadcast
func.func @single_broadcast() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[MEMREF_0:.*]] = memref.alloc() : memref<7x128xf16>
  %0 = memref.alloc() : memref<7x128xf16>
  // CHECK: %[[MEMREF_1:.*]] = memref.alloc() : memref<1x1x7x128xf16>
  %1 = memref.alloc() : memref<1x1x7x128xf16>

  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg11 = %c2, %arg12 = %c4, %arg13 = %c1) threads(%arg6, %arg7, %arg8) in (%arg14 = %c4, %arg15 = %c1, %arg16 = %c1) {
    // broadcast result is not an input of any xegpu operation, we can't lower it
    // CHECK: linalg.broadcast
    linalg.broadcast ins(%0 : memref<7x128xf16>) outs(%1 : memref<1x1x7x128xf16>) dimensions = [0, 1]
    gpu.terminator
  }
  return
}
