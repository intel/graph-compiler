// RUN: gc-opt --split-input-file --merge-nested-forall %s | FileCheck %s

// -----

#map = affine_map<(d0) -> (d0 * 1024)>
#map1 = affine_map<(d0) -> (d0 * 2048)>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * 2048 + s0 + s1)>
#map3 = affine_map<(d0)[s0, s1] -> (d0 * 1024 + s0 + s1)>
module {
  func.func @matmul_2Dx2D_f32(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    // CHECK: scf.forall {{.*}} (4, 2)
    scf.forall (%arg3) in (4) {
      scf.forall (%arg4) in (2) {
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xf32>
        scf.for %arg5 = %c0 to %c1024 step %c256 {
          %c2048 = arith.constant 2048 : index
          scf.for %arg6 = %c0 to %c2048 step %c256 {
            %c4096 = arith.constant 4096 : index
            scf.for %arg7 = %c0 to %c4096 step %c256 {
              %c32 = arith.constant 32 : index
              scf.for %arg8 = %c0 to %c256 step %c32 {
                scf.for %arg9 = %c0 to %c256 step %c32 {
                  %0 = affine.apply #map(%arg3)
                  %1 = affine.apply #map1(%arg4)
                  %subview = memref.subview %arg2[%0, 0] [1024, 4096] [1, 1] : memref<4096x4096xf32> to memref<1024x4096xf32, strided<[4096, 1], offset: ?>>
                  %subview_0 = memref.subview %subview[0, %1] [1024, 2048] [1, 1] : memref<1024x4096xf32, strided<[4096, 1], offset: ?>> to memref<1024x2048xf32, strided<[4096, 1], offset: ?>>
                  %subview_1 = memref.subview %subview_0[%arg5, 0] [256, 2048] [1, 1] : memref<1024x2048xf32, strided<[4096, 1], offset: ?>> to memref<256x2048xf32, strided<[4096, 1], offset: ?>>
                  %subview_2 = memref.subview %subview_1[0, %arg6] [256, 256] [1, 1] : memref<256x2048xf32, strided<[4096, 1], offset: ?>> to memref<256x256xf32, strided<[4096, 1], offset: ?>>
                  %subview_3 = memref.subview %subview_2[%arg8, 0] [32, 256] [1, 1] : memref<256x256xf32, strided<[4096, 1], offset: ?>> to memref<32x256xf32, strided<[4096, 1], offset: ?>>
                  %c1 = arith.constant 1 : index
                  %c8 = arith.constant 8 : index
                  %2 = arith.cmpi eq, %arg7, %c0 : index
                  %3 = affine.apply #map2(%arg4)[%arg9, %arg6]
                  %subview_4 = memref.subview %arg1[%arg7, %3] [256, 32] [1, 1] : memref<4096x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                  %subview_5 = memref.subview %subview_3[0, %arg9] [32, 32] [1, 1] : memref<32x256xf32, strided<[4096, 1], offset: ?>> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
                  scf.parallel (%arg10, %arg11, %arg12) = (%c0, %c0, %c0) to (%c8, %c32, %c32) step (%c1, %c1, %c1) {
                    %4 = affine.apply #map3(%arg3)[%arg8, %arg5]
                    %subview_6 = memref.subview %arg0[%4, %arg7] [32, 256] [1, 1] : memref<4096x4096xf32> to memref<32x256xf32, strided<[4096, 1], offset: ?>>
                    %expand_shape_7 = memref.expand_shape %subview_6 [[0], [1, 2]] output_shape [32, 8, 32] : memref<32x256xf32, strided<[4096, 1], offset: ?>> into memref<32x8x32xf32, strided<[4096, 32, 1], offset: ?>>
                    %5 = memref.load %expand_shape_7[%arg11, %arg10, %arg12] : memref<32x8x32xf32, strided<[4096, 32, 1], offset: ?>>
                    memref.store %5, %alloc[%arg10, %arg11, %arg12] : memref<8x32x32xf32>
                    scf.reduce 
                  }
                  %expand_shape = memref.expand_shape %subview_4 [[0, 1], [2]] output_shape [8, 32, 32] : memref<256x32xf32, strided<[4096, 1], offset: ?>> into memref<8x32x32xf32, strided<[131072, 4096, 1], offset: ?>>
                  scf.if %2 {
                    scf.parallel (%arg10, %arg11) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
                      %cst = arith.constant 0.000000e+00 : f32
                      memref.store %cst, %subview_5[%arg10, %arg11] : memref<32x32xf32, strided<[4096, 1], offset: ?>>
                      scf.reduce 
                    }
                    scf.for %arg10 = %c0 to %c8 step %c1 {
                      scf.parallel (%arg11, %arg12) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
                        scf.for %arg13 = %c0 to %c32 step %c1 {
                          %4 = memref.load %alloc[%arg10, %arg11, %arg13] : memref<8x32x32xf32>
                          %5 = memref.load %expand_shape[%arg10, %arg13, %arg12] : memref<8x32x32xf32, strided<[131072, 4096, 1], offset: ?>>
                          %6 = memref.load %subview_5[%arg11, %arg12] : memref<32x32xf32, strided<[4096, 1], offset: ?>>
                          %7 = arith.mulf %4, %5 : f32
                          %8 = arith.addf %6, %7 : f32
                          memref.store %8, %subview_5[%arg11, %arg12] : memref<32x32xf32, strided<[4096, 1], offset: ?>>
                        }
                        scf.reduce 
                      }
                    }
                  } else {
                    scf.for %arg10 = %c0 to %c8 step %c1 {
                      scf.parallel (%arg11, %arg12) = (%c0, %c0) to (%c32, %c32) step (%c1, %c1) {
                        scf.for %arg13 = %c0 to %c32 step %c1 {
                          %4 = memref.load %alloc[%arg10, %arg11, %arg13] : memref<8x32x32xf32>
                          %5 = memref.load %expand_shape[%arg10, %arg13, %arg12] : memref<8x32x32xf32, strided<[131072, 4096, 1], offset: ?>>
                          %6 = memref.load %subview_5[%arg11, %arg12] : memref<32x32xf32, strided<[4096, 1], offset: ?>>
                          %7 = arith.mulf %4, %5 : f32
                          %8 = arith.addf %6, %7 : f32
                          memref.store %8, %subview_5[%arg11, %arg12] : memref<32x32xf32, strided<[4096, 1], offset: ?>>
                        }
                        scf.reduce 
                      }
                    }
                  }
                }
              }
            }
          }
        }
        memref.dealloc %alloc : memref<8x32x32xf32>
      }
    }
    return
  }
}

