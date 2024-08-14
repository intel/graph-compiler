// RUN: gc-opt --split-input-file --sink-op-into-inner-loop %s | FileCheck %s

func.func @matmul_2Dx2D_f32(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c256 = arith.constant 256 : index
  %c2048 = arith.constant 2048 : index
  %c4096 = arith.constant 4096 : index
  %c32 = arith.constant 32 : index
  // CHECK: scf.forall
  // CHECK-NOT: affine.apply
  // CHECK-NOT: memref.subview
  // CHECK-NEXT: scf.forall
  scf.forall (%arg3) in (4) {
    %0 = affine.apply affine_map<(d0) -> (d0 * 1024)>(%arg3)
    %subview = memref.subview %arg2[%0, 0] [1024, 4096] [1, 1] : memref<4096x4096xf32> to memref<1024x4096xf32, strided<[4096, 1], offset: ?>>
    scf.forall (%arg4) in (2) {
      %1 = affine.apply affine_map<(d0) -> (d0 * 2048)>(%arg4)
      %subview_0 = memref.subview %subview[0, %1] [1024, 2048] [1, 1] : memref<1024x4096xf32, strided<[4096, 1], offset: ?>> to memref<1024x2048xf32, strided<[4096, 1], offset: ?>>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xf32>
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        %subview_1 = memref.subview %subview_0[%arg5, 0] [256, 2048] [1, 1] : memref<1024x2048xf32, strided<[4096, 1], offset: ?>> to memref<256x2048xf32, strided<[4096, 1], offset: ?>>
        scf.for %arg6 = %c0 to %c2048 step %c256 {
          %subview_2 = memref.subview %subview_1[0, %arg6] [256, 256] [1, 1] : memref<256x2048xf32, strided<[4096, 1], offset: ?>> to memref<256x256xf32, strided<[4096, 1], offset: ?>>
          scf.for %arg7 = %c0 to %c4096 step %c256 {
            %2 = arith.cmpi eq, %arg7, %c0 : index
            scf.for %arg8 = %c0 to %c256 step %c32 {
              %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * 1024 + s0 + s1)>(%arg3)[%arg8, %arg5]
              %subview_3 = memref.subview %arg0[%3, %arg7] [32, 256] [1, 1] : memref<4096x4096xf32> to memref<32x256xf32, strided<[4096, 1], offset: ?>>
              %subview_4 = memref.subview %subview_2[%arg8, 0] [32, 256] [1, 1] : memref<256x256xf32, strided<[4096, 1], offset: ?>> to memref<32x256xf32, strided<[4096, 1], offset: ?>>
              %expand_shape = memref.expand_shape %subview_3 [[0], [1, 2]] output_shape [32, 8, 32] : memref<32x256xf32, strided<[4096, 1], offset: ?>> into memref<32x8x32xf32, strided<[4096, 32, 1], offset: ?>>
              scf.for %arg9 = %c0 to %c256 step %c32 {

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