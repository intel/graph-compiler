// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file | FileCheck %s

// TODO: write CHECK directives

#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 16)>

func.func @entry(%arg0: memref<128x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<128x1024xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c2, %arg12 = %c4, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c4, %arg15 = %c16, %arg16 = %c1) {
    %x_group_idx = affine.apply #map(%arg5)
    %y_group_idx = affine.apply #map(%arg6)

    %x_thread_idx = affine.apply #map1(%arg8)
    %y_thread_idx = affine.apply #map1(%arg9)

    %x_global_idx = arith.addi %x_group_idx, %x_thread_idx : index
    %y_global_idx = arith.addi %y_group_idx, %y_thread_idx : index

    %a_subview = memref.subview %arg0[%x_global_idx, 0] [16, 1024] [1, 1] : memref<128x1024xf16> to memref<16x1024xf16, strided<[1024, 1], offset: ?>>
    %b_subview = memref.subview %arg1[0, %y_global_idx] [1024, 16] [1, 1] : memref<1024x1024xf16> to memref<1024x16xf16, strided<[1024, 1], offset: ?>>

    %slm_buff = memref.alloc() : memref<64x256xf16, 3>
    %slm_subview = memref.subview %slm_buff[%x_thread_idx, %y_thread_idx] [16, 16] [1, 1] : memref<64x256xf16, 3> to memref<16x16xf16, strided<[256, 1], offset: ?>, 3>

    linalg.fill ins(%cst : f16) outs(%slm_subview : memref<16x16xf16, strided<[256, 1], offset: ?>, 3>)
    linalg.matmul ins(%a_subview, %b_subview : memref<16x1024xf16, strided<[1024, 1], offset: ?>>, memref<1024x16xf16, strided<[1024, 1], offset: ?>>) outs(%slm_subview : memref<16x16xf16, strided<[256, 1], offset: ?>, 3>)

    %a_add_subview = memref.subview %arg0[%x_global_idx, %y_global_idx] [16, 16] [1, 1] : memref<128x1024xf16> to memref<16x16xf16, strided<[1024, 1], offset: ?>>
    %out_subview = memref.subview %arg2[%x_global_idx, %y_global_idx] [16, 16] [1, 1] : memref<128x1024xf16> to memref<16x16xf16, strided<[1024, 1], offset: ?>>

    linalg.add ins(%slm_subview, %a_add_subview : memref<16x16xf16, strided<[256, 1], offset: ?>, 3>, memref<16x16xf16, strided<[1024, 1], offset: ?>>) outs(%out_subview : memref<16x16xf16, strided<[1024, 1], offset: ?>>)
    gpu.terminator
  }
  return
}
