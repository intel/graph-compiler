// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file -cse | FileCheck %s

!input_type = memref<2x4x8x16xf16>
!chunk_type = memref<1x1x8x16xf16, strided<[512, 128, 16, 1], offset: ?>>
!slm_chunk = memref<1x1x8x16xf16, strided<[128, 128, 16, 1], offset: ?>, 3>

// The map that computes an offset for SLM
// CHECK: #map = affine_map<(d0, d1) -> (d0 * 4 + d1)>
#map = affine_map<(xi, yi) -> (xi * 4 + yi)>

func.func @entry(%arg0: !input_type, %arg1: !input_type, %arg2: !input_type) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c2, %arg13 = %c4, %arg14 = %c1) {
    // CHECK: %[[ARG0_SB:.+]] = memref.subview %arg0[%arg6, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1]
    %arg0_sb = memref.subview %arg0[%arg6, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : !input_type to !chunk_type
    // CHECK: %[[ARG1_SB:.+]] = memref.subview %arg1[%arg6, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1]
    %arg1_sb = memref.subview %arg1[%arg6, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : !input_type to !chunk_type
    // CHECK: %[[ARG2_SB:.+]] = memref.subview %arg2[%arg6, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1]
    %arg2_sb = memref.subview %arg2[%arg6, %arg7, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : !input_type to !chunk_type

    // CHECK: %[[SLM_BUFF:.+]] = memref.alloc() : memref<8x1x8x16xf16, 3>
    %slm_root = memref.alloc() : memref<8x1x8x16xf16, 3>

    %slm_idx = affine.apply #map(%arg6, %arg7)
    %slm = memref.subview %slm_root[%slm_idx, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<8x1x8x16xf16, 3> to !slm_chunk

    // Squeezing the arguments of 'linalg.mul'
    // CHECK: %[[ARG0_SQUEEZ:.+]] = memref.subview %[[ARG0_SB]][0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] :
    // CHECK-SAME: memref<1x1x8x16xf16, strided<[512, 128, 16, 1], offset: ?>> to memref<8x16xf16, strided<[16, 1], offset: ?>>

    // CHECK: %[[ARG1_SQUEEZ:.+]] = memref.subview %[[ARG1_SB]][0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] :
    // CHECK-SAME: memref<1x1x8x16xf16, strided<[512, 128, 16, 1], offset: ?>> to memref<8x16xf16, strided<[16, 1], offset: ?>>

    // Verify that tensor descriptors are created from the squeezed memrefs
    // CHECK: xegpu.create_nd_tdesc %[[ARG0_SQUEEZ]]
    // CHECK: xegpu.create_nd_tdesc %[[ARG1_SQUEEZ]]

    // Verify that the SLM output of linalg.mul is squeezed correctly
    // CHECK-NOT: .* = memref.subview %[[SLM_BUFF]] .*
    // CHECK: %[[SLM_THREAD_OFF:.+]] = affine.apply #map(%arg6, %arg7)
    // CHECK: %[[SLM_OFF:.+]] = arith.muli %[[SLM_THREAD_OFF]], %c128 : index
    // CHECK: %[[FLAT_SLM:.+]] = memref.reinterpret_cast %[[SLM_BUFF]] to offset: [%c0], sizes: [%c1024], strides: [%c1] : memref<8x1x8x16xf16, 3> to memref<1024xf16, 3>
    // CHECK: xegpu.create_tdesc %[[FLAT_SLM]]
    linalg.mul ins(%arg0_sb, %arg1_sb : !chunk_type, !chunk_type) outs(%slm : !slm_chunk)

    // Squeezing the result buffer of 'linalg.add'
    // CHECK: %[[ARG2_SQUEEZ:.+]] = memref.subview %[[ARG2_SB]][0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] :
    // CHECK-SAME: memref<1x1x8x16xf16, strided<[512, 128, 16, 1], offset: ?>> to memref<8x16xf16, strided<[16, 1], offset: ?>>

    // Verify that tensor descriptors are created from the squeezed memrefs
    // CHECK: xegpu.create_nd_tdesc %[[ARG2_SQUEEZ]]
    linalg.add ins(%arg0_sb, %slm : !chunk_type, !slm_chunk) outs(%arg2_sb : !chunk_type)

    gpu.terminator
  } {SCFToGPU_visited}

  return
}
