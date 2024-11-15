// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file -cse | FileCheck %s

#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 16)>

func.func @entry(%arg0: memref<128x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<128x1024xf16>) {
  // CHECK: %[[loadAccumMatmul:.+]] = arith.constant dense<0.000000e+00> : vector<4x32xf16>
  // CHECK: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<32xf16>
  // CHECK: %[[colTileShift:.+]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271]> : vector<32xindex>
  // CHECK: %[[loadOffset:.+]] = arith.constant dense<512> : vector<32xindex>
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg11 = %c2, %arg12 = %c4, %arg13 = %c1) threads(%arg6, %arg7, %arg8) in (%arg14 = %c4, %arg15 = %c16, %arg16 = %c1) {
    %x_group_idx = affine.apply #map(%arg3)
    %y_group_idx = affine.apply #map(%arg4)

    // CHECK: %[[X_THREAD_IDX:.+]] = affine.apply #map1(%arg6)
    // CHECK: %[[Y_THREAD_IDX:.+]] = affine.apply #map1(%arg7)
    %x_thread_idx = affine.apply #map1(%arg6)
    %y_thread_idx = affine.apply #map1(%arg7)

    %x_global_idx = arith.addi %x_group_idx, %x_thread_idx : index
    %y_global_idx = arith.addi %y_group_idx, %y_thread_idx : index

    %a_subview = memref.subview %arg0[%x_global_idx, 0] [16, 1024] [1, 1] : memref<128x1024xf16> to memref<16x1024xf16, strided<[1024, 1], offset: ?>>
    %b_subview = memref.subview %arg1[0, %y_global_idx] [1024, 16] [1, 1] : memref<1024x1024xf16> to memref<1024x16xf16, strided<[1024, 1], offset: ?>>

    // CHECK: %[[SLM_BUFF:.+]] = memref.alloc() : memref<64x256xf16, 3>
    %slm_buff = memref.alloc() : memref<64x256xf16, 3>
    // CHECK-NOT: .* = memref.subview %[[SLM_BUFF]] .*
    // CHECK: %[[SLM_X_OFF:.+]] = arith.muli %[[X_THREAD_IDX]], %c256 : index
    // CHECK: %[[SLM_THREAD_OFF:.+]] = arith.addi %[[SLM_X_OFF]], %[[Y_THREAD_IDX]] : index
    // CHECK: %[[FLAT_SLM:.+]] = memref.reinterpret_cast %[[SLM_BUFF]] to offset: [%c0], sizes: [%c16384], strides: [%c1] : memref<64x256xf16, 3> to memref<16384xf16, 3>
    %slm_subview = memref.subview %slm_buff[%x_thread_idx, %y_thread_idx] [16, 16] [1, 1] : memref<64x256xf16, 3> to memref<16x16xf16, strided<[256, 1], offset: ?>, 3>

    // CHECK: %[[SLM_THREAD_OFF_V:.+]] = vector.splat %[[SLM_THREAD_OFF]] : vector<32xindex>
    // CHECK: %[[DESC_OFFSET0:.+]] = arith.addi %[[SLM_THREAD_OFF_V]], %[[colTileShift]] : vector<32xindex>
    // CHECK: %[[ROOT_DESC:.+]] = xegpu.create_tdesc %[[FLAT_SLM]], %[[DESC_OFFSET0]] : memref<16384xf16, 3>, vector<32xindex> -> !xegpu.tensor_desc<32xf16, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 1 : i64>>
    // CHECK: %[[FILL_DESC1:.+]] = xegpu.update_offset %[[ROOT_DESC]], %[[loadOffset]]
    // CHECK: %[[FILL_DESC2:.+]] = xegpu.update_offset %[[FILL_DESC1]], %[[loadOffset]]
    // CHECK-COUNT-5: xegpu.update_offset

    // CHECK: xegpu.store %[[ZERO]], %[[ROOT_DESC]]
    // CHECK: xegpu.store %[[ZERO]], %[[FILL_DESC1]]
    // CHECK-COUNT-6: xegpu.store
    linalg.fill ins(%cst : f16) outs(%slm_subview : memref<16x16xf16, strided<[256, 1], offset: ?>, 3>)

    // CHECK: %[[MATMUL_DESC1:.+]] = xegpu.update_offset %[[ROOT_DESC]], %[[loadOffset]]
    // CHECK: %[[MATMUL_DESC2:.+]] = xegpu.update_offset %[[MATMUL_DESC1]], %[[loadOffset]]
    // CHECK-COUNT-5: xegpu.update_offset

    // CHECK: %[[MATMUL_LOAD0:.+]] = xegpu.load %[[ROOT_DESC]]
    // CHECK-NEXT: %[[loadAccumMatmul1:.+]] = vector.insert %[[MATMUL_LOAD0]], %[[loadAccumMatmul]] [0]
    // CHECK-NEXT: %[[MATMUL_LOAD1:.+]] = xegpu.load %[[MATMUL_DESC1]]
    // CHECK-NEXT: %[[loadAccumMatmul2:.+]] = vector.insert %[[MATMUL_LOAD1]], %[[loadAccumMatmul1]] [1]
    // CHECK-COUNT-2: xegpu.load

    // CHECK: vector.shape_cast
    // CHECK-SAME: vector<4x32xf16> to vector<128xf16>
    // CHECK: vector.shape_cast
    // CHECK-SAME: vector<128xf16> to vector<8x16xf16>

    // CHECK-COUNT-4: xegpu.load
    // CHECK: vector.shape_cast
    // CHECK-SAME: vector<4x32xf16> to vector<128xf16>
    // CHECK: vector.shape_cast
    // CHECK-SAME: vector<128xf16> to vector<8x16xf16>

    // STORE:
    // %[[FLAT_MATMUL_RES0:.+]] = vector.shape_cast %[[MATMUL_RES0:.+]] : vector<8x16xf16> to vector<128xf16>
    // %[[STORE_TILE0:.+]] = vector.extract_strided_slice %[[FLAT_MATMUL_RES0]] {offsets = [0], sizes = [32], strides = [1]} : vector<128xf16> to vector<32xf16>
    // xegpu.store %[[STORE_TILE0]], %[[ROOT_DESC]]
    // %[[STORE_TILE1:.+]] = vector.extract_strided_slice %[[FLAT_MATMUL_RES0]] {offsets = [32], sizes = [32], strides = [1]} : vector<128xf16> to vector<32xf16>
    // xegpu.store %[[STORE_TILE0]], %[[MATMUL_DESC1]]
    // CHECK-COUNT-2: xegpu.store

    // %[[FLAT_MATMUL_RES1:.+]] = vector.shape_cast %[[MATMUL_RES1:.+]] : vector<8x16xf16> to vector<128xf16>
    // %[[STORE_TILE1_0:.+]] = vector.extract_strided_slice %[[FLAT_MATMUL_RES1]] {offsets = [0], sizes = [32], strides = [1]} : vector<128xf16> to vector<32xf16>
    // xegpu.store %[[STORE_TILE1_0]]
    // %[[STORE_TILE1_1:.+]] = vector.extract_strided_slice %[[FLAT_MATMUL_RES1]] {offsets = [32], sizes = [32], strides = [1]} : vector<128xf16> to vector<32xf16>
    // xegpu.store %[[STORE_TILE1_1]]
    // CHECK-COUNT-2: xegpu.store

    linalg.matmul ins(%a_subview, %b_subview : memref<16x1024xf16, strided<[1024, 1], offset: ?>>, memref<1024x16xf16, strided<[1024, 1], offset: ?>>) outs(%slm_subview : memref<16x16xf16, strided<[256, 1], offset: ?>, 3>)
    gpu.terminator
  }
  return
}
