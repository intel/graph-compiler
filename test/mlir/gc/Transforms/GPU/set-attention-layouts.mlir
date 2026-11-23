// RUN: gc-opt %s --set-attention-layouts | FileCheck %s

// Flash-attention k-loop (head_dim 256, 16 subgroups) as it reaches
// SetAttentionLayouts: S = Q K^T and O += P V as two xegpu.dpas, the K load
// transposed, the V load hoisted next to the K load, Q loaded before the loop.

// Q is re-loaded inside the loop right before the Q K^T dpas.
// CHECK-LABEL: gpu.func @sdpa_kernel
// CHECK:         scf.for
// CHECK:           xegpu.load_nd {{.*}}sg_layout = [1, 16], sg_data = [64, 256], order = [0, 1]{{.*}} -> vector<64x256xf16>
// CHECK:           xegpu.load_nd {{.*}}sg_layout = [16, 1], sg_data = [64, 256], inst_data = [32, 32]{{.*}} -> vector<64x256xf16>
// CHECK:           xegpu.load_nd {{.*}}sg_layout = [16, 1], sg_data = [8, 256], inst_data = [8, 32]{{.*}} -> vector<128x256xf16>
// CHECK-NEXT:      xegpu.dpas {{.*}}layout_a = #xegpu.layout<sg_layout = [16, 1], sg_data = [8, 256]>, layout_b = #xegpu.layout<sg_layout = [16, 1], sg_data = [256, 64]>, layout_cd = #xegpu.layout<sg_layout = [16, 1], sg_data = [8, 64]>
// CHECK:           xegpu.dpas {{.*}}layout_a = #xegpu.layout<sg_layout = [16, 1], sg_data = [8, 64]>, layout_b = #xegpu.layout<sg_layout = [16, 1], sg_data = [64, 256]>, layout_cd = #xegpu.layout<sg_layout = [16, 1], sg_data = [8, 256]>
// CHECK:         xegpu.store_nd {{.*}}sg_layout = [16, 1], sg_data = [8, 256]

#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0, s1] -> (128, s0 - s1 * 128)>
#map2 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * s1 + s2 * s3 + s4 * 32768)>
#map3 = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 * s3)>
#map4 = affine_map<()[s0] -> ((s0 floordiv 64) * 64, 0)>
#map5 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * s1 + s2 * s3 + s4 * 256)>
#map6 = affine_map<(d0)[s0] -> (-d0 + s0)>
gpu.module @sdpa_kernel [#xevm.target<O = 3>] {
  gpu.func @sdpa_kernel(%arg0: memref<?x16x?x256xf16>, %arg1: memref<?x16x?x256xf16>, %arg2: memref<?x16x?x256xf16>, %arg3: memref<?x16x?x256xf16>) kernel attributes {intel_reqd_sub_group_size = 16 : i32, known_block_size = array<i32: 256, 1, 1>} {
    %cst = arith.constant dense<0xFF800000> : vector<128x64xf32>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xindex>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<128x256xf32>
    %c64 = arith.constant 64 : index
    %cst_2 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_3 = arith.constant dense<0xFF800000> : vector<128xf32>
    %cst_4 = arith.constant dense<6.250000e-02> : vector<128x64xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %block_id_z = gpu.block_id z
    %dim = memref.dim %arg0, %c2 : memref<?x16x?x256xf16>
    %dim_6 = memref.dim %arg1, %c2 : memref<?x16x?x256xf16>
    %0 = affine.min #map1()[%dim, %block_id_z]
    %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %arg0 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
    %1 = affine.apply #map2()[%block_id_x, %strides#0, %block_id_y, %strides#1, %block_id_z]
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%1], sizes: [%0, 256], strides: [256, 1] : memref<f16> to memref<?x256xf16, strided<[256, 1], offset: ?>>
    %cast = memref.cast %reinterpret_cast : memref<?x256xf16, strided<[256, 1], offset: ?>> to memref<?x256xf16, strided<[?, 1], offset: ?>>
    %base_buffer_7, %offset_8, %sizes_9:4, %strides_10:4 = memref.extract_strided_metadata %arg1 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
    %2 = affine.apply #map3()[%block_id_x, %strides_10#0, %block_id_y, %strides_10#1]
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%2], sizes: [%dim_6, 256], strides: [256, 1] : memref<f16> to memref<?x256xf16, strided<[256, 1], offset: ?>>
    %dim_12 = memref.dim %reinterpret_cast_11, %c0 : memref<?x256xf16, strided<[256, 1], offset: ?>>
    %3 = affine.max #map4()[%dim_12]
    %4 = xegpu.create_nd_tdesc %cast : memref<?x256xf16, strided<[?, 1], offset: ?>> -> !xegpu.tensor_desc<128x256xf16>
    %5 = xegpu.load_nd %4[0, 0] : !xegpu.tensor_desc<128x256xf16> -> vector<128x256xf16>
    %6:3 = scf.for %arg4 = %c0 to %3 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_3, %arg7 = %cst_2) -> (vector<128x256xf32>, vector<128xf32>, vector<128xf32>) {
      %base_buffer_19, %offset_20, %sizes_21:4, %strides_22:4 = memref.extract_strided_metadata %arg1 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
      %14 = affine.apply #map5()[%block_id_x, %strides_22#0, %block_id_y, %strides_22#1, %arg4]
      %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [%14], sizes: [64, 256], strides: [256, 1] : memref<f16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
      %base_buffer_24, %offset_25, %sizes_26:4, %strides_27:4 = memref.extract_strided_metadata %arg2 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
      %15 = affine.apply #map5()[%block_id_x, %strides_27#0, %block_id_y, %strides_27#1, %arg4]
      %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [%15], sizes: [64, 256], strides: [256, 1] : memref<f16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
      %16 = xegpu.create_nd_tdesc %reinterpret_cast_23 : memref<64x256xf16, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<64x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %17 = xegpu.load_nd %16[0, 0] : !xegpu.tensor_desc<64x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x256xf16>
      %18 = xegpu.create_nd_tdesc %reinterpret_cast_28 : memref<64x256xf16, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<64x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %19 = xegpu.load_nd %18[0, 0] : !xegpu.tensor_desc<64x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x256xf16>
      %20 = vector.transpose %17, [1, 0] : vector<64x256xf16> to vector<256x64xf16>
      %21 = xegpu.dpas %5, %20, %cst_5 : vector<128x256xf16>, vector<256x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %22 = arith.mulf %21, %cst_4 : vector<128x64xf32>
      %23 = vector.multi_reduction <maximumf>, %22, %cst_3 [1] : vector<128x64xf32> to vector<128xf32>
      %24 = arith.maximumf %arg6, %23 : vector<128xf32>
      %25 = vector.broadcast %24 : vector<128xf32> to vector<64x128xf32>
      %26 = vector.transpose %25, [1, 0] : vector<64x128xf32> to vector<128x64xf32>
      %27 = arith.subf %22, %26 : vector<128x64xf32>
      %28 = math.exp %27 fastmath<fast> : vector<128x64xf32>
      %29 = vector.multi_reduction <add>, %28, %cst_2 [1] : vector<128x64xf32> to vector<128xf32>
      %30 = arith.subf %arg6, %24 : vector<128xf32>
      %31 = math.exp %30 fastmath<fast> : vector<128xf32>
      %32 = arith.mulf %arg7, %31 : vector<128xf32>
      %33 = arith.addf %32, %29 : vector<128xf32>
      %34 = vector.broadcast %31 : vector<128xf32> to vector<256x128xf32>
      %35 = vector.transpose %34, [1, 0] : vector<256x128xf32> to vector<128x256xf32>
      %36 = arith.mulf %arg5, %35 : vector<128x256xf32>
      %37 = arith.truncf %28 : vector<128x64xf32> to vector<128x64xf16>
      %38 = xegpu.dpas %37, %19, %36 : vector<128x64xf16>, vector<64x256xf16>, vector<128x256xf32> -> vector<128x256xf32>
      scf.yield %38, %24, %33 : vector<128x256xf32>, vector<128xf32>, vector<128xf32>
    }
    %7:3 = scf.for %arg4 = %3 to %dim_12 step %c64 iter_args(%arg5 = %6#0, %arg6 = %6#1, %arg7 = %6#2) -> (vector<128x256xf32>, vector<128xf32>, vector<128xf32>) {
      %14 = affine.apply #map6(%arg4)[%dim_12]
      %base_buffer_19, %offset_20, %sizes_21:4, %strides_22:4 = memref.extract_strided_metadata %arg1 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
      %15 = affine.apply #map5()[%block_id_x, %strides_22#0, %block_id_y, %strides_22#1, %arg4]
      %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [%15], sizes: [%14, 256], strides: [256, 1] : memref<f16> to memref<?x256xf16, strided<[256, 1], offset: ?>>
      %cast_24 = memref.cast %reinterpret_cast_23 : memref<?x256xf16, strided<[256, 1], offset: ?>> to memref<?x256xf16, strided<[?, 1], offset: ?>>
      %base_buffer_25, %offset_26, %sizes_27:4, %strides_28:4 = memref.extract_strided_metadata %arg2 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
      %16 = affine.apply #map5()[%block_id_x, %strides_28#0, %block_id_y, %strides_28#1, %arg4]
      %reinterpret_cast_29 = memref.reinterpret_cast %base_buffer_25 to offset: [%16], sizes: [%14, 256], strides: [256, 1] : memref<f16> to memref<?x256xf16, strided<[256, 1], offset: ?>>
      %cast_30 = memref.cast %reinterpret_cast_29 : memref<?x256xf16, strided<[256, 1], offset: ?>> to memref<?x256xf16, strided<[?, 1], offset: ?>>
      %dim_31 = memref.dim %reinterpret_cast_23, %c0 : memref<?x256xf16, strided<[256, 1], offset: ?>>
      %17 = xegpu.create_nd_tdesc %cast_24 : memref<?x256xf16, strided<[?, 1], offset: ?>> -> !xegpu.tensor_desc<64x256xf16>
      %18 = xegpu.load_nd %17[0, 0] : !xegpu.tensor_desc<64x256xf16> -> vector<64x256xf16>
      %19 = xegpu.create_nd_tdesc %cast_30 : memref<?x256xf16, strided<[?, 1], offset: ?>> -> !xegpu.tensor_desc<64x256xf16>
      %20 = xegpu.load_nd %19[0, 0] : !xegpu.tensor_desc<64x256xf16> -> vector<64x256xf16>
      %21 = vector.transpose %18, [1, 0] : vector<64x256xf16> to vector<256x64xf16>
      %22 = xegpu.dpas %5, %21, %cst_5 : vector<128x256xf16>, vector<256x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %23 = arith.mulf %22, %cst_4 : vector<128x64xf32>
      %24 = vector.broadcast %dim_31 : index to vector<64xindex>
      %25 = arith.cmpi ugt, %24, %cst_0 : vector<64xindex>
      %26 = vector.broadcast %25 : vector<64xi1> to vector<128x64xi1>
      %27 = arith.select %26, %23, %cst : vector<128x64xi1>, vector<128x64xf32>
      %28 = vector.multi_reduction <maximumf>, %27, %cst_3 [1] : vector<128x64xf32> to vector<128xf32>
      %29 = arith.maximumf %arg6, %28 : vector<128xf32>
      %30 = vector.broadcast %29 : vector<128xf32> to vector<64x128xf32>
      %31 = vector.transpose %30, [1, 0] : vector<64x128xf32> to vector<128x64xf32>
      %32 = arith.subf %23, %31 : vector<128x64xf32>
      %33 = math.exp %32 fastmath<fast> : vector<128x64xf32>
      %34 = arith.select %26, %33, %cst_5 : vector<128x64xi1>, vector<128x64xf32>
      %35 = vector.multi_reduction <add>, %34, %cst_2 [1] : vector<128x64xf32> to vector<128xf32>
      %36 = arith.subf %arg6, %29 : vector<128xf32>
      %37 = math.exp %36 fastmath<fast> : vector<128xf32>
      %38 = arith.mulf %arg7, %37 : vector<128xf32>
      %39 = arith.addf %38, %35 : vector<128xf32>
      %40 = vector.broadcast %37 : vector<128xf32> to vector<256x128xf32>
      %41 = vector.transpose %40, [1, 0] : vector<256x128xf32> to vector<128x256xf32>
      %42 = arith.mulf %arg5, %41 : vector<128x256xf32>
      %43 = arith.truncf %33 : vector<128x64xf32> to vector<128x64xf16>
      %44 = xegpu.dpas %43, %20, %42 : vector<128x64xf16>, vector<64x256xf16>, vector<128x256xf32> -> vector<128x256xf32>
      scf.yield %44, %29, %39 : vector<128x256xf32>, vector<128xf32>, vector<128xf32>
    }
    %8 = vector.broadcast %7#2 : vector<128xf32> to vector<256x128xf32>
    %9 = vector.transpose %8, [1, 0] : vector<256x128xf32> to vector<128x256xf32>
    %10 = arith.divf %7#0, %9 : vector<128x256xf32>
    %base_buffer_13, %offset_14, %sizes_15:4, %strides_16:4 = memref.extract_strided_metadata %arg3 : memref<?x16x?x256xf16> -> memref<f16>, index, index, index, index, index, index, index, index, index
    %11 = affine.apply #map2()[%block_id_x, %strides_16#0, %block_id_y, %strides_16#1, %block_id_z]
    %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [%11], sizes: [%0, 256], strides: [256, 1] : memref<f16> to memref<?x256xf16, strided<[256, 1], offset: ?>>
    %cast_18 = memref.cast %reinterpret_cast_17 : memref<?x256xf16, strided<[256, 1], offset: ?>> to memref<?x256xf16, strided<[?, 1], offset: ?>>
    %12 = arith.truncf %10 : vector<128x256xf32> to vector<128x256xf16>
    %13 = xegpu.create_nd_tdesc %cast_18 : memref<?x256xf16, strided<[?, 1], offset: ?>> -> !xegpu.tensor_desc<128x256xf16>
    xegpu.store_nd %12, %13[0, 0] : vector<128x256xf16>, !xegpu.tensor_desc<128x256xf16>
    gpu.return
  }
}
