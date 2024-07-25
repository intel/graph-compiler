// RUN: gc-opt %s --split-input-file --lower-to-tile-vector --CPU-physical-register-pass | FileCheck %s

// CHECK-LABEL: func @add_tensor_test0
func.func @add_tensor_test0(%arg0: tensor<11008x4096xf32>, %arg1: tensor<11008x4096xf32>) -> tensor<11008x4096xf32> {
  // CHECK: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: %[[C11008:.*]] = arith.constant 11008 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[TENSOR0:.*]] = tensor.empty() : tensor<11008x4096xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<11008x4096xf32>, vector<16xf32>
  // CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<11008x4096xf32>, vector<16xf32>
  // CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<16xf32>
  // CHECK: %[[ADD1:.*]] = arith.addf %[[ADD0]], %[[READ1]] : vector<16xf32>
  // CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}}, {{.*}} : vector<16xf32>, tensor<11008x4096xf32>
  %0 = tensor.empty() : tensor<11008x4096xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<11008x4096xf32>, tensor<11008x4096xf32>) outs(%0: tensor<11008x4096xf32>) -> tensor<11008x4096xf32>
  %2 = linalg.add ins(%1, %arg1 : tensor<11008x4096xf32>, tensor<11008x4096xf32>) outs(%0: tensor<11008x4096xf32>) -> tensor<11008x4096xf32>
  return %2 : tensor<11008x4096xf32>
}

// CHECK-LABEL: func @fc_relu
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<512x512x512xf32>, vector<16xf32>
  // CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<512x512x512xf32>, vector<16xf32>
  // CHECK: %[[MULF0:.*]] = arith.mulf %[[READ0]], %[[READ1]] : vector<16xf32>
  // CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}}, {{.*}} : vector<16xf32>, tensor<512x512x512xf32>
  // CHECK-DAG: vector.multi_reduction
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<512x512xf32>, vector<16xf32>
  // CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<512x512xf32>, vector<16xf32>
  // CHECK:  %[[ADD0:.*]] = arith.addf %[[READ1]], %[[READ0]] : vector<16xf32>
  // CHECK:  %[[ADD1:.*]] = arith.maximumf %[[ADD0]], {{.*}} : vector<16xf32>
  // CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}}, {{.*}} : vector<16xf32>, tensor<512x512xf32>
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  // expected-remark @below {{elementwise binary}}
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

// CHECK-LABEL: func @matmul_add
func.func @matmul_add(%arg0: tensor<8192x12288xf16>, %arg1: tensor<12288x16384xf16>, %arg2: tensor<8192x16384xf32>, %arg3: tensor<8192x16384xf32>) -> tensor<8192x16384xf32> {
  // CHECK: vector.broadcast
  // CHECK: vector.transpose
  // CHECK: vector.broadcast
  // CHECK: vector.transpose
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<32x32x12288xf16>, vector<16xf16>
  // CHECK: %[[EXTF0:.*]] = arith.extf %[[READ0]] : vector<16xf16> to vector<16xf32>
  // CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} : tensor<32x32x12288xf16>, vector<16xf16>
  // CHECK: %[[EXTF1:.*]] = arith.extf %[[READ1]] : vector<16xf16> to vector<16xf32>
  // CHECK: %[[MULF0:.*]] = arith.mulf %[[EXTF0]], %[[EXTF1]] : vector<16xf32>
  // CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<32x32x12288xf32>
  // CHECK-DAG: vector.multi_reduction
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<8192x16384xf32>, vector<16xf32>
  // CHECK: %[[READ3:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<32x32xf32>, vector<16xf32>
  // CHECK: %[[ADD0:.*]] = arith.addf %[[READ3]], %[[READ2]] : vector<16xf32>
  // CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} : vector<16xf32>, tensor<128x128xf32>
  %0 = linalg.matmul {__fused_op__ = [0], name = "dot", tile_sizes = [[128, 128], [32, 32]]} ins(%arg0, %arg1 : tensor<8192x12288xf16>, tensor<12288x16384xf16>) outs(%arg2 : tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
  %1 = tensor.empty() : tensor<8192x16384xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %arg3 : tensor<8192x16384xf32>, tensor<8192x16384xf32>) outs(%1 : tensor<8192x16384xf32>) attrs =  {__root_op__ = 0 : i64} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    linalg.yield %4 : f32
  } -> tensor<8192x16384xf32>
  %c0 = arith.constant 0 : index
  %c8192 = arith.constant 8192 : index
  %c128 = arith.constant 128 : index
  %3 = scf.for %arg4 = %c0 to %c8192 step %c128 iter_args(%arg5 = %1) -> (tensor<8192x16384xf32>) {
    %c0_0 = arith.constant 0 : index
    %c16384 = arith.constant 16384 : index
    %c128_1 = arith.constant 128 : index
    %4 = scf.for %arg6 = %c0_0 to %c16384 step %c128_1 iter_args(%arg7 = %arg5) -> (tensor<8192x16384xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [128, 12288] [1, 1] : tensor<8192x12288xf16> to tensor<128x12288xf16>
      %extracted_slice_2 = tensor.extract_slice %arg1[0, %arg6] [12288, 128] [1, 1] : tensor<12288x16384xf16> to tensor<12288x128xf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, %arg6] [128, 128] [1, 1] : tensor<8192x16384xf32> to tensor<128x128xf32>
      %5 = linalg.matmul {__fused_op__ = [0], name = "dot", tile_sizes = [[128, 128], [32, 32]]} ins(%extracted_slice, %extracted_slice_2 : tensor<128x12288xf16>, tensor<12288x128xf16>) outs(%extracted_slice_3 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %extracted_slice_4 = tensor.extract_slice %0[%arg4, %arg6] [128, 128] [1, 1] : tensor<8192x16384xf32> to tensor<128x128xf32>
      %extracted_slice_5 = tensor.extract_slice %arg3[%arg4, %arg6] [128, 128] [1, 1] : tensor<8192x16384xf32> to tensor<128x128xf32>
      %extracted_slice_6 = tensor.extract_slice %arg7[%arg4, %arg6] [128, 128] [1, 1] : tensor<8192x16384xf32> to tensor<128x128xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %extracted_slice_5 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%extracted_slice_6 : tensor<128x128xf32>) attrs =  {__root_op__ = 0 : i64} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %8 = arith.addf %in, %in_9 : f32
        linalg.yield %8 : f32
      } -> tensor<128x128xf32>
      %c0_7 = arith.constant 0 : index
      %c128_8 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %7 = scf.for %arg8 = %c0_7 to %c128_8 step %c32 iter_args(%arg9 = %extracted_slice_6) -> (tensor<128x128xf32>) {
        %c0_9 = arith.constant 0 : index
        %c128_10 = arith.constant 128 : index
        %c32_11 = arith.constant 32 : index
        %8 = scf.for %arg10 = %c0_9 to %c128_10 step %c32_11 iter_args(%arg11 = %arg9) -> (tensor<128x128xf32>) {
          %extracted_slice_12 = tensor.extract_slice %extracted_slice[%arg8, 0] [32, 12288] [1, 1] : tensor<128x12288xf16> to tensor<32x12288xf16>
          %extracted_slice_13 = tensor.extract_slice %extracted_slice_2[0, %arg10] [12288, 32] [1, 1] : tensor<12288x128xf16> to tensor<12288x32xf16>
          %extracted_slice_14 = tensor.extract_slice %extracted_slice_3[%arg8, %arg10] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %9 = linalg.matmul {__fused_op__ = [0], name = "dot", tile_sizes = [[128, 128], [32, 32]]} ins(%extracted_slice_12, %extracted_slice_13 : tensor<32x12288xf16>, tensor<12288x32xf16>) outs(%extracted_slice_14 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %extracted_slice_15 = tensor.extract_slice %5[%arg8, %arg10] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %extracted_slice_16 = tensor.extract_slice %extracted_slice_5[%arg8, %arg10] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %extracted_slice_17 = tensor.extract_slice %arg11[%arg8, %arg10] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
          %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %extracted_slice_16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_17 : tensor<32x32xf32>) attrs =  {__root_op__ = 0 : i64} {
          ^bb0(%in: f32, %in_19: f32, %out: f32):
            %11 = arith.addf %in, %in_19 : f32
            linalg.yield %11 : f32
          } -> tensor<32x32xf32>
          %inserted_slice_18 = tensor.insert_slice %10 into %arg11[%arg8, %arg10] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice_18 : tensor<128x128xf32>
        } {__parallel_loop__ = 1 : i64}
        scf.yield %8 : tensor<128x128xf32>
      } {__parallel_loop__ = 1 : i64}
      %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<8192x16384xf32>
      scf.yield %inserted_slice : tensor<8192x16384xf32>
    } {__parallel_loop__ = 0 : i64}
    scf.yield %4 : tensor<8192x16384xf32>
  } {__parallel_loop__ = 0 : i64}
  return %3 : tensor<8192x16384xf32>
}

#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (d0 * 4)>
#map3 = affine_map<(d0) -> (d0 floordiv 16)>
#map4 = affine_map<(d0) -> (d0 floordiv 32)>
  func.func @mlp(%arg0: tensor<128x512xbf16>, %arg1: tensor<32x8x16x32xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x256xbf16>
    %1 = tensor.empty() : tensor<512x256xbf16>
    %2:3 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %0, %arg6 = %0, %arg7 = %0) -> (tensor<128x256xbf16>, tensor<128x256xbf16>, tensor<128x256xbf16>) {
      %3 = affine.apply #map(%arg3)
      %4 = affine.apply #map1(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [64, 512] [1, 1] : tensor<128x512xbf16> to tensor<64x512xbf16>
      %5 = affine.apply #map2(%arg4)
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %5, 0, 0] [32, 4, 16, 32] [1, 1, 1, 1] : tensor<32x8x16x32xbf16> to tensor<32x4x16x32xbf16>
      %extracted_slice_1 = tensor.extract_slice %1[0, %4] [512, 128] [1, 1] : tensor<512x256xbf16> to tensor<512x128xbf16>
      %6 = affine.apply #map(%arg3)
      %7 = affine.apply #map1(%arg4)
      %extracted_slice_2 = tensor.extract_slice %arg5[%6, %7] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %8 = affine.apply #map(%arg3)
      %9 = affine.apply #map1(%arg4)
      %10 = affine.apply #map(%arg3)
      %11 = affine.apply #map1(%arg4)
      %12 = affine.apply #map1(%arg4)
      %13 = affine.apply #map(%arg3)
      %14 = affine.apply #map1(%arg4)
      %extracted_slice_3 = tensor.extract_slice %arg2[%12] [128] [1] : tensor<256xbf16> to tensor<128xbf16>
      %extracted_slice_4 = tensor.extract_slice %0[%13, %14] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %extracted_slice_5 = tensor.extract_slice %arg6[%10, %11] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %15 = affine.apply #map(%arg3)
      %16 = affine.apply #map1(%arg4)
      %17 = affine.apply #map(%arg3)
      %18 = affine.apply #map1(%arg4)
      %extracted_slice_6 = tensor.extract_slice %arg7[%17, %18] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %19:3 = scf.for %arg8 = %c0 to %c64 step %c64 iter_args(%arg9 = %extracted_slice_2, %arg10 = %extracted_slice_5, %arg11 = %extracted_slice_6) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
        %22:3 = scf.for %arg12 = %c0 to %c128 step %c128 iter_args(%arg13 = %arg9, %arg14 = %arg10, %arg15 = %arg11) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
          %23:3 = scf.for %arg16 = %c0 to %c512 step %c512 iter_args(%arg17 = %arg13, %arg18 = %arg14, %arg19 = %arg15) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
            %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg8, %arg16] [64, 512] [1, 1] : tensor<64x512xbf16> to tensor<64x512xbf16>
            %24 = affine.apply #map3(%arg16)
            %25 = affine.apply #map4(%arg12)
            %extracted_slice_8 = tensor.extract_slice %extracted_slice_0[%24, %25, 0, 0] [32, 4, 16, 32] [1, 1, 1, 1] : tensor<32x4x16x32xbf16> to tensor<32x4x16x32xbf16>
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_1[%arg16, %arg12] [512, 128] [1, 1] : tensor<512x128xbf16> to tensor<512x128xbf16>
            %extracted_slice_10 = tensor.extract_slice %arg17[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %extracted_slice_11 = tensor.extract_slice %extracted_slice_3[%arg12] [128] [1] : tensor<128xbf16> to tensor<128xbf16>
            %extracted_slice_12 = tensor.extract_slice %extracted_slice_4[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %extracted_slice_13 = tensor.extract_slice %arg18[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %extracted_slice_14 = tensor.extract_slice %arg19[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %26:3 = scf.for %arg20 = %c0 to %c64 step %c32 iter_args(%arg21 = %extracted_slice_10, %arg22 = %extracted_slice_13, %arg23 = %extracted_slice_14) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
              %27:3 = scf.for %arg24 = %c0 to %c128 step %c32 iter_args(%arg25 = %arg21, %arg26 = %arg22, %arg27 = %arg23) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
                %28:3 = scf.for %arg28 = %c0 to %c512 step %c512 iter_args(%arg29 = %arg25, %arg30 = %arg26, %arg31 = %arg27) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
                  %extracted_slice_17 = tensor.extract_slice %extracted_slice_7[%arg20, %arg28] [32, 512] [1, 1] : tensor<64x512xbf16> to tensor<32x512xbf16>
                  %29 = affine.apply #map3(%arg28)
                  %30 = affine.apply #map4(%arg24)
                  %extracted_slice_18 = tensor.extract_slice %extracted_slice_8[%29, %30, 0, 0] [32, 1, 16, 32] [1, 1, 1, 1] : tensor<32x4x16x32xbf16> to tensor<32x1x16x32xbf16>
                  %extracted_slice_19 = tensor.extract_slice %extracted_slice_9[%arg28, %arg24] [512, 32] [1, 1] : tensor<512x128xbf16> to tensor<512x32xbf16>
                  %unpack = tensor.unpack %extracted_slice_18 inner_dims_pos = [0, 1] inner_tiles = [16, 32] into %extracted_slice_19 : tensor<32x1x16x32xbf16> -> tensor<512x32xbf16>
                  %extracted_slice_20 = tensor.extract_slice %arg29[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %31 = linalg.fill ins(%cst : bf16) outs(%extracted_slice_20 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %expanded = tensor.expand_shape %extracted_slice_17 [[0, 1], [2]] output_shape [1, 32, 512] : tensor<32x512xbf16> into tensor<1x32x512xbf16>
                  %expanded_21 = tensor.expand_shape %unpack [[0, 1], [2]] output_shape [1, 32, 512] : tensor<512x32xbf16> into tensor<1x512x32xbf16>
                  %32 = linalg.batch_reduce_matmul ins(%expanded, %expanded_21 : tensor<1x32x512xbf16>, tensor<1x512x32xbf16>) outs(%31 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %extracted_slice_22 = tensor.extract_slice %extracted_slice_11[%arg24] [32] [1] : tensor<128xbf16> to tensor<32xbf16>
                  %extracted_slice_23 = tensor.extract_slice %extracted_slice_12[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %broadcasted = linalg.broadcast ins(%extracted_slice_22 : tensor<32xbf16>) outs(%extracted_slice_23 : tensor<32x32xbf16>) dimensions = [0] 
                  %extracted_slice_24 = tensor.extract_slice %arg30[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %33 = linalg.add ins(%32, %broadcasted : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%extracted_slice_24 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_25 = tensor.insert_slice %32 into %arg29[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  %extracted_slice_26 = tensor.extract_slice %arg31[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %34 = linalg.exp ins(%33 : tensor<32x32xbf16>) outs(%extracted_slice_26 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_27 = tensor.insert_slice %33 into %arg30[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  %inserted_slice_28 = tensor.insert_slice %34 into %arg31[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  scf.yield %inserted_slice_25, %inserted_slice_27, %inserted_slice_28 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
                }
                scf.yield %28#0, %28#1, %28#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
              }
              scf.yield %27#0, %27#1, %27#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
            }
            %inserted_slice = tensor.insert_slice %26#0 into %arg17[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            %inserted_slice_15 = tensor.insert_slice %26#1 into %arg18[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            %inserted_slice_16 = tensor.insert_slice %26#2 into %arg19[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            scf.yield %inserted_slice, %inserted_slice_15, %inserted_slice_16 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
          }
          scf.yield %23#0, %23#1, %23#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
        }
        scf.yield %22#0, %22#1, %22#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
      }
      %20 = affine.apply #map(%arg3)
      %21 = affine.apply #map1(%arg4)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %19#2 into %arg7[%20, %21] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
        tensor.parallel_insert_slice %19#1 into %arg6[%15, %16] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
        tensor.parallel_insert_slice %19#0 into %arg5[%8, %9] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
      }
    }
    return %2#2 : tensor<128x256xbf16>
  }



  func.func @main_entry(%arg0: tensor<128x128x64x64xbf16>, %arg1: tensor<128x128x32x64x2xbf16>) -> tensor<128x128x64x64xbf16> attributes {llvm.emit_c_interface} {
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x128x64x64xbf16>
    %1 = tensor.empty() : tensor<128x128x64x64xf32>
    %2 = tensor.empty() : tensor<2x1x1x128x128x64x64xf32>
    %3 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %2) -> (tensor<2x1x1x128x128x64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg3[%arg2, 0, 0, 0, 0, 0, 0] [1, 1, 1, 128, 128, 64, 64] [1, 1, 1, 1, 1, 1, 1] : tensor<2x1x1x128x128x64x64xf32> to tensor<128x128x64x64xf32>
      %5 = scf.forall (%arg4) in (7) shared_outs(%arg5 = %extracted_slice) -> (tensor<128x128x64x64xf32>) {
        %6 = affine.min affine_map<(d0) -> (d0 * -19 + 128, 19)>(%arg4)
        %7 = affine.max affine_map<(d0) -> (0, d0)>(%6)
        %8 = affine.apply affine_map<(d0) -> (d0 * 19)>(%arg4)
        %extracted_slice_0 = tensor.extract_slice %arg5[%8, 0, 0, 0] [%7, 128, 64, 64] [1, 1, 1, 1] : tensor<128x128x64x64xf32> to tensor<?x128x64x64xf32>
        %9 = scf.forall (%arg6) in (4) shared_outs(%arg7 = %extracted_slice_0) -> (tensor<?x128x64x64xf32>) {
          %11 = affine.max affine_map<(d0) -> (0, d0)>(%6)
          %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg6)
          %extracted_slice_1 = tensor.extract_slice %arg7[0, %12, 0, 0] [%11, 32, 64, 64] [1, 1, 1, 1] : tensor<?x128x64x64xf32> to tensor<?x32x64x64xf32>
          %13 = scf.for %arg8 = %c0 to %11 step %c4 iter_args(%arg9 = %extracted_slice_1) -> (tensor<?x32x64x64xf32>) {
            %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%arg8)[%11]
            %extracted_slice_2 = tensor.extract_slice %arg9[%arg8, 0, 0, 0] [%15, 32, 64, 64] [1, 1, 1, 1] : tensor<?x32x64x64xf32> to tensor<?x32x64x64xf32>
            %16 = scf.for %arg10 = %c0 to %c32 step %c4 iter_args(%arg11 = %extracted_slice_2) -> (tensor<?x32x64x64xf32>) {
              %extracted_slice_3 = tensor.extract_slice %arg11[0, %arg10, 0, 0] [%15, 4, 64, 64] [1, 1, 1, 1] : tensor<?x32x64x64xf32> to tensor<?x4x64x64xf32>
              %17 = scf.for %arg12 = %c0 to %c64 step %c16 iter_args(%arg13 = %extracted_slice_3) -> (tensor<?x4x64x64xf32>) {
                %extracted_slice_5 = tensor.extract_slice %arg13[0, 0, 0, 0] [%15, 4, 64, 64] [1, 1, 1, 1] : tensor<?x4x64x64xf32> to tensor<?x4x64x64xf32>
                %18 = scf.for %arg14 = %c0 to %15 step %c1 iter_args(%arg15 = %extracted_slice_5) -> (tensor<?x4x64x64xf32>) {
                  %extracted_slice_7 = tensor.extract_slice %arg15[%arg14, 0, 0, 0] [1, 4, 64, 64] [1, 1, 1, 1] : tensor<?x4x64x64xf32> to tensor<1x4x64x64xf32>
                  %19 = scf.for %arg16 = %c0 to %c4 step %c1 iter_args(%arg17 = %extracted_slice_7) -> (tensor<1x4x64x64xf32>) {
                    %20 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * 19 + s0 + s1)>(%arg4)[%arg14, %arg8]
                    %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg2)[%arg12]
                    %extracted_slice_9 = tensor.extract_slice %arg0[%20, %21, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : tensor<128x128x64x64xbf16> to tensor<16x64x64xbf16>
                    %22 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * 32 + s0 + s1)>(%arg6)[%arg16, %arg10]
                    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg2)[%arg12]
                    %extracted_slice_10 = tensor.extract_slice %arg1[%22, %23, 0, 0, 0] [1, 16, 32, 64, 2] [1, 1, 1, 1, 1] : tensor<128x128x32x64x2xbf16> to tensor<16x32x64x2xbf16>
                    %extracted_slice_11 = tensor.extract_slice %arg17[0, %arg16, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<1x4x64x64xf32> to tensor<64x64xf32>
                    %24 = arith.cmpi eq, %arg12, %c0 : index
                    %25 = scf.if %24 -> (tensor<64x64xf32>) {
                      %26 = linalg.fill ins(%cst : bf16) outs(%extracted_slice_11 : tensor<64x64xf32>) -> tensor<64x64xf32>
                      %27 = linalgx.batch_reduce_matmul_vnni ins(%extracted_slice_9, %extracted_slice_10 : tensor<16x64x64xbf16>, tensor<16x32x64x2xbf16>) outs(%26 : tensor<64x64xf32>) -> tensor<64x64xf32>
                      scf.yield %27 : tensor<64x64xf32>
                    } else {
                      %26 = linalgx.batch_reduce_matmul_vnni ins(%extracted_slice_9, %extracted_slice_10 : tensor<16x64x64xbf16>, tensor<16x32x64x2xbf16>) outs(%extracted_slice_11 : tensor<64x64xf32>) -> tensor<64x64xf32>
                      scf.yield %26 : tensor<64x64xf32>
                    }
                    %inserted_slice_12 = tensor.insert_slice %25 into %arg17[0, %arg16, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<64x64xf32> into tensor<1x4x64x64xf32>
                    scf.yield %inserted_slice_12 : tensor<1x4x64x64xf32>
                  }
                  %inserted_slice_8 = tensor.insert_slice %19 into %arg15[%arg14, 0, 0, 0] [1, 4, 64, 64] [1, 1, 1, 1] : tensor<1x4x64x64xf32> into tensor<?x4x64x64xf32>
                  scf.yield %inserted_slice_8 : tensor<?x4x64x64xf32>
                }
                %inserted_slice_6 = tensor.insert_slice %18 into %arg13[0, 0, 0, 0] [%15, 4, 64, 64] [1, 1, 1, 1] : tensor<?x4x64x64xf32> into tensor<?x4x64x64xf32>
                scf.yield %inserted_slice_6 : tensor<?x4x64x64xf32>
              }
              %inserted_slice_4 = tensor.insert_slice %17 into %arg11[0, %arg10, 0, 0] [%15, 4, 64, 64] [1, 1, 1, 1] : tensor<?x4x64x64xf32> into tensor<?x32x64x64xf32>
              scf.yield %inserted_slice_4 : tensor<?x32x64x64xf32>
            }
            %inserted_slice = tensor.insert_slice %16 into %arg9[%arg8, 0, 0, 0] [%15, 32, 64, 64] [1, 1, 1, 1] : tensor<?x32x64x64xf32> into tensor<?x32x64x64xf32>
            scf.yield %inserted_slice : tensor<?x32x64x64xf32>
          }
          %14 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg6)
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %13 into %arg7[0, %14, 0, 0] [%11, 32, 64, 64] [1, 1, 1, 1] : tensor<?x32x64x64xf32> into tensor<?x128x64x64xf32>
          }
        }
        %10 = affine.apply affine_map<(d0) -> (d0 * 19)>(%arg4)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %9 into %arg5[%10, 0, 0, 0] [%7, 128, 64, 64] [1, 1, 1, 1] : tensor<?x128x64x64xf32> into tensor<128x128x64x64xf32>
        }
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg3[%arg2, 0, 0, 0, 0, 0, 0] [1, 1, 1, 128, 128, 64, 64] [1, 1, 1, 1, 1, 1, 1] : tensor<128x128x64x64xf32> into tensor<2x1x1x128x128x64x64xf32>
      }
    }
    %4 = scf.forall (%arg2) in (128) shared_outs(%arg3 = %0) -> (tensor<128x128x64x64xbf16>) {
      %extracted_slice = tensor.extract_slice %1[%arg2, 0, 0, 0] [1, 128, 64, 64] [1, 1, 1, 1] : tensor<128x128x64x64xf32> to tensor<1x128x64x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, 0, 0, 0] [1, 128, 64, 64] [1, 1, 1, 1] : tensor<128x128x64x64xbf16> to tensor<1x128x64x64xbf16>
      %5:2 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %extracted_slice, %arg6 = %extracted_slice_0) -> (tensor<1x128x64x64xf32>, tensor<1x128x64x64xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %arg5[0, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<1x128x64x64xf32> to tensor<1x1x64x64xf32>
        %extracted_slice_2 = tensor.extract_slice %arg6[0, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<1x128x64x64xbf16> to tensor<1x1x64x64xbf16>
        %6:2 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %extracted_slice_1, %arg9 = %extracted_slice_2) -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>) {
          %extracted_slice_4 = tensor.extract_slice %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x64x64xf32> to tensor<1x1x1x64xf32>
          %extracted_slice_5 = tensor.extract_slice %arg9[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x64x64xbf16> to tensor<1x1x1x64xbf16>
          %7:2 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %extracted_slice_4, %arg12 = %extracted_slice_5) -> (tensor<1x1x1x64xf32>, tensor<1x1x1x64xbf16>) {
            %extracted_slice_8 = tensor.extract_slice %arg11[0, 0, 0, %arg10] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<1x1x1x64xf32> to tensor<1x1x1x32xf32>
            %8 = linalg.fill ins(%cst : bf16) outs(%extracted_slice_8 : tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
            %9 = scf.for %arg13 = %c0 to %c2 step %c1 iter_args(%arg14 = %8) -> (tensor<1x1x1x32xf32>) {
              %extracted_slice_12 = tensor.extract_slice %3[%arg13, 0, 0, %arg2, %arg4, %arg7, %arg10] [1, 1, 1, 1, 1, 1, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<2x1x1x128x128x64x64xf32> to tensor<1x1x1x1x1x1x32xf32>
              %reduced = linalg.reduce ins(%extracted_slice_12 : tensor<1x1x1x1x1x1x32xf32>) outs(%arg14 : tensor<1x1x1x32xf32>) dimensions = [0, 1, 2]
                (%in: f32, %init: f32) {
                  %11 = arith.addf %in, %init : f32
                  linalg.yield %11 : f32
                }
              scf.yield %reduced : tensor<1x1x1x32xf32>
            }
            %extracted_slice_9 = tensor.extract_slice %arg12[0, 0, 0, %arg10] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<1x1x1x64xbf16> to tensor<1x1x1x32xbf16>
            %10 = linalg.copy ins(%9 : tensor<1x1x1x32xf32>) outs(%extracted_slice_9 : tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
            %inserted_slice_10 = tensor.insert_slice %9 into %arg11[0, 0, 0, %arg10] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<1x1x1x32xf32> into tensor<1x1x1x64xf32>
            %inserted_slice_11 = tensor.insert_slice %10 into %arg12[0, 0, 0, %arg10] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<1x1x1x32xbf16> into tensor<1x1x1x64xbf16>
            scf.yield %inserted_slice_10, %inserted_slice_11 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xbf16>
          }
          %inserted_slice_6 = tensor.insert_slice %7#0 into %arg8[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<1x1x64x64xf32>
          %inserted_slice_7 = tensor.insert_slice %7#1 into %arg9[0, 0, %arg7, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xbf16> into tensor<1x1x64x64xbf16>
          scf.yield %inserted_slice_6, %inserted_slice_7 : tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>
        }
        %inserted_slice = tensor.insert_slice %6#0 into %arg5[0, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<1x1x64x64xf32> into tensor<1x128x64x64xf32>
        %inserted_slice_3 = tensor.insert_slice %6#1 into %arg6[0, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<1x1x64x64xbf16> into tensor<1x128x64x64xbf16>
        scf.yield %inserted_slice, %inserted_slice_3 : tensor<1x128x64x64xf32>, tensor<1x128x64x64xbf16>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5#1 into %arg3[%arg2, 0, 0, 0] [1, 128, 64, 64] [1, 1, 1, 1] : tensor<1x128x64x64xbf16> into tensor<128x128x64x64xbf16>
      }
    }
    return %4 : tensor<128x128x64x64xbf16>
  }

