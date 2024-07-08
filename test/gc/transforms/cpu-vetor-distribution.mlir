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

