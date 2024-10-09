// RUN: gc-opt %s --split-input-file --fold-tensor-operation --lower-to-tile-vector --CPU-physical-register-pass | FileCheck %s


// CHECK-DAG: #[[map0:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + s1)>
// CHECK-DAG: #[[map1:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0) -> (d0 * 128)>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG: #[[map5:.*]] = affine_map<(d0, d1) -> (d0 floordiv 32 + d1 floordiv 32)>
// CHECK-DAG: #[[map6:.*]] = affine_map<(d0, d1) -> (d0 floordiv 16 + d1 floordiv 16)>
// CHECK-DAG: #[[map7:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-DAG: #[[map8:.*]] = affine_map<()[s0, s1] -> (s0 * 16 + s1)>
// CHECK-DAG: #[[map9:.*]] = affine_map<(d0, d1) -> (d0 + d1)>



// CHECK-LABEL: func @add_tensor_test0
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
func.func @add_tensor_test0(%arg0: tensor<11008x4096xf32>, %arg1: tensor<11008x4096xf32>) -> tensor<11008x4096xf32> {
  %0 = tensor.empty() : tensor<11008x4096xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<11008x4096xf32>, tensor<11008x4096xf32>) outs(%0: tensor<11008x4096xf32>) -> tensor<11008x4096xf32>
  %2 = linalg.add ins(%1, %arg1 : tensor<11008x4096xf32>, tensor<11008x4096xf32>) outs(%0: tensor<11008x4096xf32>) -> tensor<11008x4096xf32>
  return %2 : tensor<11008x4096xf32>
}

// CHECK-LABEL: func @reduce_keepdimtest1
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16x64xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16x1x64xf32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<16x64xf32>, vector<16xf32>
// CHECK: scf.for
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ1]],  {{.*}} : vector<16xf32>
// CHECK: scf.yield
// CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<16x64xf32>
// CHECK: scf.yield
// CHECK: scf.for %[[arg1:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg2:.*]] = %[[EMPTY1]]) -> (tensor<16x1x64xf32>)
// CHECK: scf.for %[[arg3:.*]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[arg4:.*]] = %[[arg2]]) -> (tensor<16x1x64xf32>)
// CHECK: scf.for %[[arg5:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg6:.*]] = %[[arg4]]) -> (tensor<16x1x64xf32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map0]]()[%[[arg3]], %[[arg5]]]
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<16x64xf32>, vector<16xf32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<16x1x64xf32>
func.func @reduce_keepdimtest1(%arg0: tensor<16x32x64xf32>) -> tensor<16x1x64xf32> {
  %0 = tensor.empty() : tensor<16x64xf32>
  %reduce = linalg.reduce
      ins(%arg0:tensor<16x32x64xf32>)
      outs(%0:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %1 = arith.addf %out, %in: f32
        linalg.yield %1: f32
      }
  %2 = tensor.expand_shape %reduce [[0],[1, 2]] output_shape [16, 1, 64] : tensor<16x64xf32> into tensor<16x1x64xf32>
  return %2 : tensor<16x1x64xf32>
}

// CHECK-LABEL: func @fc_relu_test2
// CHECK: %[[MATMUL:.*]] = linalg.matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<512x512xf32>, vector<16xf32>
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<512x512xf32>, vector<16xf32>
// CHECK:  %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<16xf32>
// CHECK:  %[[ADD1:.*]] = arith.maximumf %[[ADD0]], {{.*}} : vector<16xf32>
// CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}}, {{.*}} : vector<16xf32>, tensor<512x512xf32>
func.func @fc_relu_test2(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {

  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

// CHECK-LABEL: func @matmul_add_test3
// CHECK: %[[MATMUL0:.*]] = linalg.matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32x32xf32>, vector<16xf32>
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32x32xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<16xf32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<32x32xf32>
// CHECK: scf.yield
// CHECK: scf.yield
func.func @matmul_add_test3(%arg0: tensor<8192x12288xf16>, %arg1: tensor<12288x16384xf16>, %arg2: tensor<8192x16384xf32>, %arg3: tensor<8192x16384xf32>) -> tensor<8192x16384xf32> {
  
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

// CHECK-LABEL: func @fuse_mlp_test4
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (d0 * 4)>
#map3 = affine_map<(d0) -> (d0 floordiv 16)>
#map4 = affine_map<(d0) -> (d0 floordiv 32)>
func.func @fuse_mlp_test4(%arg0: tensor<128x512xbf16>, %arg1: tensor<32x8x16x32xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
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
      %extracted_slice_2 = tensor.extract_slice %arg5[%3, %4] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%4] [128] [1] : tensor<256xbf16> to tensor<128xbf16>
      %extracted_slice_4 = tensor.extract_slice %0[%3, %4] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %extracted_slice_5 = tensor.extract_slice %arg6[%3, %4] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %extracted_slice_6 = tensor.extract_slice %arg7[%3, %4] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %6:3 = scf.for %arg8 = %c0 to %c64 step %c64 iter_args(%arg9 = %extracted_slice_2, %arg10 = %extracted_slice_5, %arg11 = %extracted_slice_6) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
        %7:3 = scf.for %arg12 = %c0 to %c128 step %c128 iter_args(%arg13 = %arg9, %arg14 = %arg10, %arg15 = %arg11) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
          %8:3 = scf.for %arg16 = %c0 to %c512 step %c512 iter_args(%arg17 = %arg13, %arg18 = %arg14, %arg19 = %arg15) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
            %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg8, %arg16] [64, 512] [1, 1] : tensor<64x512xbf16> to tensor<64x512xbf16>
            %9 = affine.apply #map3(%arg16)
            %10 = affine.apply #map4(%arg12)
            %extracted_slice_8 = tensor.extract_slice %extracted_slice_0[%9, %10, 0, 0] [32, 4, 16, 32] [1, 1, 1, 1] : tensor<32x4x16x32xbf16> to tensor<32x4x16x32xbf16>
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_1[%arg16, %arg12] [512, 128] [1, 1] : tensor<512x128xbf16> to tensor<512x128xbf16>
            %extracted_slice_10 = tensor.extract_slice %arg17[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %extracted_slice_11 = tensor.extract_slice %extracted_slice_3[%arg12] [128] [1] : tensor<128xbf16> to tensor<128xbf16>
            %extracted_slice_12 = tensor.extract_slice %extracted_slice_4[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %extracted_slice_13 = tensor.extract_slice %arg18[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %extracted_slice_14 = tensor.extract_slice %arg19[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %11:3 = scf.for %arg20 = %c0 to %c64 step %c32 iter_args(%arg21 = %extracted_slice_10, %arg22 = %extracted_slice_13, %arg23 = %extracted_slice_14) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
              %12:3 = scf.for %arg24 = %c0 to %c128 step %c32 iter_args(%arg25 = %arg21, %arg26 = %arg22, %arg27 = %arg23) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
                %13:3 = scf.for %arg28 = %c0 to %c512 step %c512 iter_args(%arg29 = %arg25, %arg30 = %arg26, %arg31 = %arg27) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
                  %extracted_slice_17 = tensor.extract_slice %extracted_slice_7[%arg20, %arg28] [32, 512] [1, 1] : tensor<64x512xbf16> to tensor<32x512xbf16>
                  %14 = affine.apply #map3(%arg28)
                  %15 = affine.apply #map4(%arg24)
                  %extracted_slice_18 = tensor.extract_slice %extracted_slice_8[%14, %15, 0, 0] [32, 1, 16, 32] [1, 1, 1, 1] : tensor<32x4x16x32xbf16> to tensor<32x1x16x32xbf16>
                  %extracted_slice_19 = tensor.extract_slice %extracted_slice_9[%arg28, %arg24] [512, 32] [1, 1] : tensor<512x128xbf16> to tensor<512x32xbf16>
                  %unpack = tensor.unpack %extracted_slice_18 inner_dims_pos = [0, 1] inner_tiles = [16, 32] into %extracted_slice_19 : tensor<32x1x16x32xbf16> -> tensor<512x32xbf16>
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C512:.*]] = arith.constant 512 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK: scf.forall
// CHECK-COUNT-6: scf.for
// CHECK-COUNT-4: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32x1x16x32xbf16>, vector<32xbf16>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<32xbf16>, tensor<32x16x1x32xbf16>
// CHECK: %[[FILL0:.*]] =  linalg.fill
// CHECK-COUNT-3: scf.for
// CHECK: %[[APPLY0:.*]] = affine.apply
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32x512xbf16>, vector<32xbf16> 
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<32xbf16>, tensor<1x32x512xbf16>
// CHECK-COUNT-4: scf.for
// CHECK: %[[APPLY1:.*]] = affine.apply
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32x16x1x32xbf16>, vector<32xbf16>
// CHECK: %[[WRITE2:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<32xbf16>, tensor<1x512x32xbf16>
// CHECK: %[[MATMUL0:.*]] = linalg.batch_reduce_matmul
// CHECK-COUNT-2: scf.for
// CHECK: %[[READ3:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32xbf16>, vector<32xbf16>
// CHECK: %[[READ4:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<32x32xbf16>, vector<32xbf16>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ4]], %[[READ3]] : vector<32xbf16> 
// CHECK: %[[EXP0:.*]] = math.exp %[[ADD0]] : vector<32xbf16>
// CHECK: %[[WRITE3:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<32xbf16>, tensor<32x32xbf16>
// CHECK: %[[WRITE4:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<32xbf16>, tensor<32x32xbf16>
                  %extracted_slice_20 = tensor.extract_slice %arg29[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %16 = linalg.fill ins(%cst : bf16) outs(%extracted_slice_20 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %expanded = tensor.expand_shape %extracted_slice_17 [[0, 1], [2]] output_shape [1, 32, 512] : tensor<32x512xbf16> into tensor<1x32x512xbf16>
                  %expanded_21 = tensor.expand_shape %unpack [[0, 1], [2]] output_shape [1, 32, 512] : tensor<512x32xbf16> into tensor<1x512x32xbf16>
                  %17 = linalg.batch_reduce_matmul ins(%expanded, %expanded_21 : tensor<1x32x512xbf16>, tensor<1x512x32xbf16>) outs(%16 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %extracted_slice_22 = tensor.extract_slice %extracted_slice_11[%arg24] [32] [1] : tensor<128xbf16> to tensor<32xbf16>
                  %extracted_slice_23 = tensor.extract_slice %extracted_slice_12[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %broadcasted = linalg.broadcast ins(%extracted_slice_22 : tensor<32xbf16>) outs(%extracted_slice_23 : tensor<32x32xbf16>) dimensions = [0] 
                  %extracted_slice_24 = tensor.extract_slice %arg30[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %18 = linalg.add ins(%17, %broadcasted : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%extracted_slice_24 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_25 = tensor.insert_slice %17 into %arg29[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  %extracted_slice_26 = tensor.extract_slice %arg31[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %19 = linalg.exp ins(%18 : tensor<32x32xbf16>) outs(%extracted_slice_26 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_27 = tensor.insert_slice %18 into %arg30[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  %inserted_slice_28 = tensor.insert_slice %19 into %arg31[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  scf.yield %inserted_slice_25, %inserted_slice_27, %inserted_slice_28 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
                }
                scf.yield %13#0, %13#1, %13#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
              }
              scf.yield %12#0, %12#1, %12#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
            }
            %inserted_slice = tensor.insert_slice %11#0 into %arg17[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            %inserted_slice_15 = tensor.insert_slice %11#1 into %arg18[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            %inserted_slice_16 = tensor.insert_slice %11#2 into %arg19[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            scf.yield %inserted_slice, %inserted_slice_15, %inserted_slice_16 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
          }
          scf.yield %8#0, %8#1, %8#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
        }
        scf.yield %7#0, %7#1, %7#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6#2 into %arg7[%3, %4] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
        tensor.parallel_insert_slice %6#1 into %arg6[%3, %4] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
        tensor.parallel_insert_slice %6#0 into %arg5[%3, %4] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
      }
    }
    return %2#2 : tensor<128x256xbf16>
  }

// CHECK-LABEL: func @elem_pack_transpose_inner_dims_test5
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C256:.*]] = arith.constant 256 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C32I32:.*]] = arith.constant 0 : i32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x32x16x16xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<4x16x16x32xi32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<128x256xi32>, vector<16xi32>
// CHECK: %[[ADD0:.*]] = arith.addi %[[READ0]], %[[READ0]] : vector<16xi32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<128x256xi32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY0]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C16]] step %[[C16]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<4x32x16x16xi32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map7]]()[%[[arg2]], %[[arg4]]]
// CHECK: %[[APPLY1:.*]] = affine.apply #[[map8]]()[%[[arg6]], %[[arg8]]]
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<128x256xi32>, vector<16xi32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<4x32x16x16xi32>
// CHECK-COUNT-4: scf.for
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<4x32x16x16xi32>, vector<1xi32>
// CHECK: %[[WRITE2:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<1xi32>, tensor<4x16x16x32xi32>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_inner_dims_test5(%arg0: tensor<128x256xi32>, %dest: tensor<4x16x16x32xi32>) -> tensor<4x16x16x32xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 32]
    into %dest : tensor<128x256xi32> -> tensor<4x16x16x32xi32>
  return %pack : tensor<4x16x16x32xi32>
}

// CHECK-LABEL: func @elem_pack_transpose_outer_dims_test6
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C256:.*]] = arith.constant 256 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C0I32:.*]] = arith.constant 0 : i32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x32x16x16xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16x4x32x16xi32>
// CHECK-COUNT-2: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<128x256xi32>, vector<16xi32>
// CHECK: %[[ADD0:.*]] = arith.addi %[[READ0]], %[[READ0]] : vector<16xi32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<128x256xi32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY0]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C16]] step %[[C16]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<4x32x16x16xi32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map7]]()[%[[arg2]], %[[arg4]]]
// CHECK: %[[APPLY1:.*]] = affine.apply #[[map8]]()[%[[arg6]], %[[arg8]]]
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<128x256xi32>, vector<16xi32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<4x32x16x16xi32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY2]]) -> (tensor<16x4x32x16xi32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<16x4x32x16xi32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<16x4x32x16xi32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C16]] step %[[C16]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<16x4x32x16xi32>)
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<4x32x16x16xi32>, vector<16xi32>
// CHECK: %[[WRITE2:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<16x4x32x16xi32>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_outer_dims_test6(%arg0: tensor<128x256xi32>, %dest: tensor<16x4x32x16xi32>) -> tensor<16x4x32x16xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 16]
    into %dest : tensor<128x256xi32> -> tensor<16x4x32x16xi32>
  return %pack : tensor<16x4x32x16xi32>
}

// CHECK-LABEL: func @elem_pack_transpose_inner_and_outer_dims_test7
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C256:.*]] = arith.constant 256 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C0I32:.*]] = arith.constant 0 : i32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x32x16x16xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16x4x16x32xi32>
// CHECK-COUNT-2: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<128x256xi32>, vector<16xi32>
// CHECK: %[[ADD0:.*]] = arith.addi %[[READ0]], %[[READ0]] : vector<16xi32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<128x256xi32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY0]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<4x32x16x16xi32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C16]] step %[[C16]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<4x32x16x16xi32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map7]]()[%[[arg2]], %[[arg4]]]
// CHECK: %[[APPLY1:.*]] = affine.apply #[[map8]]()[%[[arg6]], %[[arg8]]]
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<128x256xi32>, vector<16xi32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xi32>, tensor<4x32x16x16xi32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY2]]) -> (tensor<16x4x16x32xi32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<16x4x16x32xi32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<16x4x16x32xi32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<16x4x16x32xi32>)
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<4x32x16x16xi32>, vector<1xi32>
// CHECK: %[[WRITE2:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<1xi32>, tensor<16x4x16x32xi32>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_inner_and_outer_dims_test7(%arg0: tensor<128x256xi32>, %dest: tensor<16x4x16x32xi32>) -> tensor<16x4x16x32xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 32]
    into %dest : tensor<128x256xi32> -> tensor<16x4x16x32xi32>
  return %pack : tensor<16x4x16x32xi32>
}

// CHECK-LABEL: func @elem_pack_transpose_inner_and_outer_dims2_test8
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C57:.*]] = arith.constant 57 : index
// CHECK: %[[C56:.*]] = arith.constant 56 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1x56x57x2x32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1x56x57x64xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1x2x56x57x32xf32>
// CHECK-COUNT-4: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<64xf32>, vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<1x56x57x64xf32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY0]]) -> (tensor<1x56x57x2x32xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<1x56x57x2x32xf32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C57]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<1x56x57x2x32xf32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<1x56x57x2x32xf32>)
// CHECK: scf.for %[[arg10:.*]] = %[[C0]] to %[[C32]] step %[[C16]] iter_args(%[[arg11:.*]] = %[[arg9]]) -> (tensor<1x56x57x2x32xf32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map7]]()[%[[arg8]], %[[arg10]]]
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<1x56x57x64xf32>, vector<16xf32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<1x56x57x2x32xf32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY2]]) -> (tensor<1x2x56x57x32xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<1x2x56x57x32xf32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C57]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<1x2x56x57x32xf32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (tensor<1x2x56x57x32xf32>)
// CHECK: scf.for %[[arg10:.*]] = %[[C0]] to %[[C32]] step %[[C16]] iter_args(%[[arg11:.*]] = %[[arg9]]) -> (tensor<1x2x56x57x32xf32>)
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<1x56x57x2x32xf32>, vector<16xf32>
// CHECK: %[[WRITE2:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<1x2x56x57x32xf32>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @elem_pack_transpose_inner_and_outer_dims2_test8(%arg0: tensor<64xf32>, %dest: tensor<1x2x56x57x32xf32>) -> tensor<1x2x56x57x32xf32> {
  %0 = tensor.empty() : tensor<1x56x57x64xf32>
  %1 = linalg.generic {
      indexing_maps = [#map8, #map9],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<64xf32>)
    outs(%0 : tensor<1x56x57x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x57x64xf32>
  %2 = tensor.pack %1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %dest : tensor<1x56x57x64xf32> -> tensor<1x2x56x57x32xf32>
  return %2 : tensor<1x2x56x57x32xf32>
}


// CHECK-LABEL: func @broadcast_same_shape_test9
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<2x16xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<2x16xf32>
func.func @broadcast_same_shape_test9(%input: tensor<16xf32>, %init: tensor<2x16xf32>) -> tensor<2x16xf32> {
  %empty = tensor.empty() : tensor<2x16xf32>
  %0 = linalg.broadcast ins(%input: tensor<16xf32>) outs(%empty: tensor<2x16xf32>) dimensions = [0]
  %1 = linalg.add ins(%0, %init : tensor<2x16xf32>, tensor<2x16xf32>) outs(%init : tensor<2x16xf32>) -> tensor<2x16xf32>
  return %1 : tensor<2x16xf32>
}

// CHECK-LABEL: func @reduce_single_test10
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg3:.*]] = %arg1) -> (tensor<16x64xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<16x64xf32>)
// CHECK: %[[READ0:.*]] = vector.transfer_read %[[arg5]][%[[arg2]], %[[arg4]]], %[[CST]] {in_bounds = [true]} : tensor<16x64xf32>, vector<16xf32>
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[READ0]]) -> (vector<16xf32>)
// CHECK: %[[READ1:.*]] = vector.transfer_read %arg0[%[[arg2]], %[[arg6]], %[[arg4]]], {{.*}} {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ1]], %[[arg7]] : vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, %[[arg5]][%[[arg2]], %[[arg4]]] {in_bounds = [true]} : vector<16xf32>, tensor<16x64xf32>
func.func @reduce_single_test10(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}

// CHECK-LABEL: func @reduce_fusePostOp_test11
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16x32x64xf32>
// CHECK-COUNT-3: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}} {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ0]] : vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, {{.*}} {in_bounds = [true]} : vector<16xf32>, tensor<16x32x64xf32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg3:.*]] = %arg1) -> (tensor<16x64xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<16x64xf32>)
// CHECK: %[[READ1:.*]] = vector.transfer_read %[[arg5]][%[[arg2]], %[[arg4]]], %[[CST]] {in_bounds = [true]} : tensor<16x64xf32>, vector<16xf32>
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[READ1]]) -> (vector<16xf32>)
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}[%[[arg2]], %[[arg6]], %[[arg4]]], {{.*}} {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ2]], %[[arg7]] : vector<16xf32>
// CHECK: %[[MUL:.*]] = arith.mulf {{.*}}, {{.*}} : vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, %[[arg5]][%[[arg2]], %[[arg4]]] {in_bounds = [true]} : vector<16xf32>, tensor<16x64xf32>
func.func @reduce_fusePostOp_test11(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %0 = linalg.add ins(%input, %input : tensor<16x32x64xf32>,tensor<16x32x64xf32>)
       outs(%input : tensor<16x32x64xf32>) -> tensor<16x32x64xf32>
  %reduce = linalg.reduce
      ins(%0:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %2 = arith.addf %out, %in: f32
        linalg.yield %2: f32
      }
  %1 = linalg.mul ins(%reduce, %reduce : tensor<16x64xf32>, tensor<16x64xf32>) outs(%init: tensor<16x64xf32>) -> tensor<16x64xf32>
  func.return %1 : tensor<16x64xf32>
}

// CHECK-LABEL: func @reduce_fuse_test12
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg3:.*]] = %arg1) -> (tensor<16x32xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C16]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<16x32xf32>)
// CHECK: %[[READ0:.*]] = vector.transfer_read %[[arg5]][%[[arg2]], %[[arg4]]], %[[CST_0]] {in_bounds = [true]} : tensor<16x32xf32>, vector<16xf32> 
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[READ0]]) -> (vector<16xf32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg9:.*]] = %[[CST]]) -> (vector<16xf32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map9]](%[[arg4]], %[[arg6]])
// CHECK: %[[READ1:.*]] = vector.transfer_read %arg0[%[[arg2]], %[[APPLY0]], %[[arg8]]], {{.*}} {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ1]], %[[READ1]] : vector<16xf32>
// CHECK: %[[ADD1:.*]] = arith.addf %[[ADD0]], %[[arg9]] : vector<16xf32>
// CHECK: %[[REDUCTION:.*]] = vector.reduction <add>, {{.*}} : vector<16xf32> into f32
// CHECK: %[[INSERT:.*]] = vector.insert %[[REDUCTION]], %[[arg7]] [%[[arg6]]] : f32 into vector<16xf32>
// CHECK: %[[MUL:.*]] = arith.mulf {{.*}}, {{.*}} : vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write {{.*}}, %[[arg5]][%[[arg2]], %[[arg4]]] {in_bounds = [true]} : vector<16xf32>, tensor<16x32xf32>
func.func @reduce_fuse_test12(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.add ins(%input, %input : tensor<16x32x64xf32>,tensor<16x32x64xf32>)
       outs(%input : tensor<16x32x64xf32>) -> tensor<16x32x64xf32>
  %reduce = linalg.reduce
      ins(%0:tensor<16x32x64xf32>)
      outs(%init:tensor<16x32xf32>)
      dimensions = [2]
      (%in: f32, %out: f32) {
        %2 = arith.addf %out, %in: f32
        linalg.yield %2: f32
      }
  %1 = linalg.mul ins(%reduce, %reduce : tensor<16x32xf32>, tensor<16x32xf32>) outs(%init: tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %1 : tensor<16x32xf32>
}

// CHECK-LABEL: func @reduce_fuse_test13
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16x32x64xf32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg3:.*]] = %[[EMPTY0]]) -> (tensor<16x32x64xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<16x32x64xf32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg7:.*]] = %[[arg5]]) -> (tensor<16x32x64xf32>)
// CHECK: %[[READ0:.*]] = vector.transfer_read %{{.*}}[%[[arg2]], %[[arg4]], %[[arg6]]], %[[CST_0]] {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ0]] : vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write %[[ADD0]], %[[arg7]][%[[arg2]], %[[arg4]], %[[arg6]]] {in_bounds = [true]} : vector<16xf32>, tensor<16x32x64xf32>
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C16]] step %[[C16]] iter_args(%[[arg3:.*]] = %[[arg1]]) -> (tensor<16xf32>)
// CHECK: %[[READ1:.*]] = vector.transfer_read %[[arg3]][%[[arg2]]], %[[CST_0]] {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[arg5:.*]] = %[[READ1]]) -> (vector<16xf32>)
// CHECK: scf.for %[[arg6:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[arg7:.*]] = %[[CST]]) -> (vector<16xf32>)
// CHECK: scf.for %[[arg8:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg9:.*]] = %[[arg7]]) -> (vector<16xf32>)
// CHECK: %[[APPLY0:.*]] = affine.apply #[[map9]](%[[arg2]], %[[arg4]])
// CHECK: %[[READ2:.*]] = vector.transfer_read {{.*}}[%[[APPLY0]], %[[arg6]], %[[arg8]]], {{.*}} {in_bounds = [true]} : tensor<16x32x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ2]], %[[arg9]] : vector<16xf32>
// CHECK: %[[REDUCTION:.*]] = vector.reduction <add>, {{.*}} : vector<16xf32> into f32
// CHECK: %[[INSERT:.*]] = vector.insert %[[REDUCTION]], %[[arg5]] [%[[arg4]]] : f32 into vector<16xf32>
// CHECK: %[[MUL:.*]] = arith.mulf {{.*}}, {{.*}} : vector<16xf32>
// CHECK: %[[WRITE1:.*]] = vector.transfer_write {{.*}}, %[[arg3]][%[[arg2]]] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
func.func @reduce_fuse_test13(%input: tensor<16x32x64xf32>,
                  %init: tensor<16xf32>) -> tensor<16xf32> {
  %0 = linalg.add ins(%input, %input : tensor<16x32x64xf32>,tensor<16x32x64xf32>)
       outs(%input : tensor<16x32x64xf32>) -> tensor<16x32x64xf32>
  %reduce = linalg.reduce
      ins(%0:tensor<16x32x64xf32>)
      outs(%init:tensor<16xf32>)
      dimensions = [1, 2]
      (%in: f32, %out: f32) {
        %2 = arith.addf %out, %in: f32
        linalg.yield %2: f32
      }
  %1 = linalg.mul ins(%reduce, %reduce : tensor<16xf32>, tensor<16xf32>) outs(%init: tensor<16xf32>) -> tensor<16xf32>
  func.return %1 : tensor<16xf32>
}

// CHECK-LABEL: func @add_small_tensor_test14
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[TENSOR0:.*]] = tensor.empty() : tensor<2xf32>
// CHECK: scf.for
// CHECK: %[[READ0:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<2xf32>, vector<1xf32>
// CHECK: %[[READ1:.*]] = vector.transfer_read {{.*}}, {{.*}}: tensor<2xf32>, vector<1xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<1xf32>
// CHECK: %[[ADD1:.*]] = arith.maximumf %[[ADD0]], %[[CST]] : vector<1xf32>
// CHECK: %[[WRITE:.*]] = vector.transfer_write {{.*}}, {{.*}} : vector<1xf32>, tensor<2xf32>
func.func @add_small_tensor_test14(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tensor.empty() : tensor<2xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<2xf32>, tensor<2xf32>) outs(%0: tensor<2xf32>) -> tensor<2xf32>
  %2 = linalg.max ins(%1, %cst : tensor<2xf32>, tensor<2xf32>) outs(%0: tensor<2xf32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: func @broadcast_add_test15
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[arg2:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[arg3:.*]] = {{.*}}) -> (tensor<64x64xf32>)
// CHECK: scf.for %[[arg4:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg5:.*]] = %[[arg3]]) -> (tensor<64x64xf32>)
// CHECK: %[[READ0:.*]] = vector.transfer_read %arg0[%[[arg4]]], %[[CST]] {in_bounds = [true]} : tensor<64xf32>, vector<16xf32>
// CHECK: %[[READ1:.*]] = vector.transfer_read %[[arg5]][%[[arg2]], %[[arg4]]], %[[CST]] {in_bounds = [true]} : tensor<64x64xf32>, vector<16xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<16xf32>
// CHECK: %[[WRITE:.*]] = vector.transfer_write %[[ADD0]], %[[arg5]][%[[arg2]], %[[arg4]]] {in_bounds = [true]} : vector<16xf32>, tensor<64x64xf32>
func.func @broadcast_add_test15(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = tensor.empty() : tensor<64x64xf32>
  %bcast = linalg.broadcast
      ins(%arg0:tensor<64xf32>)
      outs(%0:tensor<64x64xf32>)
      dimensions = [0]
  %out3 = linalg.add ins(%bcast, %arg1: tensor<64x64xf32>, tensor<64x64xf32>) 
  outs(%arg1: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %out3: tensor<64x64xf32>
}

// CHECK-LABEL: func @broadcast_single_test16
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK: scf.for %[[arg1:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[arg2:.*]] = %[[EMPTY0]]) -> (tensor<64x64xf32>)
// CHECK: scf.for %[[arg3:.*]] = %[[C0]] to %[[C64]] step %[[C16]] iter_args(%[[arg4:.*]] = %[[arg2]]) -> (tensor<64x64xf32>)
// CHECK: %[[READ0:.*]] = vector.transfer_read %arg0[%[[arg3]]], %[[CST]] {in_bounds = [true]} : tensor<64xf32>, vector<16xf32>
// CHECK: %[[WRITE0:.*]] = vector.transfer_write %[[READ0]], %[[arg4]][%[[arg1]], %[[arg3]]] {in_bounds = [true]} : vector<16xf32>, tensor<64x64xf32> 
func.func @broadcast_single_test16(%arg0: tensor<64xf32>) -> tensor<64x64xf32> {
  %0 = tensor.empty() : tensor<64x64xf32>
  %bcast = linalg.broadcast
      ins(%arg0: tensor<64xf32>)
      outs(%0:tensor<64x64xf32>)
      dimensions = [0]
  return %bcast: tensor<64x64xf32>
}

