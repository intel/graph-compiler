// RUN: gc-opt --split-input-file --lower-to-tile-vector --CPU-physical-register-pass --mlir-print-ir-after-all -- %s

// CHECK-LABEL: func @add_tensor
func.func @add_tensor_test0(%arg0: tensor<11008x4096xf32>, %arg1: tensor<11008x4096xf32>) -> tensor<11008x4096xf32> {
  %0 = tensor.empty() : tensor<11008x4096xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<11008x4096xf32>, tensor<11008x4096xf32>) outs(%0: tensor<11008x4096xf32>) -> tensor<11008x4096xf32>
  %2 = linalg.add ins(%1, %arg1 : tensor<11008x4096xf32>, tensor<11008x4096xf32>) outs(%0: tensor<11008x4096xf32>) -> tensor<11008x4096xf32>
  return %2 : tensor<11008x4096xf32>
}

func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
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

func.func @reduce_keepdim0(%arg0: tensor<16x32x64xf32>) -> tensor<16x1x64xf32> {
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

#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (d0 floordiv 16)>
#map3 = affine_map<(d0) -> (d0 floordiv 32)>
#map4 = affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 128)>
#map5 = affine_map<(d0, d1, d2) -> (d0 + d1 + d2 * 64)>
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
      %extracted_slice_0 = tensor.extract_slice %arg5[%3, %4] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %5:3 = scf.for %arg8 = %c0 to %c64 step %c64 iter_args(%arg9 = %extracted_slice_0, %arg10 = %extracted_slice_0, %arg11 = %extracted_slice_0) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
        %6:3 = scf.for %arg12 = %c0 to %c128 step %c128 iter_args(%arg13 = %arg9, %arg14 = %arg10, %arg15 = %arg11) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
          %7:3 = scf.for %arg16 = %c0 to %c512 step %c512 iter_args(%arg17 = %arg13, %arg18 = %arg14, %arg19 = %arg15) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
            %extracted_slice_1 = tensor.extract_slice %extracted_slice[%arg8, %arg16] [64, 512] [1, 1] : tensor<64x512xbf16> to tensor<64x512xbf16>
            %extracted_slice_2 = tensor.extract_slice %arg17[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %8:3 = scf.for %arg20 = %c0 to %c64 step %c32 iter_args(%arg21 = %extracted_slice_2, %arg22 = %arg18, %arg23 = %arg19) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
              %9:3 = scf.for %arg24 = %c0 to %c128 step %c32 iter_args(%arg25 = %arg21, %arg26 = %arg22, %arg27 = %arg23) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
                %10:3 = scf.for %arg28 = %c0 to %c512 step %c512 iter_args(%arg29 = %arg25, %arg30 = %arg26, %arg31 = %arg27) -> (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) {
                  %extracted_slice_5 = tensor.extract_slice %extracted_slice_1[%arg20, %arg28] [32, 512] [1, 1] : tensor<64x512xbf16> to tensor<32x512xbf16>
                  %11 = affine.apply #map2(%arg28)
                  %12 = affine.apply #map3(%arg24)
                  %extracted_slice_6 = tensor.extract_slice %arg1[%11, %12, 0, 0] [32, 1, 16, 32] [1, 1, 1, 1] : tensor<32x8x16x32xbf16> to tensor<32x1x16x32xbf16>
                  %extracted_slice_7 = tensor.extract_slice %1[%arg28, %arg24] [512, 32] [1, 1] : tensor<512x256xbf16> to tensor<512x32xbf16>
                  %unpack = tensor.unpack %extracted_slice_6 inner_dims_pos = [0, 1] inner_tiles = [16, 32] into %extracted_slice_7 : tensor<32x1x16x32xbf16> -> tensor<512x32xbf16>
                  %extracted_slice_8 = tensor.extract_slice %arg29[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %13 = linalg.fill ins(%cst : bf16) outs(%extracted_slice_8 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %expanded = tensor.expand_shape %extracted_slice_5 [[0, 1], [2]] output_shape [1, 32, 512] : tensor<32x512xbf16> into tensor<1x32x512xbf16>
                  %expanded_9 = tensor.expand_shape %unpack [[0, 1], [2]] output_shape [1, 32, 512] : tensor<512x32xbf16> into tensor<1x512x32xbf16>
                  %14 = linalg.batch_reduce_matmul ins(%expanded, %expanded_9 : tensor<1x32x512xbf16>, tensor<1x512x32xbf16>) outs(%13 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %15 = affine.apply #map4(%arg12, %arg24, %arg4)
                  %16 = affine.apply #map5(%arg8, %arg20, %arg3)
                  %extracted_slice_10 = tensor.extract_slice %arg2[%15] [32] [1] : tensor<256xbf16> to tensor<32xbf16>
                  %extracted_slice_11 = tensor.extract_slice %0[%16, %15] [32, 32] [1, 1] : tensor<128x256xbf16> to tensor<32x32xbf16>
                  %broadcasted = linalg.broadcast ins(%extracted_slice_10 : tensor<32xbf16>) outs(%extracted_slice_11 : tensor<32x32xbf16>) dimensions = [0] 
                  %extracted_slice_12 = tensor.extract_slice %arg30[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %17 = linalg.add ins(%14, %broadcasted : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%extracted_slice_12 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_13 = tensor.insert_slice %14 into %arg29[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  %extracted_slice_14 = tensor.extract_slice %arg31[%arg20, %arg24] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %18 = linalg.exp ins(%17 : tensor<32x32xbf16>) outs(%extracted_slice_14 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_15 = tensor.insert_slice %17 into %arg30[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  %inserted_slice_16 = tensor.insert_slice %18 into %arg31[%arg20, %arg24] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  scf.yield %inserted_slice_13, %inserted_slice_15, %inserted_slice_16 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
                }
                scf.yield %10#0, %10#1, %10#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
              }
              scf.yield %9#0, %9#1, %9#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
            }
            %inserted_slice = tensor.insert_slice %8#0 into %arg17[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            %inserted_slice_3 = tensor.insert_slice %8#1 into %arg18[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            %inserted_slice_4 = tensor.insert_slice %8#2 into %arg19[%arg8, %arg12] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            scf.yield %inserted_slice, %inserted_slice_3, %inserted_slice_4 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
          }
          scf.yield %7#0, %7#1, %7#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
        }
        scf.yield %6#0, %6#1, %6#2 : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5#2 into %arg7[%3, %4] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
        tensor.parallel_insert_slice %5#1 into %arg6[%3, %4] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
        tensor.parallel_insert_slice %5#0 into %arg5[%3, %4] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
      }
    }
    return %2#2 : tensor<128x256xbf16>
  }

func.func @matmul_add(%arg0: tensor<8192x12288xf16>, %arg1: tensor<12288x16384xf16>, %arg2: tensor<8192x16384xf32>, %arg3: tensor<8192x16384xf32>) -> tensor<8192x16384xf32> {
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

