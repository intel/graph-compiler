// RUN: gc-opt --split-input-file -iterative-tiling-and-fusion %s --cse

module {
  /// CHECK-LABEL: @fuse_mlp
  func.func @fuse_mlp(%arg0: tensor<128x512xbf16>, %arg1: tensor<32x8x16x32xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    /// CHECK: tensor.empty
    %0 = tensor.empty() : tensor<128x256xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    /// CHECK: tensor.empty
    %dest = tensor.empty() : tensor<512x256xbf16>
    %unpack = tensor.unpack %arg1 inner_dims_pos = [0, 1] inner_tiles = [16, 32] into %dest : tensor<32x8x16x32xbf16> -> tensor<512x256xbf16>
    /// CHECK:   %[[FINAL_RESULT:.*]]:3 = scf.forall (%{{.*}}) in (2, 2)
    %2 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %1) -> (tensor<128x256xbf16>) {
      %5 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %7 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %8 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%5, 0] [64, 512] [1, 1] : tensor<128x512xbf16> to tensor<64x512xbf16>
      %extracted_slice_0 = tensor.extract_slice %unpack[0, %6] [512, 128] [1, 1] : tensor<512x256xbf16> to tensor<512x128xbf16>
      %extracted_slice_1 = tensor.extract_slice %arg5[%7, %8] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      /// CHECK: scf.for
      /// CHECK: scf.for
      /// CHECK: scf.for
      %9 = scf.for %arg6 = %c0 to %c64 step %c64 iter_args(%arg7 = %extracted_slice_1) -> (tensor<64x128xbf16>) {
        %12 = scf.for %arg8 = %c0 to %c128 step %c128 iter_args(%arg9 = %arg7) -> (tensor<64x128xbf16>) {
          %13 = scf.for %arg10 = %c0 to %c512 step %c512 iter_args(%arg11 = %arg9) -> (tensor<64x128xbf16>) {
            %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg6, %arg10] [64, 512] [1, 1] : tensor<64x512xbf16> to tensor<64x512xbf16>
            %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg10, %arg8] [512, 128] [1, 1] : tensor<512x128xbf16> to tensor<512x128xbf16>
            %extracted_slice_4 = tensor.extract_slice %arg11[%arg6, %arg8] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            /// CHECK: scf.for
            /// CHECK: scf.for
            /// CHECK: scf.for
            %14 = scf.for %arg12 = %c0 to %c64 step %c32 iter_args(%arg13 = %extracted_slice_4) -> (tensor<64x128xbf16>) {
              %15 = scf.for %arg14 = %c0 to %c128 step %c32 iter_args(%arg15 = %arg13) -> (tensor<64x128xbf16>) {
                %16 = scf.for %arg16 = %c0 to %c512 step %c512 iter_args(%arg17 = %arg15) -> (tensor<64x128xbf16>) {
                  %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[%arg12, %arg16] [32, 512] [1, 1] : tensor<64x512xbf16> to tensor<32x512xbf16>
                  %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[%arg16, %arg14] [512, 32] [1, 1] : tensor<512x128xbf16> to tensor<512x32xbf16>
                  %extracted_slice_7 = tensor.extract_slice %arg17[%arg12, %arg14] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  /// CHECK: %[[UNPACK_OUT:.*]] = tensor.unpack
                  /// CHECK: %[[FILL_OUT:.*]] = linalg.fill
                  /// CHECK: %[[EXPAND_OUT_1:.*]] = tensor.expand_shape
                  /// CHECK: %[[EXPAND_OUT_2:.*]] = tensor.expand_shape
                  %expanded = tensor.expand_shape %extracted_slice_5 [[0, 1], [2]] output_shape [1, 32, 512] : tensor<32x512xbf16> into tensor<1x32x512xbf16>
                  %expanded_8 = tensor.expand_shape %extracted_slice_6 [[0, 1], [2]] output_shape [1, 32, 512] : tensor<512x32xbf16> into tensor<1x512x32xbf16>
                  /// CHECK: %[[MATMUL_OUT:.*]] = linalg.batch_reduce_matmul ins(%[[EXPAND_OUT_1]], %[[EXPAND_OUT_2]] :
                  %17 = linalg.batch_reduce_matmul ins(%expanded, %expanded_8 : tensor<1x32x512xbf16>, tensor<1x512x32xbf16>) outs(%extracted_slice_7 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  /// CHECK: %[[BROADCAST_OUT:.*]] = linalg.broadcast
                  /// CHECK: %[[ADD_OUT:.*]] = linalg.add ins(%[[MATMUL_OUT]], %[[BROADCAST_OUT]] :
                  /// CHECK: %[[EXP_OUT:.*]] = linalg.exp ins(%[[ADD_OUT]] :
                  %inserted_slice_9 = tensor.insert_slice %17 into %arg17[%arg12, %arg14] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
                  scf.yield %inserted_slice_9 : tensor<64x128xbf16>
                }
                /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
                scf.yield %16 : tensor<64x128xbf16>
              }
              /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
              scf.yield %15 : tensor<64x128xbf16>
            }
            /// CHECK: tensor.insert_slice
            /// CHECK: tensor.insert_slice
            /// CHECK: tensor.insert_slice
            %inserted_slice = tensor.insert_slice %14 into %arg11[%arg6, %arg8] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}  : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
            scf.yield %inserted_slice : tensor<64x128xbf16>
          }
          /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}  : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
          scf.yield %13 : tensor<64x128xbf16>
        }
        /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}  : tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>
        scf.yield %12 : tensor<64x128xbf16>
      }
      %10 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %11 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      scf.forall.in_parallel {
        /// CHECK: tensor.parallel_insert_slice
        /// CHECK: tensor.parallel_insert_slice
        /// CHECK: tensor.parallel_insert_slice
        tensor.parallel_insert_slice %9 into %arg5[%10, %11] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
      }
    }
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<256xbf16>) outs(%0 : tensor<128x256xbf16>) dimensions = [0] 
    %3 = linalg.add ins(%2, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %4 = linalg.exp ins(%3 : tensor<128x256xbf16>) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    /// CHECK: return %[[FINAL_RESULT]]#2
    return %4 : tensor<128x256xbf16>
  }
}

// -----

#map = affine_map<(d0) -> (d0 * 128)>
module {
  /// CHECK-LABEL: @fuse_multiple_consumer
  func.func @fuse_multiple_consumer(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256x256xf32>, %arg3: tensor<256x256xf32>) -> (tensor<256x256xf32>, tensor<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dest0 = tensor.empty() : tensor<256x256xf32>
    %dest1 = linalg.fill ins(%cst : f32) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    /// CHECK:   %[[FINAL_RESULT:.*]]:3 = scf.forall (%{{.*}}) in (2, 2)
    %1 = scf.forall (%arg4, %arg5) in (2, 2) shared_outs(%arg6 = %dest1) -> tensor<256x256xf32> {
      %iv0 = affine.apply #map(%arg4)
      %iv1 = affine.apply #map(%arg5)
      %extracted_slice_1 = tensor.extract_slice %arg6[%iv0, %iv1] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
      %extracted_slice_2 = tensor.extract_slice %arg0[%iv0, 0] [128, 512] [1, 1] : tensor<256x512xf32> to tensor<128x512xf32>
      %extracted_slice_3 = tensor.extract_slice %arg1[0, %iv1] [512, 128] [1, 1] : tensor<512x256xf32> to tensor<512x128xf32>
      /// CHECK: scf.for
      /// CHECK: scf.for
      %2 = scf.for %arg7 = %c0 to %c128 step %c64 iter_args(%arg8 = %extracted_slice_1) -> (tensor<128x128xf32>) {
        %3 = scf.for %arg9 = %c0 to %c128 step %c64 iter_args(%arg10 = %arg8) -> (tensor<128x128xf32>) {
          %extracted_slice_4 = tensor.extract_slice %arg10[%arg7, %arg9] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
          %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[%arg7, 0] [64, 512] [1, 1] : tensor<128x512xf32> to tensor<64x512xf32>
          %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg9] [512, 64] [1, 1] : tensor<512x128xf32> to tensor<512x64xf32>
          /// CHECK: %[[MATMUL_OUT:.*]] = linalg.matmul
          %4 = linalg.matmul ins(%extracted_slice_5, %extracted_slice_6 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%extracted_slice_4 : tensor<64x64xf32>) -> tensor<64x64xf32>
          /// CHECK: %[[MUL_OUT:.*]] = linalg.mul ins(%[[MATMUL_OUT]],
          /// CHECK: %[[ADD_OUT:.*]] = linalg.add ins(%[[MATMUL_OUT]],
          %insert_slice = tensor.insert_slice %4 into %arg10[%arg7, %arg9] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
          /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}  : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
          scf.yield %insert_slice : tensor<128x128xf32>
        }
        /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}  : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
        scf.yield %3 : tensor<128x128xf32>
      }
      scf.forall.in_parallel {
        /// CHECK: tensor.parallel_insert_slice
        /// CHECK: tensor.parallel_insert_slice
        /// CHECK: tensor.parallel_insert_slice
        tensor.parallel_insert_slice %2 into %arg6[%iv0, %iv1] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x256xf32>
      }
    }
    %5 = linalg.add ins(%1, %arg2 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %6 = linalg.mul ins(%1, %arg3 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    /// CHECK: return %[[FINAL_RESULT]]#2, %[[FINAL_RESULT]]#1
    return %5, %6 : tensor<256x256xf32>, tensor<256x256xf32>
  }
}

// -----

#map = affine_map<(d0) -> (d0 * 128)>
module {
  /// CHECK-LABEL: @fuse_reduce
  func.func @fuse_reduce(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dest0 = tensor.empty() : tensor<256x256xf32>
    %dest1 = linalg.fill ins(%cst : f32) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    /// CHECK:   %[[FINAL_RESULT:.*]]:3 = scf.forall (%{{.*}}) in (2, 1)
    %1 = scf.forall (%arg3, %arg4) in (2, 1) shared_outs(%arg5 = %dest1) -> tensor<256x256xf32> {
      %iv0 = affine.apply #map(%arg3)
      %iv1 = affine.apply #map(%arg4)
      %extracted_slice_1 = tensor.extract_slice %arg5[%iv0, %iv1] [128, 256] [1, 1] : tensor<256x256xf32> to tensor<128x256xf32>
      %extracted_slice_2 = tensor.extract_slice %arg0[%iv0, 0] [128, 512] [1, 1] : tensor<256x512xf32> to tensor<128x512xf32>
      %extracted_slice_3 = tensor.extract_slice %arg1[0, %iv1] [512, 256] [1, 1] : tensor<512x256xf32> to tensor<512x256xf32>
      /// CHECK: %[[FOR_RESULT:.*]]:2 = scf.for
      /// CHECK: scf.for
      %2 = scf.for %arg6 = %c0 to %c128 step %c64 iter_args(%arg7 = %extracted_slice_1) -> (tensor<128x256xf32>) {
        %3 = scf.for %arg8 = %c0 to %c256 step %c64 iter_args(%arg9 = %arg7) -> (tensor<128x256xf32>) {
          %extracted_slice_4 = tensor.extract_slice %arg9[%arg6, %arg8] [64, 64] [1, 1] : tensor<128x256xf32> to tensor<64x64xf32>
          %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[%arg6, 0] [64, 512] [1, 1] : tensor<128x512xf32> to tensor<64x512xf32>
          %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg8] [512, 64] [1, 1] : tensor<512x256xf32> to tensor<512x64xf32>
          /// CHECK: %[[MATMUL_OUT:.*]] = linalg.matmul
          %4 = linalg.matmul ins(%extracted_slice_5, %extracted_slice_6 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%extracted_slice_4 : tensor<64x64xf32>) -> tensor<64x64xf32>
          /// CHECK: %[[ADD_OUT:.*]] = linalg.add ins(%[[MATMUL_OUT]],
          %insert_slice = tensor.insert_slice %4 into %arg9[%arg6, %arg8] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x256xf32>
          /// CHECK: scf.yield {{.*}}, {{.*}} : tensor<128x256xf32>, tensor<128x256xf32>
          scf.yield %insert_slice : tensor<128x256xf32>
        }
        /// CHECK: scf.yield {{.*}}, {{.*}} : tensor<128x256xf32>, tensor<128x256xf32>
        scf.yield %3 : tensor<128x256xf32>
      }
      /// CHECK:  %[[REDUCE_OUT:.*]] = linalg.reduce { arith.addf } ins(%[[FOR_RESULT]]#1 :
      scf.forall.in_parallel {
        /// CHECK: tensor.parallel_insert_slice
        /// CHECK: tensor.parallel_insert_slice
        /// CHECK: tensor.parallel_insert_slice
        tensor.parallel_insert_slice %2 into %arg5[%iv0, %iv1] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<256x256xf32>
      }
    }
    %5 = linalg.add ins(%1, %arg2 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %dest2 = tensor.empty() : tensor<256xf32>
    %6 = linalg.reduce { arith.addf } ins(%5 : tensor<256x256xf32>) outs(%dest2 : tensor<256xf32>) dimensions = [1]
    /// CHECK: return %[[FINAL_RESULT]]#2
    return %6 : tensor<256xf32>
  }
}

// -----

module {
  /// CHECK-LABEL: @fuse_with_default_tiling
  func.func @fuse_with_default_tiling(%arg0: tensor<128x256x256xf32>, %arg1: tensor<128x256x256xf32>) -> tensor<128x256xf32> {
    %dest0 = tensor.empty() : tensor<128x256x256xf32>
    %0 = linalg.add ins(%arg0, %arg1 : tensor<128x256x256xf32>, tensor<128x256x256xf32>) outs(%dest0 : tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %dest1 = tensor.empty() : tensor<128x256xf32>
    /// CHECK:   %[[FINAL_RESULT:.*]] = scf.forall (%{{.*}}) = (0, 0) to (128, 256) step (1, 32)
    /// CHECK: tensor.extract_slice {{.*}} [1, 256, 32] [1, 1, 1]
    /// CHECK: tensor.extract_slice {{.*}} [1, 256, 32] [1, 1, 1]
    /// CHECK: tensor.extract_slice {{.*}} [1, 256, 32] [1, 1, 1]
    /// CHECK: %[[ADD_OUT:.*]] = linalg.add
    /// CHECK: tensor.extract_slice {{.*}} [1, 32] [1, 1]
    /// CHECK: %[[REDUCE_OUT:.*]] = linalg.reduce { arith.addf } ins(%[[ADD_OUT]] :
    %1 = linalg.reduce { arith.addf } ins(%0 : tensor<128x256x256xf32>) outs(%dest1 : tensor<128x256xf32>) dimensions = [1]
    /// CHECK: scf.forall.in_parallel
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: return %[[FINAL_RESULT]]
    return %1 : tensor<128x256xf32>
  }
}

// -----

module {
  /// CHECK-LABEL: @fuse_llama2_matmul_add
  func.func @fuse_llama2_matmul_add(%arg0: tensor<32x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: tensor<32x4096xbf16>) -> tensor<32x4096xbf16> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<1x128x32x32xbf16>
    %pack = tensor.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<32x4096xbf16> -> tensor<1x128x32x32xbf16>
    %1 = tensor.empty() : tensor<128x128x32x32xbf16>
    %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<4096x4096xbf16> -> tensor<128x128x32x32xbf16>
    %2 = tensor.empty() : tensor<1x128x32x32xbf16>
    %3 = tensor.empty() : tensor<128x128x16x32x2xbf16>
    %pack_1 = tensor.pack %pack_0 inner_dims_pos = [2] inner_tiles = [2] into %3 : tensor<128x128x32x32xbf16> -> tensor<128x128x16x32x2xbf16>
    /// CHECK:   %[[FINAL_RESULT:.*]]:3 = scf.for
    %4 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %2) -> (tensor<1x128x32x32xbf16>) {
      %extracted_slice = tensor.extract_slice %arg4[0, %arg3, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<1x128x32x32xbf16> to tensor<1x16x32x32xbf16>
      %9 = tensor.empty() : tensor<1x16x32x32xf32>
      /// CHECK: scf.for
      /// CHECK: scf.for
      %10:2 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %9, %arg7 = %extracted_slice) -> (tensor<1x16x32x32xf32>, tensor<1x16x32x32xbf16>) {
        %11:2 = scf.for %arg8 = %c0 to %c16 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<1x16x32x32xf32>, tensor<1x16x32x32xbf16>) {
          %extracted_slice_3 = tensor.extract_slice %pack[0, %arg5, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : tensor<1x128x32x32xbf16> to tensor<1x32x32x32xbf16>
          %12 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg8, %arg3]
          /// CHECK: %[[PACK_OUT_1:.*]] = tensor.pack
          /// CHECK: %[[PACK_OUT_2:.*]] = tensor.pack
          /// CHECK: %[[PACK_OUT_3:.*]] = tensor.pack %[[PACK_OUT_2]]
          %extracted_slice_4 = tensor.extract_slice %pack_1[%12, %arg5, 0, 0, 0] [1, 32, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<128x128x16x32x2xbf16> to tensor<1x32x16x32x2xbf16>
          %extracted_slice_5 = tensor.extract_slice %arg9[0, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x16x32x32xf32> to tensor<1x1x32x32xf32>
          %extracted_slice_6 = tensor.extract_slice %arg10[0, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x16x32x32xbf16> to tensor<1x1x32x32xbf16>
          /// CHECK: %[[COLLAPSE_OUT_1:.*]] = tensor.collapse_shape %[[PACK_OUT_1]]
          /// CHECK: %[[COLLAPSE_OUT_2:.*]] = tensor.collapse_shape %[[PACK_OUT_3]]
          %collapse_3 = tensor.collapse_shape %extracted_slice_3 [[0, 1], [2], [3]] : tensor<1x32x32x32xbf16> into tensor<32x32x32xbf16>
          %collapse_4 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2], [3], [4]] : tensor<1x32x16x32x2xbf16> into tensor<32x16x32x2xbf16>
          %collapse_5 = tensor.collapse_shape %extracted_slice_5 [[0, 1, 2], [3]] : tensor<1x1x32x32xf32> into tensor<32x32xf32>
          %collapse_6 = tensor.collapse_shape %extracted_slice_6 [[0, 1, 2], [3]] : tensor<1x1x32x32xbf16> into tensor<32x32xbf16>
          %13 = arith.cmpi eq, %arg5, %c0 : index
          /// CHECK: %[[IF_RESULT_1:.*]] = scf.if
          /// CHECK: linalgx.batch_reduce_matmul_vnni ins(%[[COLLAPSE_OUT_1]], %[[COLLAPSE_OUT_2]] :
          /// CHECK: } else {
          /// CHECK: linalgx.batch_reduce_matmul_vnni ins(%[[COLLAPSE_OUT_1]], %[[COLLAPSE_OUT_2]] :
          /// CHECK: }
          %14 = scf.if %13 -> (tensor<32x32xf32>) {
            %18 = linalg.fill ins(%cst : bf16) outs(%collapse_5 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %19 = linalgx.batch_reduce_matmul_vnni ins(%collapse_3, %collapse_4 : tensor<32x32x32xbf16>, tensor<32x16x32x2xbf16>) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
            scf.yield %19 : tensor<32x32xf32>
          } else {
            %18 = linalgx.batch_reduce_matmul_vnni ins(%collapse_3, %collapse_4 : tensor<32x32x32xbf16>, tensor<32x16x32x2xbf16>) outs(%collapse_5 : tensor<32x32xf32>) -> tensor<32x32xf32>
            scf.yield %18 : tensor<32x32xf32>
          }
          %15 = arith.addi %arg5, %c32 : index
          %16 = arith.cmpi sge, %15, %c128 : index
          /// CHECK: %[[IF_RESULT_2:.*]] = scf.if
          %17 = scf.if %16 -> (tensor<32x32xbf16>) {
            %18 = linalg.copy ins(%14 : tensor<32x32xf32>) outs(%collapse_6 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
            scf.yield %18 : tensor<32x32xbf16>
          } else {
            scf.yield %collapse_6 : tensor<32x32xbf16>
          }
          /// CHECK: %[[EXPAND_OUT_1:.*]] = tensor.expand_shape %[[IF_RESULT_1]]
          /// CHECK: %[[EXPAND_OUT_2:.*]] = tensor.expand_shape %[[IF_RESULT_2]]
          %expand_14 = tensor.expand_shape %14 [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xf32> into tensor<1x1x32x32xf32>
          %expand_17 = tensor.expand_shape %17 [[0, 1, 2], [3]] output_shape [1, 1, 32, 32] : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
          /// CHECK: %[[PACK_OUT_4:.*]] = tensor.pack
          /// CHECK: %[[ADD_OUT:.*]] = linalg.add ins(%[[EXPAND_OUT_2]], %[[PACK_OUT_4]] :
          /// CHECK: %[[UNPACK_OUT:.*]] = tensor.unpack %[[ADD_OUT]]
          %inserted_slice_7 = tensor.insert_slice %expand_14 into %arg9[0, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x32xf32> into tensor<1x16x32x32xf32>
          %inserted_slice_8 = tensor.insert_slice %expand_17 into %arg10[0, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x32xbf16> into tensor<1x16x32x32xbf16>
          /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}} : tensor<1x16x32x32xf32>, tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>, tensor<32x512xbf16>
          scf.yield %inserted_slice_7, %inserted_slice_8 : tensor<1x16x32x32xf32>, tensor<1x16x32x32xbf16>
        }
        /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}} : tensor<1x16x32x32xf32>, tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>, tensor<32x512xbf16>
        scf.yield %11#0, %11#1 : tensor<1x16x32x32xf32>, tensor<1x16x32x32xbf16>
      }
      /// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}: tensor<1x128x32x32xbf16>, tensor<1x128x32x32xbf16>, tensor<32x4096xbf16>
      %inserted_slice = tensor.insert_slice %10#1 into %arg4[0, %arg3, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<1x16x32x32xbf16> into tensor<1x128x32x32xbf16>
      scf.yield %inserted_slice : tensor<1x128x32x32xbf16>
    }
    %5 = tensor.empty() : tensor<32x4096xbf16>
    %6 = tensor.empty() : tensor<1x128x32x32xbf16>
    %pack_2 = tensor.pack %arg2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %6 : tensor<32x4096xbf16> -> tensor<1x128x32x32xbf16>
    %7 = tensor.empty() : tensor<1x128x32x32xbf16>
    %8 = linalg.add ins(%4, %pack_2 : tensor<1x128x32x32xbf16>, tensor<1x128x32x32xbf16>) outs(%7 : tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>
    %unpack = tensor.unpack %8 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %5 : tensor<1x128x32x32xbf16> -> tensor<32x4096xbf16>
    /// CHECK: return %[[FINAL_RESULT]]#2
    return %unpack : tensor<32x4096xbf16>
  }
}

// -----

module {
  /// CHECK-LABEL: @fuse_residual_pattern
  func.func @fuse_residual_pattern(%arg0: tensor<128x256x256xf32>, %arg1: tensor<128x256x256xf32>) -> tensor<128x256x256xf32> {
    %dest0 = tensor.empty() : tensor<128x256x256xf32>
    /// CHECK: %[[FINAL_RESULT:.*]]:3 = scf.forall (%{{.*}}) = (0, 0, 0) to (128, 256, 256) step (1, 32, 32)
    /// CHECK: %[[ADD_OUT:.*]] = linalg.add
    /// CHECK: %[[EXP_OUT:.*]] = linalg.exp ins(%[[ADD_OUT:.*]] :
    /// CHECK: %[[MUL_OUT:.*]] = linalg.mul ins(%[[ADD_OUT:.*]], %[[EXP_OUT:.*]] :
    %0 = linalg.add ins(%arg0, %arg1 : tensor<128x256x256xf32>, tensor<128x256x256xf32>) outs(%dest0 : tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %1 = linalg.exp ins(%0 : tensor<128x256x256xf32>) outs(%dest0 : tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %2 = linalg.mul ins(%0, %1 : tensor<128x256x256xf32>, tensor<128x256x256xf32>) outs(%dest0 : tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    /// CHECK: scf.forall.in_parallel
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: return %[[FINAL_RESULT]]#2
    return %2 : tensor<128x256x256xf32>
  }
}

// -----

module {
  /// CHECK-LABEL: @not_fuse_pack
  func.func @not_fuse_pack(%arg0: tensor<1x35x4096xbf16>, %arg1: tensor<1x35x4096xbf16>) -> tensor<1x2x128x32x32xbf16> {
    %dest0 = tensor.empty() : tensor<1x35x4096xbf16>
    /// CHECK: scf.forall
    /// CHECK: linalg.add
    %add = linalg.add ins(%arg0, %arg1 : tensor<1x35x4096xbf16>, tensor<1x35x4096xbf16>) outs(%dest0 : tensor<1x35x4096xbf16>) -> tensor<1x35x4096xbf16>
    /// CHECK: }
    %dest1 = tensor.empty() : tensor<1x2x128x32x32xbf16>
    %pad = arith.constant 0.000000e+00 : bf16
    /// CHECK: %[[PACK_OUT:.*]] = scf.forall
    /// CHECK: tensor.pack
    %pack = tensor.pack %add padding_value(%pad : bf16) outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %dest1 : tensor<1x35x4096xbf16> -> tensor<1x2x128x32x32xbf16>
    /// CHECK: }
    /// CHECK: return %[[PACK_OUT]]
    return %pack : tensor<1x2x128x32x32xbf16>
  }
}

// -----

module {
  /// CHECK-LABEL: @fuse_pack
  func.func @fuse_pack(%arg0: tensor<1x32x4096xbf16>, %arg1: tensor<1x32x4096xbf16>) -> tensor<1x1x128x32x32xbf16> {
    %dest0 = tensor.empty() : tensor<1x32x4096xbf16>
    /// CHECK: %[[FINAL_RESULT:.*]]:2 = scf.forall (%{{.*}}) = (0, 0) to (1, 4096) step (1, 32)
    /// CHECK: linalg.add
    %add = linalg.add ins(%arg0, %arg1 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%dest0 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
    %dest1 = tensor.empty() : tensor<1x1x128x32x32xbf16>
    /// CHECK-NEXT: affine.apply
    /// CHECK-NEXT: tensor.extract_slice
    /// CHECK-NEXT: tensor.pack
    %pack = tensor.pack %add outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %dest1 : tensor<1x32x4096xbf16> -> tensor<1x1x128x32x32xbf16>
    /// CHECK: scf.forall.in_parallel
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: return %[[FINAL_RESULT]]#1
    return %pack : tensor<1x1x128x32x32xbf16>
  }
}

// -----

module {
  /// CHECK-LABEL:    @fuse_generic_matmul
  /// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
  /// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<2x16x16xf32>
  /// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<4x16x16xf32>
  func.func @fuse_generic_matmul(%arg0: tensor<32x32xf32>, %arg1: tensor<2x16x16xf32>, %arg2: tensor<4x16x16xf32>) -> tensor<32x64xf32> {
    /// CHECK: %[[EMPTY_OUT_0:.*]] = tensor.empty
    %0 = tensor.empty() : tensor<2x2x16x16xf32>
    %pack = tensor.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<32x32xf32> -> tensor<2x2x16x16xf32>
    /// CHECK: %[[EMPTY_OUT_1:.*]] = tensor.empty
    %1 = tensor.empty() : tensor<2x16x16xf32>
    /// CHECK: %[[FIRST_MATMUL_OUT:.*]] = scf.forall (%{{.*}}) in (2)
    /// CHECK: %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[ARG0]]{{.*}} [16, 32]
    /// CHECK: %[[EXTRACT_SLICE_1:.*]] = tensor.extract_slice %[[EMPTY_OUT_0]]{{.*}} [1, 2, 16, 16]
    /// CHECK: %[[PACK_OUT:.*]] = tensor.pack %[[EXTRACT_SLICE_0]]
    /// CHECK: %[[EXTRACT_SLICE_2:.*]] = tensor.extract_slice %[[ARG1]]{{.*}} [2, 16, 16]
    /// CHECK: %[[MATMUL_OUT_0:.*]] = linalg.generic {{.*}} ins(%[[PACK_OUT]], %[[EXTRACT_SLICE_2]] :
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %arg1 : tensor<2x2x16x16xf32>, tensor<2x16x16xf32>) outs(%1 : tensor<2x16x16xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %9 = arith.mulf %in, %in_3 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<2x16x16xf32>
    /// CHECK: scf.forall.in_parallel
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: }
    /// CHECK: %[[EMPTY_OUT_2:.*]] = tensor.empty
    /// CHECK: %[[EMPTY_OUT_3:.*]] = tensor.empty
    %3 = tensor.empty() : tensor<2x4x16x16xf32>
    /// CHECK: %[[FINAL_RESULT:.*]]:2 = scf.forall (%{{.*}}) in (2, 4)
    /// CHECK: %[[EXTRACT_SLICE_3:.*]] = tensor.extract_slice %[[FIRST_MATMUL_OUT]]{{.*}} [1, 16, 16]
    /// CHECK: %[[EXTRACT_SLICE_4:.*]] = tensor.extract_slice %[[ARG2]]{{.*}} [1, 16, 16]
    /// CHECK: %[[MATMUL_OUT_1:.*]] = linalg.generic {{.*}} ins(%[[EXTRACT_SLICE_3]], %[[EXTRACT_SLICE_4]] :
    /// CHECK: %[[UNPACK_OUT:.*]] = tensor.unpack %[[MATMUL_OUT_1]]
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2, %arg2 : tensor<2x16x16xf32>, tensor<4x16x16xf32>) outs(%3 : tensor<2x4x16x16xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %9 = arith.mulf %in, %in_3 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<2x4x16x16xf32>
    /// CHECK: scf.forall.in_parallel
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: tensor.parallel_insert_slice
    /// CHECK: }
    %5 = tensor.empty() : tensor<32x64xf32>
    %unpack = tensor.unpack %4 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %5 : tensor<2x4x16x16xf32> -> tensor<32x64xf32>
    /// CHECK: return %[[FINAL_RESULT]]#1
    return %unpack : tensor<32x64xf32>
  }
}

// -----

module {
  /// CHECK-LABEL: @yield_fused_producer
  func.func @yield_fused_producer(%arg0: tensor<16x32x32xf32>) -> (tensor<16x32x32xf32>, tensor<16x32xf32>) {
    /// CHECK: arith.constant
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<16x32x32xf32>
    /// CHECK-NEXT: tensor.empty
    %dest0 = tensor.empty() : tensor<16x32x32xf32>
    %0 = linalg.powf ins(%arg0, %cst_0 : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%dest0 : tensor<16x32x32xf32>) -> tensor<16x32x32xf32>
    /// CHECK-NEXT: tensor.empty
    %dest1 = tensor.empty() : tensor<16x32xf32>
    /// CHECK-NEXT: %[[FINAL_RESULT:.*]]:2 = scf.forall (%{{.*}}) in (16)
    /// CHECK-NEXT: tensor.extract_slice
    /// CHECK-NEXT: tensor.extract_slice
    /// CHECK-NEXT: tensor.extract_slice
    /// CHECK-NEXT: linalg.powf
    /// CHECK-NEXT: tensor.extract_slice
    /// CHECK-NEXT: linalg.reduce
    %1 = linalg.reduce { arith.addf } ins(%0 : tensor<16x32x32xf32>) outs(%dest1 : tensor<16x32xf32>) dimensions = [2]
    /// CHECK-NEXT: scf.forall.in_parallel
    /// CHECK-NEXT: tensor.parallel_insert_slice
    /// CHECK-NEXT: tensor.parallel_insert_slice
    /// CHECK-NEXT: }
    /// CHECK: return %[[FINAL_RESULT]]#1, %[[FINAL_RESULT]]#0
    return %0, %1 : tensor<16x32x32xf32>, tensor<16x32xf32>
  }
}