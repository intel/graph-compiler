// RUN: gc-opt --split-input-file -any-tilable-fusion %s

func.func @mlp(%arg0: tensor<128x512xbf16>, %arg1: tensor<32x8x16x32xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x256xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %dest = tensor.empty() : tensor<512x256xbf16>
    %unpack = tensor.unpack %arg1 inner_dims_pos = [0, 1] inner_tiles = [16, 32] into %dest : tensor<32x8x16x32xbf16> -> tensor<512x256xbf16>
    %2 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %1) -> (tensor<128x256xbf16>) {
      %5 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %7 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %8 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%5, 0] [64, 512] [1, 1] : tensor<128x512xbf16> to tensor<64x512xbf16>
      %extracted_slice_0 = tensor.extract_slice %unpack[0, %6] [512, 128] [1, 1] : tensor<512x256xbf16> to tensor<512x128xbf16>
      %extracted_slice_1 = tensor.extract_slice %arg5[%7, %8] [64, 128] [1, 1] : tensor<128x256xbf16> to tensor<64x128xbf16>
      %9 = scf.for %arg6 = %c0 to %c64 step %c64 iter_args(%arg7 = %extracted_slice_1) -> (tensor<64x128xbf16>) {
        %12 = scf.for %arg8 = %c0 to %c128 step %c128 iter_args(%arg9 = %arg7) -> (tensor<64x128xbf16>) {
          %13 = scf.for %arg10 = %c0 to %c512 step %c512 iter_args(%arg11 = %arg9) -> (tensor<64x128xbf16>) {
            %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg6, %arg10] [64, 512] [1, 1] : tensor<64x512xbf16> to tensor<64x512xbf16>
            %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg10, %arg8] [512, 128] [1, 1] : tensor<512x128xbf16> to tensor<512x128xbf16>
            %extracted_slice_4 = tensor.extract_slice %arg11[%arg6, %arg8] [64, 128] [1, 1] : tensor<64x128xbf16> to tensor<64x128xbf16>
            %14 = scf.for %arg12 = %c0 to %c64 step %c32 iter_args(%arg13 = %extracted_slice_4) -> (tensor<64x128xbf16>) {
              %15 = scf.for %arg14 = %c0 to %c128 step %c32 iter_args(%arg15 = %arg13) -> (tensor<64x128xbf16>) {
                %16 = scf.for %arg16 = %c0 to %c512 step %c512 iter_args(%arg17 = %arg15) -> (tensor<64x128xbf16>) {
                  %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[%arg12, %arg16] [32, 512] [1, 1] : tensor<64x512xbf16> to tensor<32x512xbf16>
                  %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[%arg16, %arg14] [512, 32] [1, 1] : tensor<512x128xbf16> to tensor<512x32xbf16>
                  %extracted_slice_7 = tensor.extract_slice %arg17[%arg12, %arg14] [32, 32] [1, 1] : tensor<64x128xbf16> to tensor<32x32xbf16>
                  %expanded = tensor.expand_shape %extracted_slice_5 [[0, 1], [2]] : tensor<32x512xbf16> into tensor<1x32x512xbf16>
                  %expanded_8 = tensor.expand_shape %extracted_slice_6 [[0, 1], [2]] : tensor<512x32xbf16> into tensor<1x512x32xbf16>
                  %17 = linalg.batch_reduce_matmul ins(%expanded, %expanded_8 : tensor<1x32x512xbf16>, tensor<1x512x32xbf16>) outs(%extracted_slice_7 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
                  %inserted_slice_9 = tensor.insert_slice %17 into %arg17[%arg12, %arg14] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x128xbf16>
                  scf.yield %inserted_slice_9 : tensor<64x128xbf16>
                }
                scf.yield %16 : tensor<64x128xbf16>
              }
              scf.yield %15 : tensor<64x128xbf16>
            }
            %inserted_slice = tensor.insert_slice %14 into %arg11[%arg6, %arg8] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<64x128xbf16>
            scf.yield %inserted_slice : tensor<64x128xbf16>
          }
          scf.yield %13 : tensor<64x128xbf16>
        }
        scf.yield %12 : tensor<64x128xbf16>
      }
      %10 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
      %11 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg5[%10, %11] [64, 128] [1, 1] : tensor<64x128xbf16> into tensor<128x256xbf16>
      }
    }
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<256xbf16>) outs(%0 : tensor<128x256xbf16>) dimensions = [0] 
    %3 = linalg.add ins(%2, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %4 = linalg.exp ins(%3 : tensor<128x256xbf16>) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    return %4 : tensor<128x256xbf16>
  }