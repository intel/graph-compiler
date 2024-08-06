#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0, d1) -> (d0 * 16 + d1 * 64)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main_entry(%arg0: tensor<128x512xf32>, %arg1: tensor<512x64xf32>) -> tensor<128x64xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %0) -> (tensor<128x64xf32>) {
      %7 = affine.apply #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg3[%7, 0] [64, 64] [1, 1] : tensor<128x64xf32> to tensor<64x64xf32>
      %8 = scf.forall (%arg4, %arg5) in (4, 4) shared_outs(%arg6 = %extracted_slice) -> (tensor<64x64xf32>) {
        %10 = affine.apply #map1(%arg4)
        %11 = affine.apply #map1(%arg5)
        %12 = affine.apply #map2(%arg4, %arg2)
        %extracted_slice_0 = tensor.extract_slice %arg0[%12, 0] [16, 256] [1, 1] : tensor<128x512xf32> to tensor<16x256xf32>
        %13 = affine.apply #map1(%arg5)
        %extracted_slice_1 = tensor.extract_slice %arg1[0, %13] [256, 16] [1, 1] : tensor<512x64xf32> to tensor<256x16xf32>
        %extracted_slice_2 = tensor.extract_slice %arg6[%10, %11] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
        %is_zero = arith.cmpi "eq", %arg2, %idx0 : index
        %ifrun = scf.if %is_zero -> (tensor<16x16xf32>) {
          %expanded = tensor.expand_shape %extracted_slice_0 [[0], [1, 2]] output_shape [16, 16, 16] : tensor<16x256xf32> into tensor<16x16x16xf32>
          %14 = tensor.empty() : tensor<16x16x16xf32>
          %transposed = linalg.transpose ins(%expanded : tensor<16x16x16xf32>) outs(%14 : tensor<16x16x16xf32>) permutation = [1, 0, 2] 
          %expanded_3 = tensor.expand_shape %extracted_slice_1 [[0, 1], [2]] output_shape [16, 16, 16] : tensor<256x16xf32> into tensor<16x16x16xf32>
          %15 = linalg.fill ins(%cst : f32) outs(%extracted_slice_2 : tensor<16x16xf32>) -> tensor<16x16xf32>
          %16 = linalg.batch_reduce_matmul ins(%transposed, %expanded_3 : tensor<16x16x16xf32>, tensor<16x16x16xf32>) outs(%15 : tensor<16x16xf32>) -> tensor<16x16xf32>
          scf.yield %16 : tensor<16x16xf32>
        } else {
            scf.yield %extracted_slice_2 : tensor<16x16xf32>
        }
        %17 = affine.apply #map1(%arg4)
        %18 = affine.apply #map1(%arg5)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %ifrun into %arg6[%17, %18] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
        }
      }
      %9 = affine.apply #map(%arg2)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg3[%9, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x64xf32>
      }
    }
    %2 = tensor.empty() : tensor<128x64xf32>
    %3 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %2) -> (tensor<128x64xf32>) {
      %7 = affine.apply #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg3[%7, 0] [64, 64] [1, 1] : tensor<128x64xf32> to tensor<64x64xf32>
      %8 = scf.forall (%arg4, %arg5) in (4, 4) shared_outs(%arg6 = %extracted_slice) -> (tensor<64x64xf32>) {
        %10 = affine.apply #map1(%arg4)
        %11 = affine.apply #map1(%arg5)
        %12 = affine.apply #map2(%arg4, %arg2)
        %extracted_slice_0 = tensor.extract_slice %arg0[%12, 256] [16, 256] [1, 1] : tensor<128x512xf32> to tensor<16x256xf32>
        %13 = affine.apply #map1(%arg5)
        %extracted_slice_1 = tensor.extract_slice %arg1[256, %13] [256, 16] [1, 1] : tensor<512x64xf32> to tensor<256x16xf32>
        %extracted_slice_2 = tensor.extract_slice %arg6[%10, %11] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
        %is_one = arith.cmpi "eq", %arg2, %idx1 : index
        %ifrun = scf.if %is_one -> (tensor<16x16xf32>) {
          %expanded = tensor.expand_shape %extracted_slice_0 [[0], [1, 2]] output_shape [16, 16, 16] : tensor<16x256xf32> into tensor<16x16x16xf32>
          %14 = tensor.empty() : tensor<16x16x16xf32>
          %transposed = linalg.transpose ins(%expanded : tensor<16x16x16xf32>) outs(%14 : tensor<16x16x16xf32>) permutation = [1, 0, 2] 
          %expanded_3 = tensor.expand_shape %extracted_slice_1 [[0, 1], [2]] output_shape [16, 16, 16] : tensor<256x16xf32> into tensor<16x16x16xf32>
          %15 = linalg.fill ins(%cst : f32) outs(%extracted_slice_2 : tensor<16x16xf32>) -> tensor<16x16xf32>
          %16 = linalg.batch_reduce_matmul ins(%transposed, %expanded_3 : tensor<16x16x16xf32>, tensor<16x16x16xf32>) outs(%15 : tensor<16x16xf32>) -> tensor<16x16xf32>
          scf.yield %16 : tensor<16x16xf32>
        } else {
            scf.yield %extracted_slice_2 : tensor<16x16xf32>
        }
        %17 = affine.apply #map1(%arg4)
        %18 = affine.apply #map1(%arg5)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %ifrun into %arg6[%17, %18] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
        }
      }
      %9 = affine.apply #map(%arg2)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg3[%9, 0] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x64xf32>
      }
    }
    %4 = tensor.empty() : tensor<128x64xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %3 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%5 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<128x64xf32>
    return %6 : tensor<128x64xf32>
  }
}