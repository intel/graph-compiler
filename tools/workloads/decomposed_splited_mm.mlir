#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0) -> (d0 * 8)>
#map3 = affine_map<(d0, d1) -> (d0 * 16 + d1 * 64)>
module {
  func.func @main_entry(%arg0: tensor<128x512xf32>, %arg1: tensor<512x32xf32>) -> tensor<128x32xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_idx = arith.constant 0 : index
    %0 = tensor.empty() : tensor<128x32xf32>
    %1 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %0) -> (tensor<128x32xf32>) {
      %is_zero = arith.cmpi "eq", %arg2, %cst_idx : index
      %2 = affine.apply #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg3[%2, 0] [64, 32] [1, 1] : tensor<128x32xf32> to tensor<64x32xf32>
      %3 = scf.forall (%arg4, %arg5) in (4, 4) shared_outs(%arg6 = %extracted_slice) -> (tensor<64x32xf32>) {
        %5 = affine.apply #map1(%arg4)
        %6 = affine.apply #map2(%arg5)
        %7 = affine.apply #map3(%arg4, %arg2)
        %extracted_slice_0 = tensor.extract_slice %arg0[%7, 0] [16, 512] [1, 1] : tensor<128x512xf32> to tensor<16x512xf32>
        %8 = affine.apply #map2(%arg5)
        %extracted_slice_1 = tensor.extract_slice %arg1[0, %8] [512, 8] [1, 1] : tensor<512x32xf32> to tensor<512x8xf32>
        %extracted_slice_2 = tensor.extract_slice %arg6[%5, %6] [16, 8] [1, 1] : tensor<64x32xf32> to tensor<16x8xf32>
        %ifrun = scf.if %is_zero -> (tensor<16x8xf32>) {
            %expanded = tensor.expand_shape %extracted_slice_0 [[0], [1, 2]] output_shape [16, 32, 16] : tensor<16x512xf32> into tensor<16x32x16xf32>
            %9 = tensor.empty() : tensor<32x16x16xf32>
            %transposed = linalg.transpose ins(%expanded : tensor<16x32x16xf32>) outs(%9 : tensor<32x16x16xf32>) permutation = [1, 0, 2] 
            %expanded_3 = tensor.expand_shape %extracted_slice_1 [[0, 1], [2]] output_shape [32, 16, 8] : tensor<512x8xf32> into tensor<32x16x8xf32>
            %10 = linalg.fill ins(%cst : f32) outs(%extracted_slice_2 : tensor<16x8xf32>) -> tensor<16x8xf32>
            %11 = linalg.batch_reduce_matmul ins(%transposed, %expanded_3 : tensor<32x16x16xf32>, tensor<32x16x8xf32>) outs(%10 : tensor<16x8xf32>) -> tensor<16x8xf32>
            scf.yield %11 : tensor<16x8xf32>
        } else {
            scf.yield %extracted_slice_2 : tensor<16x8xf32>
        }
        %12 = affine.apply #map1(%arg4)
        %13 = affine.apply #map2(%arg5)
        scf.forall.in_parallel {
            tensor.parallel_insert_slice %ifrun into %arg6[%12, %13] [16, 8] [1, 1] : tensor<16x8xf32> into tensor<64x32xf32>
        }
      }
      %4 = affine.apply #map(%arg2)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg3[%4, 0] [64, 32] [1, 1] : tensor<64x32xf32> into tensor<128x32xf32>
      }
    }
    return %1 : tensor<128x32xf32>
  }
}