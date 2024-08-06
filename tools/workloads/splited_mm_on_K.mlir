#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main_entry(%arg0: tensor<128x512xf32>, %arg1: tensor<512x64xf32>) -> tensor<128x64xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [128, 256] [1, 1] : tensor<128x512xf32> to tensor<128x256xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, 256] [128, 256] [1, 1] : tensor<128x512xf32> to tensor<128x256xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, 0] [256, 64] [1, 1] : tensor<512x64xf32> to tensor<256x64xf32>
    %extracted_slice_2 = tensor.extract_slice %arg1[256, 0] [256, 64] [1, 1] : tensor<512x64xf32> to tensor<256x64xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<128x64xf32>
    %3 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %4 = linalg.matmul {splited = true} ins(%extracted_slice, %extracted_slice_1 : tensor<128x256xf32>, tensor<256x64xf32>) outs(%3 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %5 = tensor.empty() : tensor<128x64xf32>
    %6 = linalg.fill ins(%cst_4 : f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %7 = linalg.matmul {splited = true} ins(%extracted_slice_0, %extracted_slice_2 : tensor<128x256xf32>, tensor<256x64xf32>) outs(%6 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %8 = tensor.empty() : tensor<128x64xf32>
    %9 = linalg.fill ins(%cst_5 : f32) outs(%8 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %7 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%9 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %11 = arith.addf %in, %in_6 : f32
      linalg.yield %11 : f32
    } -> tensor<128x64xf32>
    return %10 : tensor<128x64xf32>
  }
}