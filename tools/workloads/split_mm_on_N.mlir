func.func @main_entry(%arg0: tensor<128x512xf32>, %arg1: tensor<512x64xf32>) -> tensor<128x64xf32> attributes {llvm.emit_c_interface} {
    %extracted_slice = tensor.extract_slice %arg1[0, 0] [512, 32] [1, 1] : tensor<512x64xf32> to tensor<512x32xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 32] [512, 32] [1, 1] : tensor<512x64xf32> to tensor<512x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %2 = linalg.matmul {splited = true} ins(%arg0, %extracted_slice : tensor<128x512xf32>, tensor<512x32xf32>) outs(%1 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %3 = tensor.empty() : tensor<128x32xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %5 = linalg.matmul {splited = true} ins(%arg0, %extracted_slice_0 : tensor<128x512xf32>, tensor<512x32xf32>) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %concat = tensor.concat dim(1) %2, %5 : (tensor<128x32xf32>, tensor<128x32xf32>) -> tensor<128x64xf32>
    return %concat : tensor<128x64xf32>
  }