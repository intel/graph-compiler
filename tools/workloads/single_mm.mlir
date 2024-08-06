func.func @main_entry(%arg0: tensor<128x512xf32>, %arg1: tensor<512x64xf32>) -> tensor<128x64xf32> attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x512xf32>, tensor<512x64xf32>) outs(%1 : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}