func.func @entry(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16>  attributes {llvm.emit_c_interface}  {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x128xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%1 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %2 : tensor<128x128xbf16>
}