// RUN: gc-opt %s --split-compute-intensive-patterns | FileCheck %s

func.func @mlp(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x64xbf16>, %arg2: tensor<64xbf16>, %arg3: tensor<64x256xbf16>, %arg4: tensor<256xbf16>) -> tensor<128x256xbf16> {
%cst = arith.constant 0.000000e+00 : bf16
%0 = tensor.empty() : tensor<128x64xbf16>
%1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
%2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x64xbf16>) outs(%1 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
%3 = tensor.empty() : tensor<128x64xbf16>
%broadcasted = linalg.broadcast ins(%arg2 : tensor<64xbf16>) outs(%3 : tensor<128x64xbf16>) dimensions = [0] 
%4 = tensor.empty() : tensor<128x64xbf16>
%5 = linalg.add ins(%2, %broadcasted : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%4 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
%6 = tensor.empty() : tensor<128x64xbf16>
%7 = linalg.max ins(%5, %cst_0 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%6 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
%cst_1 = arith.constant 0.000000e+00 : bf16
%8 = tensor.empty() : tensor<128x256xbf16>
%9 = linalg.fill ins(%cst_1 : bf16) outs(%8 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
%10 = linalg.matmul ins(%7, %arg3 : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%9 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
%11 = tensor.empty() : tensor<128x256xbf16>
%broadcasted_2 = linalg.broadcast ins(%arg4 : tensor<256xbf16>) outs(%11 : tensor<128x256xbf16>) dimensions = [0] 
%12 = tensor.empty() : tensor<128x256xbf16>
%13 = linalg.add ins(%10, %broadcasted_2 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%12 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
%cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
%14 = tensor.empty() : tensor<128x256xbf16>
%15 = linalg.max ins(%13, %cst_3 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%14 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
return %15 : tensor<128x256xbf16>
}
