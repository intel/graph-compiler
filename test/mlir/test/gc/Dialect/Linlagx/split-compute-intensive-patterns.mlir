// RUN: gc-opt %s --split-compute-intensive-patterns | FileCheck %s
func.func @basic_mlp(%in: tensor<128x512xbf16>,
               %weight: tensor<512x256xbf16>,
               %offset: tensor<128x256xbf16>,
               %scale: tensor<128x256xbf16>,
               %weight2: tensor<256x1024xbf16>) -> tensor<128x1024xbf16> {
  %0 = tensor.empty() : tensor<128x256xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %2 = linalg.matmul ins(%in, %weight : tensor<128x512xbf16>, tensor<512x256xbf16>) outs(%1 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %3 = tensor.empty() : tensor<128x256xbf16>
  %4 = linalg.add ins(%2, %offset : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%3 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %5 = tensor.empty() : tensor<128x256xbf16>
  %6 = linalg.mul ins(%4, %scale : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%5 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %9 = tensor.empty() : tensor<128x256xbf16>
  %10 = linalg.max ins(%6, %1 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%9 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %11 = tensor.empty() : tensor<128x1024xbf16>
  %12 = linalg.fill ins(%cst : bf16) outs(%11 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  %13 = linalg.matmul ins(%10, %weight2 : tensor<128x256xbf16>, tensor<256x1024xbf16>) outs(%12 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  return %13 : tensor<128x1024xbf16>
}

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

func.func @mlp_transpose_a_b(%arg0: tensor<512x128xbf16>, %arg1: tensor<256x512xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<256x128xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<256x128xbf16>) -> tensor<256x128xbf16>
    %2 = linalg.matmul ins(%arg1, %arg0 : tensor<256x512xbf16>, tensor<512x128xbf16>) outs(%1 : tensor<256x128xbf16>) -> tensor<256x128xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %3 = tensor.empty() : tensor<128x256xbf16>
    %4 = linalg.fill ins(%cst_0 : bf16) outs(%3 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %transposed = linalg.transpose ins(%2 : tensor<256x128xbf16>) outs(%4 : tensor<128x256xbf16>) permutation = [1, 0] 
    %5 = tensor.empty() : tensor<128x256xbf16>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<256xbf16>) outs(%5 : tensor<128x256xbf16>) dimensions = [0] 
    %6 = tensor.empty() : tensor<128x256xbf16>
    %7 = linalg.add ins(%transposed, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%6 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
    %8 = tensor.empty() : tensor<128x256xbf16>
    %9 = linalg.max ins(%7, %cst_1 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%8 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
    return %9 : tensor<128x256xbf16>
}
