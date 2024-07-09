// RUN: gc-opt --gc-gpu-pipeline %s | FileCheck %s

func.func @mlp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x16xf32>) 
                                 outs(%1 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %3 = tensor.empty() : tensor<8x16xf32>
  %4 = linalg.add ins(%arg2, %2 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %4 : tensor<8x16xf32> 
}
