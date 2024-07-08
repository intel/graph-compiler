// RUN: gc-opt --split-input-file --flash-attention-conversion --gc-cpu-pipeline %s | gc-cpu-runner -e main -entry-point-result=void
// | FileCheck --allow-empty

func.func @flash_attention(%arg0: tensor<4x4x384x64xf32>, %arg1: tensor<4x4x384x64xf32>, %arg2: tensor<4x4x384x64xf32>, %arg3: tensor<4x4x384x384xf32>) -> tensor<4x4x384x64xf32> {
    %0 = tensor.empty() : tensor<4x4x384x64xf32>
    %1 = linalgx.scaled_dot_product_attention ins(%arg0, %arg1, %arg2, %arg3: tensor<4x4x384x64xf32>, tensor<4x4x384x64xf32>, tensor<4x4x384x64xf32>, tensor<4x4x384x384xf32>) outs(%0 : tensor<4x4x384x64xf32>)  -> tensor<4x4x384x64xf32>
    return %1 : tensor<4x4x384x64xf32>
}

func.func @main() {
  %cst = arith.constant 4.000000e+00 : f32

  %QKVShape  = tensor.empty() : tensor<4x4x384x64xf32>
  %maskShape  = tensor.empty() : tensor<4x4x384x384xf32>

  %Q = linalg.fill ins(%cst : f32) outs(%QKVShape : tensor<4x4x384x64xf32>) -> tensor<4x4x384x64xf32>
  %K = linalg.fill ins(%cst : f32) outs(%QKVShape : tensor<4x4x384x64xf32>) -> tensor<4x4x384x64xf32>
  %V = linalg.fill ins(%cst : f32) outs(%QKVShape : tensor<4x4x384x64xf32>) -> tensor<4x4x384x64xf32>
  %mask = linalg.fill ins(%cst : f32) outs(%maskShape : tensor<4x4x384x384xf32>) -> tensor<4x4x384x384xf32>

  %out = func.call @flash_attention(%Q, %K, %V, %mask) :
  (tensor<4x4x384x64xf32>, tensor<4x4x384x64xf32>, tensor<4x4x384x64xf32>, tensor<4x4x384x384xf32>)
    -> (tensor<4x4x384x64xf32>)

  %idx = arith.constant 0 : index
  %val = tensor.extract %out[%idx, %idx, %idx, %idx] : tensor<4x4x384x64xf32>
  cpuruntime.printf "output[0, 0, 0, 0]: %f\n" %val : f32

  return
}
// CHECK: output[0, 0, 0]: 4.0
