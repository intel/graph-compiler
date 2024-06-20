// RUN: gc-opt --split-input-file --flash-attention-conversion %s

func.func @flash_attention(%arg0: tensor<1x16x384x64xf32>, %arg1: tensor<1x16x384x64xf32>, %arg2: tensor<1x16x384x64xf32>, %arg3: tensor<1x16x384x384xf32>) -> tensor<1x16x384x64xf32> {
    %0 = tensor.empty() : tensor<1x16x384x64xf32>
    %1 = linalgx.scaled_dot_product_attention ins(%arg0, %arg1, %arg2, %arg3: tensor<1x16x384x64xf32>, tensor<1x16x384x64xf32>, tensor<1x16x384x64xf32>, tensor<1x16x384x384xf32>) outs(%0 : tensor<1x16x384x64xf32>)  -> tensor<1x16x384x64xf32>
    return %1 : tensor<1x16x384x64xf32>
}
