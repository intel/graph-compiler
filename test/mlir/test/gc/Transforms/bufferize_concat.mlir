// RUN: gc-opt %s --decompose-ops-for-bufferize --one-shot-bufferize="dialect-filter=tensor,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -cse -split-input-file | FileCheck %s

func.func @decompose_1d_concat(%arg0 : tensor<1xf32>,
                            %arg1 : tensor<2xf32>,
                            %arg2 : tensor<3xf32>,
                            %arg3: tensor<4xf32>) -> tensor<10xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1, %arg2, %arg3 : (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}