// RUN: gc-opt --split-input-file --lower-to-tile-vector --CPU-physical-register-pass --mlir-print-ir-after-all -- %s

// CHECK-LABEL: func @add_tensor
// func.func @add_tensor_test0(%arg0: tensor<4x8x1024xf32>, %arg1: tensor<4x8x1024xf32>) -> tensor<4x8x1024xf32> {
//   %0 = tensor.empty() : tensor<4x8x1024xf32>
//   %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8x1024xf32>, tensor<4x8x1024xf32>) outs(%0: tensor<4x8x1024xf32>) -> tensor<4x8x1024xf32>
//   return %1 : tensor<4x8x1024xf32>
// }

// func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
//                    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
//                    -> tensor<512x512xf32> {
//   // Matrix-matrix multiplication.
//   %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
//                           outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

//   // Elementwise addition.
//   %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
//     ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
//     outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

//   // Elementwise max with 0 (ReLU).
//   %c0f = arith.constant 0.0 : f32
//   // expected-remark @below {{elementwise binary}}
//   %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
//     ins(%biased, %c0f : tensor<512x512xf32>, f32)
//     outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
//   func.return %relued : tensor<512x512xf32>
// }

func.func @reduce_keepdim0(%arg0: tensor<16x32x64xf32>) -> tensor<16x1x64xf32> {
  %0 = tensor.empty() : tensor<16x64xf32>
  %reduce = linalg.reduce
      ins(%arg0:tensor<16x32x64xf32>)
      outs(%0:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %1 = arith.addf %out, %in: f32
        linalg.yield %1: f32
      }
  %2 = tensor.expand_shape %reduce [[0],[1, 2]] : tensor<16x64xf32> into tensor<16x1x64xf32>
  return %2 : tensor<16x1x64xf32>
}

