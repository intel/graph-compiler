// module {
//   func.func @entry(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {llvm.emit_c_interface} {
//     %0 = tensor.empty() : tensor<512x512xf32>
//     %cst = arith.constant 2.000000e+00 : f32
//     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x512xf32>) -> tensor<512x512xf32>
//     %2 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<512x512xf32>) -> tensor<512x512xf32>
//     %3 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<512x512xf32>) -> tensor<512x512xf32>
//     %4 = linalg.matmul ins(%2, %3 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%1 : tensor<512x512xf32>) -> tensor<512x512xf32> 
//     return %4 : tensor<512x512xf32>
//   }
// }

module {
  func.func @entry(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> attributes {llvm.emit_c_interface} {
    %0 = onednn_graph.matmul %arg0, %arg1, %arg2 : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
}