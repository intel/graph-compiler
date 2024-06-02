// module {
// func.func @aaa() -> tensor<128xf32> {
//     %c2 = arith.constant 2.0 : f32
//     %a = tensor.empty() : tensor<128xf32>
//     %2 = linalg.fill ins(%c2 : f32) outs(%a : tensor<128xf32>) -> tensor<128xf32>
//     return %2 : tensor<128xf32>
// }

// func.func @main_entry() attributes {llvm.emit_c_interface}  {
//     %result = call @aaa() : ()-> tensor<128xf32>
//     %c0 = arith.constant 0 : index
//     %c128 = arith.constant 128 : index
//     %c1 = arith.constant 1 : index
//     scf.for %iv = %c0 to %c128 step %c1 {
//         %4 = tensor.extract %result[%iv] : tensor<128xf32>
//         // cpuruntime.printf "%f\n" %4 : f32
//     }
//     return
// }
// // CHECK-COUNT-128: 2.000000
// }

// module {
//   func.func @main_entry(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {llvm.emit_c_interface} {
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
  func.func @main_entry(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> attributes {llvm.emit_c_interface} {
    %0 = onednn_graph.matmul %arg0, %arg1, %arg2 : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
}