module {
  func.func @entry(%arg0: tensor<3x5xf32>) -> tensor<3xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3xf32>) -> tensor<3xf32>
    %reduce = linalg.reduce { arith.addf }
      ins(%arg0:tensor<3x5xf32>)
      outs(%1:tensor<3xf32>)
      dimensions = [1]
    return %reduce : tensor<3xf32>
  }
}