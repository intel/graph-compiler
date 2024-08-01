module @smoke_test {
func.func @main() {
  %0 = arith.constant dense<1.0> : tensor<32xf32>
  %1 = arith.constant dense<2.0> : tensor<32xf32>
  %2 = arith.constant dense<0.1> : tensor<32xf32>
  %3 = call @add(%0, %1, %2) : (tensor<32xf32>,tensor<32xf32>,tensor<32xf32>) -> tensor<32xf32>
  %4 = tensor.cast %3 : tensor<32xf32> to tensor<*xf32>
  call @printMemrefF32(%4) : (tensor<*xf32>) -> ()
  return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @add(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {                                                                                                                                                  
  %1 = linalg.add ins(%arg0, %arg1: tensor<32xf32>, tensor<32xf32>) outs(%dst: tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}
}