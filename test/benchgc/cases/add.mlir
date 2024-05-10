module {
  func.func @f32(%arg0: tensor<128x256xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %0 = onednn_graph.add %arg0, %arg1 : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
}
