module {
  func.func @bf16_layer2(%in: tensor<128x512xbf16>, 
                 %weight0: tensor<512x256xbf16>, %bias0: tensor<256xbf16>, 
                 %weight1: tensor<256x64xbf16>, %bias1: tensor<64xbf16>) -> tensor<128x64xbf16> {
    // layer 0
    %0 = onednn_graph.matmul %in, %weight0, %bias0 : (tensor<128x512xbf16>, tensor<512x256xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
    %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
    // layer 1
    %2 = onednn_graph.matmul %1, %weight1, %bias1 : (tensor<128x256xbf16>, tensor<256x64xbf16>, tensor<64xbf16>) -> tensor<128x64xbf16>
    %3 = onednn_graph.relu %2 : (tensor<128x64xbf16>) -> tensor<128x64xbf16>
    return %3 : tensor<128x64xbf16>
  }

  func.func @bf16_layer5(%in: tensor<128x128xbf16>, 
                 %weight0: tensor<128x128xbf16>, %bias0: tensor<128xbf16>, 
                 %weight1: tensor<128x128xbf16>, %bias1: tensor<128xbf16>, 
                 %weight2: tensor<128x128xbf16>, %bias2: tensor<128xbf16>, 
                 %weight3: tensor<128x128xbf16>, %bias3: tensor<128xbf16>, 
                 %weight4: tensor<128x128xbf16>, %bias4: tensor<128xbf16>) -> tensor<128x128xbf16> {
    // layer 0
    %0 = onednn_graph.matmul %in, %weight0, %bias0 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    %1 = onednn_graph.relu %0 : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    // layer 1
    %2 = onednn_graph.matmul %1, %weight1, %bias1 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    %3 = onednn_graph.relu %2 : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    // layer 2
    %4 = onednn_graph.matmul %3, %weight2, %bias2 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    %5 = onednn_graph.relu %4 : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    // layer 3
    %6 = onednn_graph.matmul %5, %weight3, %bias3 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    %7 = onednn_graph.relu %6 : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    // layer 4
    %8 = onednn_graph.matmul %7, %weight4, %bias4 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    %9 = onednn_graph.relu %8 : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %9 : tensor<128x128xbf16>
  }
}