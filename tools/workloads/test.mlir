func.func @main_entry(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x256xbf16>) -> tensor<128x256xbf16> attributes {llvm.emit_c_interface} {
  %0 = onednn_graph.matmul %arg0, %arg1 : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>
  return %0 : tensor<128x256xbf16>
}