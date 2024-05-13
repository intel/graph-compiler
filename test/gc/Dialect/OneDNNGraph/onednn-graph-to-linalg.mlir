// RUN: gc-opt --split-input-file --convert-onednn-graph-to-linalg %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: @matmul
func.func @matmul(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x256xbf16>) -> tensor<128x256xbf16> {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x256xbf16>) outs([[FILLED]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %0 = onednn_graph.matmul %arg0, %arg1 : (tensor<128x512xbf16>, tensor<512x256xbf16>) -> tensor<128x256xbf16>
  return %0 : tensor<128x256xbf16>
}

// CHECK-LABEL: @add
func.func @add(%arg0: tensor<128x256xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: tensor.empty()
  // CHECK: linalg.add
  %0 = onednn_graph.add %arg0, %arg1 : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-LABEL: @add_bcast
func.func @add_bcast(%arg0: tensor<128x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x256xf32> {
  // CHECK: tensor.empty()
  // CHECK: linalg.broadcast
  // CHECK: tensor.empty()
  // CHECK: linalg.add
  %0 = onednn_graph.add %arg0, %arg1 : (tensor<128x256xf32>, tensor<256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-LABEL: @relu
func.func @relu(%arg0: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: arith.constant dense<0.0{{.*}}>
  // CHECK: tensor.empty()
  // CHECK: linalg.max
  %0 = onednn_graph.relu %arg0 : (tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
