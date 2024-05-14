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

// CHECK-LABEL: @matmul_bias
func.func @matmul_bias(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x256xbf16>, %arg3: tensor<256xbf16>) -> tensor<128x256xbf16> {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x256xbf16>) outs([[FILLED]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %0 = onednn_graph.matmul %arg0, %arg1, %arg3 : (tensor<128x512xbf16>, tensor<512x256xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
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
func.func @relu(%arg0: tensor<128x256xbf16>) -> tensor<128x256xbf16> {
  // CHECK: arith.constant dense<0.0{{.*}}>
  // CHECK: tensor.empty()
  // CHECK: linalg.max
  %0 = onednn_graph.relu %arg0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %0 : tensor<128x256xbf16>
}

// CHECK-LABEL: @mlp_transpose_a
func.func @mlp_transpose_a(%arg0: tensor<512x128xbf16>, %arg1: tensor<512x256xbf16>, %arg3: tensor<256xbf16>) -> tensor<128x256xbf16> {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<512x128xbf16>, tensor<512x256xbf16>) outs([[FILLED]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: tensor.empty()
  // CHECK: linalg.broadcast
  // CHECK: tensor.empty()
  // CHECK: linalg.add
  // CHECK: arith.constant dense<0.0{{.*}}>
  // CHECK: tensor.empty()
  // CHECK: linalg.max
  %0 = onednn_graph.matmul %arg0, %arg1, %arg3 {transpose_a = true}  
       : (tensor<512x128xbf16>, tensor<512x256xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}

// CHECK-LABEL: @mlp_transpose_b
func.func @mlp_transpose_b(%arg0: tensor<128x512xbf16>, %arg1: tensor<256x512xbf16>, %arg3: tensor<256xbf16>) -> tensor<128x256xbf16> {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<256x512xbf16>) outs([[FILLED]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: tensor.empty()
  // CHECK: linalg.broadcast
  // CHECK: tensor.empty()
  // CHECK: linalg.add
  // CHECK: arith.constant dense<0.0{{.*}}>
  // CHECK: tensor.empty()
  // CHECK: linalg.max
  %0 = onednn_graph.matmul %arg0, %arg1, %arg3 {transpose_b = true}  
       : (tensor<128x512xbf16>, tensor<256x512xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}

// CHECK-LABEL: @mlp_transpose_a_b
func.func @mlp_transpose_a_b(%arg0: tensor<512x128xbf16>, %arg1: tensor<256x512xbf16>, %arg3: tensor<256xbf16>) -> tensor<128x256xbf16> {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT0:%.+]] = tensor.empty()
  // CHECK: [[FILLED0:%.+]] = linalg.fill ins([[C0]] : bf16) outs([[INIT0]] : tensor<256x128xbf16>) -> tensor<256x128xbf16>
  // CHECK: [[MMT:%.+]] = linalg.matmul ins(%arg1, %arg0 : tensor<256x512xbf16>, tensor<512x128xbf16>) outs([[FILLED0]] : tensor<256x128xbf16>) -> tensor<256x128xbf16>
  // CHECK: [[C1:%.+]] = arith.constant 0
  // CHECK: [[INIT1:%.+]] = tensor.empty()
  // CHECK: [[FILLED1:%.+]] = linalg.fill ins([[C1]] : bf16) outs([[INIT1]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  // CHECK: linalg.transpose ins([[MMT]] : tensor<256x128xbf16>) outs([[FILLED1]] : tensor<128x256xbf16>)
  // CHECK: tensor.empty()
  // CHECK: linalg.broadcast
  // CHECK: tensor.empty()
  // CHECK: linalg.add
  // CHECK: arith.constant dense<0.0{{.*}}>
  // CHECK: tensor.empty()
  // CHECK: linalg.max
  %0 = onednn_graph.matmul %arg0, %arg1, %arg3 {transpose_a = true, transpose_b = true}  
       : (tensor<512x128xbf16>, tensor<256x512xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}
