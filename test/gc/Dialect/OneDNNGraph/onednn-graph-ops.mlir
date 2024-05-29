// RUN: gc-opt -split-input-file -canonicalize -verify-diagnostics %s | FileCheck %s

//// Matmul

// CHECK-LABEL: @matmul
func.func @matmul(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x64xbf16>, 
                  %arg2: tensor<64xbf16>) -> tensor<128x64xbf16> {
  // CHECK: onednn_graph.matmul 
  %0 = onednn_graph.matmul %arg0, %arg1, %arg2 
       : (tensor<128x512xbf16>, tensor<512x64xbf16>, tensor<64xbf16>) -> tensor<128x64xbf16>
  return %0 : tensor<128x64xbf16>
}

//// Unary

// CHECK-LABEL: @relu
func.func @relu(%arg0: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  // CHECK: onednn_graph.relu 
  %0 = onednn_graph.relu %arg0 : (tensor<128x512xbf16>) -> tensor<128x512xbf16>
  return %0 : tensor<128x512xbf16>
}

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  // CHECK: onednn_graph.sigmoid 
  %0 = onednn_graph.sigmoid %arg0 : (tensor<128x512xbf16>) -> tensor<128x512xbf16>
  return %0 : tensor<128x512xbf16>
}

// CHECK-LABEL: @type_cast
func.func @type_cast(%arg0: tensor<128x512xbf16>) -> tensor<128x512xf32> {
  // CHECK: onednn_graph.type_cast 
  %0 = onednn_graph.type_cast %arg0 : (tensor<128x512xbf16>) -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}

// CHECK-LABEL: @pow
func.func @pow(%arg0: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  // CHECK: onednn_graph.pow 
  // CHECK-SAME: beta = 2 
  %0 = onednn_graph.pow %arg0 {beta = 2.0 : f32}  : (tensor<128x512xbf16>) -> tensor<128x512xbf16>
  return %0 : tensor<128x512xbf16>
}

//// Binary

// CHECK-LABEL: @add
func.func @add(%arg0: tensor<512x64xbf16>, %arg1: tensor<512x64xbf16>) -> tensor<512x64xbf16> {
  // CHECK: onednn_graph.add 
  %0 = onednn_graph.add %arg0, %arg1 : (tensor<512x64xbf16>, tensor<512x64xbf16>) -> tensor<512x64xbf16>
  return %0 : tensor<512x64xbf16>
}

// CHECK-LABEL: @mul
func.func @mul(%arg0: tensor<512x64xbf16>, %arg1: tensor<64xbf16>) -> tensor<512x64xbf16> {
  // CHECK: onednn_graph.mul 
  %0 = onednn_graph.mul %arg0, %arg1 : (tensor<512x64xbf16>, tensor<64xbf16>) -> tensor<512x64xbf16>
  return %0 : tensor<512x64xbf16>
}

// CHECK-LABEL: @sub
func.func @sub(%arg0: tensor<512x64xbf16>, %arg1: tensor<512x64xbf16>) -> tensor<512x64xbf16> {
  // CHECK: onednn_graph.sub 
  %0 = onednn_graph.sub %arg0, %arg1 : (tensor<512x64xbf16>, tensor<512x64xbf16>) -> tensor<512x64xbf16>
  return %0 : tensor<512x64xbf16>
}

// CHECK-LABEL: @div
func.func @div(%arg0: tensor<512x64xbf16>, %arg1: tensor<64xbf16>) -> tensor<512x64xbf16> {
  // CHECK: onednn_graph.div 
  %0 = onednn_graph.div %arg0, %arg1 : (tensor<512x64xbf16>, tensor<64xbf16>) -> tensor<512x64xbf16>
  return %0 : tensor<512x64xbf16>
}

//// Reduce

// CHECK-LABEL: @reduce_sum_keep_dims
func.func @reduce_sum_keep_dims(%arg0: tensor<64x128x512xbf16>) -> tensor<64x1x512xbf16> {
  // CHECK: onednn_graph.reduce_sum 
  // CHECK-SAME: axes = array<i64: 1>
  // CHECK-SAME: keep_dims = true 
  %0 = onednn_graph.reduce_sum %arg0 {axes = array<i64: 1>, keep_dims = true} : (tensor<64x128x512xbf16>) -> tensor<64x1x512xbf16>
  return %0 : tensor<64x1x512xbf16>
}

// CHECK-LABEL: @reduce_sum_no_axis
func.func @reduce_sum_no_axis(%arg0: tensor<64x128x512xbf16>) -> tensor<64x128x512xbf16> {
  // CHECK: onednn_graph.reduce_sum 
  %0 = onednn_graph.reduce_sum %arg0 : (tensor<64x128x512xbf16>) -> tensor<64x128x512xbf16>
  return %0 : tensor<64x128x512xbf16>
}

// CHECK-LABEL: @reduce_mean_no_keep_dims
func.func @reduce_mean_no_keep_dims(%arg0: tensor<64x128x512xbf16>) -> tensor<128xbf16> {
  // CHECK: onednn_graph.reduce_mean 
  // CHECK-SAME: axes = array<i64: 0, 2>
  %0 = onednn_graph.reduce_mean %arg0 {axes = array<i64: -1, 0>} : (tensor<64x128x512xbf16>) -> tensor<128xbf16>
  return %0 : tensor<128xbf16>
}
