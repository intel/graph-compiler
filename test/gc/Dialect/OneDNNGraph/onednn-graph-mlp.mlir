// RUN: gc-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @mlp
func.func @mlp(%in: tensor<128x512xbf16>, 
               %weight0: tensor<512x64xbf16>, %bias0: tensor<64xbf16>,
               %weight1: tensor<64x256xbf16>, %bias1: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %0 = onednn_graph.matmul %in, %weight0, %bias0 
       : (tensor<128x512xbf16>, tensor<512x64xbf16>, tensor<64xbf16>) -> tensor<128x64xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %2 = onednn_graph.matmul %1, %weight1
       : (tensor<128x64xbf16>, tensor<64x256xbf16>) -> tensor<128x256xbf16>
  %3 = onednn_graph.add %2, %bias1 : (tensor<128x256xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %4 = onednn_graph.relu %3 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %4 : tensor<128x256xbf16>
}

// CHECK-LABEL: @mlp_transpose_a
func.func @mlp_transpose_a(%in: tensor<512x128xbf16>, 
               %weight0: tensor<512x256xbf16>, %bias0: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %0 = onednn_graph.matmul %in, %weight0, %bias0 {transpose_a = true}  
       : (tensor<512x128xbf16>, tensor<512x256xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}

// CHECK-LABEL: @mlp_transpose_b
func.func @mlp_transpose_b(%in: tensor<128x512xbf16>, 
               %weight0: tensor<256x512xbf16>, %bias0: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %0 = onednn_graph.matmul %in, %weight0, %bias0 {transpose_b = true}  
       : (tensor<128x512xbf16>, tensor<256x512xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}

// CHECK-LABEL: @mlp_transpose_a_b
func.func @mlp_transpose_a_b(%in: tensor<512x128xbf16>, 
               %weight0: tensor<256x512xbf16>, %bias0: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %0 = onednn_graph.matmul %in, %weight0, %bias0 {transpose_a = true, transpose_b = true}  
       : (tensor<512x128xbf16>, tensor<256x512xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}
