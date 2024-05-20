// RUN: gc-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<4x256x64xbf16>, %arg1: tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16> {
  // CHECK: linalgx.sigmoid
  %0 = linalgx.sigmoid ins(%arg0 : tensor<4x256x64xbf16>) outs(%arg1 : tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16>
  return %0 : tensor<4x256x64xbf16>
}

// CHECK-LABEL: @mmt2d_vnni
func.func @mmt2d_vnni(%arg0: tensor<256x64xf32>, %arg1: tensor<16x2x8x32x4xf32>, 
                      %arg2: tensor<256x512xf32>) -> tensor<256x512xf32> {
  // CHECK: linalgx.mmt2d_vnni
  %0 = linalgx.mmt2d_vnni ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<16x2x8x32x4xf32>) 
                          outs(%arg2 : tensor<256x512xf32>) -> tensor<256x512xf32>
  return %0 : tensor<256x512xf32>
}

// CHECK-LABEL: @mmt4d_vnni
func.func @mmt4d_vnni(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, 
                      %arg2: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  // CHECK: linalgx.mmt4d_vnni
  %0 = linalgx.mmt4d_vnni ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
                          outs(%arg2 : tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %0 : tensor<2x4x32x32xbf16>
}

// CHECK-LABEL: @multi_batch_matmul
func.func @multi_batch_matmul(%arg0: tensor<13x5x6x128x512xbf16>, %arg1: tensor<13x5x6x512x256xbf16>, 
                              %arg2: tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16> {
  // CHECK: linalgx.multi_batch_matmul
  %0 = linalgx.multi_batch_matmul ins(%arg0, %arg1 : tensor<13x5x6x128x512xbf16>, tensor<13x5x6x512x256xbf16>) 
                                  outs(%arg2 : tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16>
  return %0 : tensor<13x5x6x128x256xbf16>
}
