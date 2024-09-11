// RUN: gc-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<4x256x64xbf16>, %arg1: tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16> {
  // CHECK: linalgx.sigmoid
  %0 = linalgx.sigmoid ins(%arg0 : tensor<4x256x64xbf16>) outs(%arg1 : tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16>
  return %0 : tensor<4x256x64xbf16>
}
