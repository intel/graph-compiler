// RUN: gc-opt -split-input-file -linalg-generalize-named-ops -verify-diagnostics %s | FileCheck %s

func.func @generalize_sigmoid(%arg0: tensor<4x256x64xbf16>, %arg1: tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16> {
  %0 = linalgx.sigmoid ins(%arg0 : tensor<4x256x64xbf16>) outs(%arg1 : tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16>
  return %0 : tensor<4x256x64xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_sigmoid
// CHECK-SAME: (%[[ARG:.+]]: tensor<4x256x64xbf16>, %[[OUT:.+]]: tensor<4x256x64xbf16>)

// CHECK: %[[CST:.+]] = arith.constant 1
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[ARG]] : tensor<4x256x64xbf16>) outs(%[[OUT]] : tensor<4x256x64xbf16>)

// CHECK:         ^{{.*}}(%[[BBARG0:.+]]: bf16, %[[BBARG1:.+]]: bf16)
// CHECK-NEXT:      %[[NEG:.+]] = arith.negf %[[BBARG0]] : bf16
// CHECK-NEXT:      %[[EXP:.+]] = math.exp %[[NEG]] : bf16
// CHECK-NEXT:      %[[ADD:.+]] = arith.addf %[[EXP]], %[[CST]] : bf16
// CHECK-NEXT:      %[[DIV:.+]] = arith.divf %[[CST]], %[[ADD]] : bf16
// CHECK-NEXT:      linalg.yield %[[DIV]] : bf16

// -----

func.func @generalize_mmt2d_vnni(%arg0: tensor<256x64xf32>, %arg1: tensor<16x2x8x32x4xf32>, 
                      %arg2: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %0 = linalgx.mmt2d_vnni ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<16x2x8x32x4xf32>) 
                          outs(%arg2 : tensor<256x512xf32>) -> tensor<256x512xf32>
  return %0 : tensor<256x512xf32>
}

// -----

func.func @generalize_mmt4d_vnni(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, 
                      %arg2: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = linalgx.mmt4d_vnni ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
                          outs(%arg2 : tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %0 : tensor<2x4x32x32xbf16>
}

// -----

func.func @generalize_multi_batch_matmul(%arg0: tensor<13x5x6x128x512xbf16>, %arg1: tensor<13x5x6x512x256xbf16>, 
                              %arg2: tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16> {
  %0 = linalgx.multi_batch_matmul ins(%arg0, %arg1 : tensor<13x5x6x128x512xbf16>, tensor<13x5x6x512x256xbf16>) 
                                  outs(%arg2 : tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16>
  return %0 : tensor<13x5x6x128x256xbf16>
}

// -----
