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

func.func @generalize_mm2d_vnni(%arg0: tensor<256x64xf32>, %arg1: tensor<16x2x8x32x4xf32>, 
                      %arg2: tensor<256x512xf32>) -> tensor<256x512xf32> {
  %0 = linalgx.mm2d_vnni ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<16x2x8x32x4xf32>) 
                          outs(%arg2 : tensor<256x512xf32>) -> tensor<256x512xf32>
  return %0 : tensor<256x512xf32>
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3 * 32 + d4 * 4 + d5)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d2, d5)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 32 + d2)>

// CHECK: func @generalize_mm2d_vnni

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types =  ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<256x64xf32>, tensor<16x2x8x32x4xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<256x512xf32>)

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<256x512xf32>

// -----

func.func @generalize_mm4d_vnni(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, 
                      %arg2: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = linalgx.mm4d_vnni ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
                          outs(%arg2 : tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %0 : tensor<2x4x32x32xbf16>
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2, d5 * 2 + d6)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d3, d6)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// CHECK: func @generalize_mm4d_vnni

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>)
// CHECK-SAME: outs(%{{.+}} : tensor<2x4x32x32xbf16>)

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: bf16, %[[B_ARG:.+]]: bf16, %[[C_ARG:.+]]: bf16)
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : bf16
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : bf16
// CHECK-NEXT:   linalg.yield %[[ADD]] : bf16
// CHECK-NEXT: -> tensor<2x4x32x32xbf16>

// -----

func.func @generalize_batch_reduce_matmul_vnni(%arg0: tensor<512x32x64xbf16>, %arg1: tensor<512x16x128x4xbf16>, 
                      %arg2: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
  %0 = linalgx.batch_reduce_matmul_vnni ins(%arg0, %arg1 : tensor<512x32x64xbf16>, tensor<512x16x128x4xbf16>) 
                          outs(%arg2 : tensor<32x128xbf16>) -> tensor<32x128xbf16>
  return %0 : tensor<32x128xbf16>
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3 * 4 + d4)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1, d4)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>

// CHECK: func @generalize_batch_reduce_matmul_vnni

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types =  ["parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<512x32x64xbf16>, tensor<512x16x128x4xbf16>)
// CHECK-SAME: outs(%{{.+}} : tensor<32x128xbf16>)

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: bf16, %[[B_ARG:.+]]: bf16, %[[C_ARG:.+]]: bf16)
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : bf16
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : bf16
// CHECK-NEXT:   linalg.yield %[[ADD]] : bf16
// CHECK-NEXT: -> tensor<32x128xbf16>

// -----

func.func @generalize_multi_batch_matmul(%arg0: tensor<13x5x6x128x512xbf16>, %arg1: tensor<13x5x6x512x256xbf16>, 
                              %arg2: tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16> {
  %0 = linalgx.multi_batch_matmul ins(%arg0, %arg1 : tensor<13x5x6x128x512xbf16>, tensor<13x5x6x512x256xbf16>) 
                                  outs(%arg2 : tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16>
  return %0 : tensor<13x5x6x128x256xbf16>
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>

// CHECK: func @generalize_multi_batch_matmul

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<13x5x6x128x512xbf16>, tensor<13x5x6x512x256xbf16>)
// CHECK-SAME: outs(%{{.+}} : tensor<13x5x6x128x256xbf16>)

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: bf16, %[[B_ARG:.+]]: bf16, %[[C_ARG:.+]]: bf16)
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : bf16
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : bf16
// CHECK-NEXT:   linalg.yield %[[ADD]] : bf16
// CHECK-NEXT: -> tensor<13x5x6x128x256xbf16>

// -----
