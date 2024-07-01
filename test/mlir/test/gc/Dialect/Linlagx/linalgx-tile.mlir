// RUN: mlir-opt --split-input-file --transform-interpreter --canonicalize %s | FileCheck %s

// CHECK-LABEL: @mm2d_vnni
func.func @mm2d_vnni(%arg0: tensor<256x64xi8>, %arg1: tensor<16x2x8x32x4xi8>,
                      %arg2: tensor<256x512xi32>) -> tensor<256x512xi32> {
  // CHECK: linalgx.mm2d_vnni
  %0 = linalgx.mm2d_vnni ins(%arg0, %arg1 : tensor<256x64xi8>, tensor<16x2x8x32x4xi8>)
                          outs(%arg2 : tensor<256x512xi32>) -> tensor<256x512xi32>
  return %0 : tensor<256x512xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalgx.mm2d_vnni"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mm4d_vnni
func.func @mm4d_vnni(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, 
                      %arg2: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  // CHECK: linalgx.mm4d_vnni
  %0 = linalgx.mm4d_vnni ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
                          outs(%arg2 : tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %0 : tensor<2x4x32x32xbf16>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalgx.mm4d_vnni"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
