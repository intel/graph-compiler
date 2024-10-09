// RUN: gc-opt %s --split-input-file --propagate-layout-on-named-ops --post-process-pack-unpack | FileCheck %s

// -----

// CHECK-LABEL: @matmul_add_plain_activation_f32
func.func @matmul_add_plain_activation_f32(%arg0: tensor<128x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x64xf32>) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %3 = tensor.empty() : tensor<128x64xf32>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<64xf32>) outs(%3 : tensor<128x64xf32>) dimensions = [0]
  %4 = tensor.empty() : tensor<128x64xf32>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x64xf32>, tensor<128x64xf32>) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %5 : tensor<128x64xf32>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK: linalg.add ins(%{{.*}}, %{{.*}} : tensor<{{.*}}x{{.*}}xf32>, tensor<{{.*}}x{{.*}}xf32>) outs(%{{.*}} : tensor<{{.*}}x{{.*}}xf32>) -> tensor<{{.*}}x{{.*}}xf32>
// CHECK-NOT: tensor.unpack

// -----

// CHECK-LABEL: @matmul_add_blocking_activation_f32
func.func @matmul_add_blocking_activation_f32(%arg0: tensor<128x511xf32>, %arg1: tensor<511x255xf32>, %arg2: tensor<255xf32>) -> tensor<128x255xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x255xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x255xf32>) -> tensor<128x255xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x511xf32>, tensor<511x255xf32>) outs(%0 : tensor<128x255xf32>) -> tensor<128x255xf32>
  %3 = tensor.empty() : tensor<128x255xf32>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<255xf32>) outs(%3 : tensor<128x255xf32>) dimensions = [0]
  %4 = tensor.empty() : tensor<128x255xf32>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x255xf32>, tensor<128x255xf32>) outs(%4 : tensor<128x255xf32>) -> tensor<128x255xf32>
  return %5 : tensor<128x255xf32>
}
// CHECK-COUNT-2: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK: linalg.add ins(%{{.*}}, %{{.*}} : tensor<{{.*}}x{{.*}}x{{.*}}x{{.*}}xf32>, tensor<{{.*}}x{{.*}}x{{.*}}x{{.*}}xf32>) outs(%{{.*}} : tensor<{{.*}}x{{.*}}x{{.*}}x{{.*}}xf32>) -> tensor<{{.*}}x{{.*}}x{{.*}}x{{.*}}xf32>
// CHECK-COUNT-1: tensor.unpack

// -----

// CHECK-LABEL: @matmul_add_plain_activation_bf16
func.func @matmul_add_plain_activation_bf16(%arg0: tensor<128x64xbf16>, %arg1: tensor<64x64xbf16>, %arg2: tensor<64xbf16>) -> tensor<128x64xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x64xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xbf16>, tensor<64x64xbf16>) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %3 = tensor.empty() : tensor<128x64xbf16>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<64xbf16>) outs(%3 : tensor<128x64xbf16>) dimensions = [0]
  %4 = tensor.empty() : tensor<128x64xbf16>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%4 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  return %5 : tensor<128x64xbf16>
}
// CHECK-COUNT-2: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK: linalg.add ins(%{{.*}}, %{{.*}} : tensor<{{.*}}x{{.*}}xbf16>, tensor<{{.*}}x{{.*}}xbf16>) outs(%{{.*}} : tensor<{{.*}}x{{.*}}xbf16>) -> tensor<{{.*}}x{{.*}}xbf16>
// CHECK-NOT: tensor.unpack
