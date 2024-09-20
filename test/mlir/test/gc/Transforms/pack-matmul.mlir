// RUN: gc-opt %s --split-input-file --propagate-layout-on-named-ops --post-process-pack-unpack | FileCheck %s

// -----

// CHECK-LABEL: @single_matmul_f32
func.func @single_matmul_f32(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x32xf32>) outs(%0 : tensor<128x32xf32>) -> tensor<128x32xf32>
  return %2 : tensor<128x32xf32>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: tensor.unpack

// -----

// CHECK-LABEL: @single_matmul_bf16
func.func @single_matmul_bf16(%arg0: tensor<128x64xbf16>, %arg1: tensor<64x32xbf16>) -> tensor<128x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xbf16>, tensor<64x32xbf16>) outs(%0 : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  return %2 : tensor<128x32xbf16>
}
// CHECK-COUNT-2: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: tensor.unpack

// -----

// CHECK-LABEL: @mlp_f32
func.func @mlp_f32(%arg0: tensor<128x16xf32>, %arg1: tensor<16x512xf32>, %arg2: tensor<512x256xf32>, %arg3: tensor<256x128xf32>, %arg4: tensor<512xf32>, %arg5: tensor<256xf32>, %arg6: tensor<128xf32>) -> tensor<128x128xf32> attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %2 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<128x16xf32>, tensor<16x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %3 = tensor.empty() : tensor<128x512xf32>
  %broadcasted = linalg.broadcast ins(%arg4 : tensor<512xf32>) outs(%3 : tensor<128x512xf32>) dimensions = [0]
  %4 = tensor.empty() : tensor<128x512xf32>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x512xf32>, tensor<128x512xf32>) outs(%4 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x512xf32>
  %6 = tensor.empty() : tensor<128x512xf32>
  %7 = linalg.max ins(%5, %cst_0 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %8 = tensor.empty() : tensor<128x256xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %10 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%7, %arg2 : tensor<128x512xf32>, tensor<512x256xf32>) outs(%9 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %11 = tensor.empty() : tensor<128x256xf32>
  %broadcasted_1 = linalg.broadcast ins(%arg5 : tensor<256xf32>) outs(%11 : tensor<128x256xf32>) dimensions = [0]
  %12 = tensor.empty() : tensor<128x256xf32>
  %13 = linalg.add ins(%10, %broadcasted_1 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%12 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
  %14 = tensor.empty() : tensor<128x256xf32>
  %15 = linalg.max ins(%13, %cst_2 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%14 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %16 = tensor.empty() : tensor<128x128xf32>
  %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %18 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%15, %arg3 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %19 = tensor.empty() : tensor<128x128xf32>
  %broadcasted_3 = linalg.broadcast ins(%arg6 : tensor<128xf32>) outs(%19 : tensor<128x128xf32>) dimensions = [0]
  %20 = tensor.empty() : tensor<128x128xf32>
  %21 = linalg.add ins(%18, %broadcasted_3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%20 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  %22 = tensor.empty() : tensor<128x128xf32>
  %23 = linalg.max ins(%21, %cst_4 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%22 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %23 : tensor<128x128xf32>
}
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-COUNT-1: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: tensor.unpack

// -----

// CHECK-LABEL: @mlp_bf16
func.func @mlp_bf16(%arg0: tensor<32x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: tensor<4096x11008xbf16>, %arg3: tensor<11008x4096xbf16>, %arg4: tensor<4096xbf16>, %arg5: tensor<11008xbf16>, %arg6: tensor<4096xbf16>) -> tensor<32x4096xbf16> attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<32x4096xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %2 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<32x4096xbf16>, tensor<4096x4096xbf16>) outs(%1 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %3 = tensor.empty() : tensor<32x4096xbf16>
  %broadcasted = linalg.broadcast ins(%arg4 : tensor<4096xbf16>) outs(%3 : tensor<32x4096xbf16>) dimensions = [0]
  %4 = tensor.empty() : tensor<32x4096xbf16>
  %5 = linalg.add ins(%2, %broadcasted : tensor<32x4096xbf16>, tensor<32x4096xbf16>) outs(%4 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x4096xbf16>
  %6 = tensor.empty() : tensor<32x4096xbf16>
  %7 = linalg.max ins(%5, %cst_0 : tensor<32x4096xbf16>, tensor<32x4096xbf16>) outs(%6 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %8 = tensor.empty() : tensor<32x11008xbf16>
  %9 = linalg.fill ins(%cst : bf16) outs(%8 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %10 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%7, %arg2 : tensor<32x4096xbf16>, tensor<4096x11008xbf16>) outs(%9 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %11 = tensor.empty() : tensor<32x11008xbf16>
  %broadcasted_1 = linalg.broadcast ins(%arg5 : tensor<11008xbf16>) outs(%11 : tensor<32x11008xbf16>) dimensions = [0]
  %12 = tensor.empty() : tensor<32x11008xbf16>
  %13 = linalg.add ins(%10, %broadcasted_1 : tensor<32x11008xbf16>, tensor<32x11008xbf16>) outs(%12 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x11008xbf16>
  %14 = tensor.empty() : tensor<32x11008xbf16>
  %15 = linalg.max ins(%13, %cst_2 : tensor<32x11008xbf16>, tensor<32x11008xbf16>) outs(%14 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %16 = tensor.empty() : tensor<32x4096xbf16>
  %17 = linalg.fill ins(%cst : bf16) outs(%16 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %18 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%15, %arg3 : tensor<32x11008xbf16>, tensor<11008x4096xbf16>) outs(%17 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %19 = tensor.empty() : tensor<32x4096xbf16>
  %broadcasted_3 = linalg.broadcast ins(%arg6 : tensor<4096xbf16>) outs(%19 : tensor<32x4096xbf16>) dimensions = [0]
  %20 = tensor.empty() : tensor<32x4096xbf16>
  %21 = linalg.add ins(%18, %broadcasted_3 : tensor<32x4096xbf16>, tensor<32x4096xbf16>) outs(%20 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x4096xbf16>
  %22 = tensor.empty() : tensor<32x4096xbf16>
  %23 = linalg.max ins(%21, %cst_4 : tensor<32x4096xbf16>, tensor<32x4096xbf16>) outs(%22 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  return %23 : tensor<32x4096xbf16>
}
// CHECK-COUNT-2: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-COUNT-2: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-COUNT-2: tensor.pack
// CHECK-COUNT-1: linalg.generic
// CHECK-NOT: tensor.unpack
