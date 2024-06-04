// RUN: gc-opt --split-input-file --deep-tile-contraction-named-op %s

// // -----

// /// CHECK-LABEL: @matmul_4Dx4D_f32
// func.func @matmul_4Dx4D_f32(%arg0: tensor<128x128x32x32xf32>) -> tensor<128x128x32x32xf32> {
//     %cst = arith.constant dense<1.000000e+00> : tensor<128x128x32x32x1xf32>
//     %cst_0 = arith.constant 0.000000e+00 : f32
//     %0 = tensor.empty() : tensor<128x128x32x32xf32>
//     %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x128x32x32xf32>) -> tensor<128x128x32x32xf32>
//     %2 = linalgx.mm4d_vnni ins(%arg0, %cst : tensor<128x128x32x32xf32>, tensor<128x128x32x32x1xf32>) outs(%1 : tensor<128x128x32x32xf32>)  -> tensor<128x128x32x32xf32>
//     return %2 : tensor<128x128x32x32xf32>
// }

// -----

/// CHECK-LABEL: @matmul_2Dx2D_f32
func.func @matmul_2Dx2D_f32(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4096x4096xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2 = linalg.matmul ins(%arg0, %cst : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>)  -> tensor<4096x4096xf32>
    return %2 : tensor<4096x4096xf32>
}

// // -----

// /// CHECK-LABEL: @matmul_2Dx4D_f32
// func.func @matmul_4Dx4D_f32(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
//     %cst = arith.constant dense<1.000000e+00> : tensor<128x128x32x32x1xf32>
//     %cst_0 = arith.constant 0.000000e+00 : f32
//     %0 = tensor.empty() : tensor<4096x4096xf32>
//     %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
//     %2 = linalgx.mm2d_vnni ins(%arg0, %cst : tensor<4096x4096xf32>, tensor<128x128x32x32x1xf32>) outs(%1 : tensor<4096x4096xf32>)  -> tensor<4096x4096xf32>
//     return %2 : tensor<4096x4096xf32>
// }

// -----

/// CHECK-LABEL: @matmul_4Dx4D_bf16
func.func @matmul_4Dx4D_bf16(%arg0: tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128x16x32x2xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x128x32x32xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16>
    %2 = linalgx.mm4d_vnni ins(%arg0, %cst : tensor<128x128x32x32xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<128x128x32x32xbf16>)  -> tensor<128x128x32x32xbf16>
    return %2 : tensor<128x128x32x32xbf16>
}

// // -----

// /// CHECK-LABEL: @matmul_2Dx4D_bf16
// func.func @matmul_4Dx4D_bf16(%arg0: tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16> {
//     %cst = arith.constant dense<1.000000e+00> : tensor<128x128x16x32x2xbf16>
//     %cst_0 = arith.constant 0.000000e+00 : bf16
//     %0 = tensor.empty() : tensor<4096x4096xbf16>
//     %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
//     %2 = linalgx.mm2d_vnni ins(%arg0, %cst : tensor<4096x4096xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<4096x4096xbf16>)  -> tensor<4096x4096xbf16>
//     return %2 : tensor<4096x4096xbf16>
// }

