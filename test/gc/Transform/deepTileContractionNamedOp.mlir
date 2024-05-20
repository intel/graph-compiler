// RUN: gc-opt --split-input-file --deep-tile-contraction-named-op %s

// -----

/// CHECK-LABEL: @blocked_matmul_f32
func.func @blocked_matmul_f32(%arg0: tensor<128x128x32x32xf32>) -> tensor<128x128x32x32xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128x32x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128x32x32xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x128x32x32xf32>) -> tensor<128x128x32x32xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %cst : tensor<128x128x32x32xf32>, tensor<128x128x32x32xf32>) outs(%1 : tensor<128x128x32x32xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
        %3 = arith.mulf %in, %in_1 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
    } -> tensor<128x128x32x32xf32>
    return %2 : tensor<128x128x32x32xf32>
}

// -----

/// CHECK-LABEL: @plain_matmul_f32
func.func @plain_matmul_f32(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4096x4096xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2 = linalg.matmul ins(%arg0, %cst : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>)  -> tensor<4096x4096xf32>
    return %2 : tensor<4096x4096xf32>
}

// -----

/// CHECK-LABEL: @blocked_matmul_bf16
func.func @blocked_matmul_bf16(%arg0: tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128x16x32x2xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<128x128x32x32xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %cst : tensor<128x128x32x32xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<128x128x32x32xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %3 = arith.mulf %in, %in_1 : bf16
        %4 = arith.addf %out, %3 : bf16
        linalg.yield %4 : bf16
    } -> tensor<128x128x32x32xbf16>
    return %2 : tensor<128x128x32x32xbf16>
}

