// RUN: gc-opt --split-input-file --deep-tile-contraction-named-op %s | FileCheck %s


func.func @test(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, %arg2: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d1, d5 * 2 + d6)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d4, d5, d3, d6)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d1, d3)>], 
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
          } 
          ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
          outs(%arg2 : tensor<2x4x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<2x4x32x32xbf16>
    return %0: tensor<2x4x32x32xbf16>
}
