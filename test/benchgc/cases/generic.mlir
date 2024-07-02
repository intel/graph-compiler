#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<3x2xbf16>, %arg1: tensor<2x5xbf16>) -> tensor<3x5xbf16> {
    %0 = tensor.empty() : tensor<3x5xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x2xbf16>, tensor<2x5xbf16>) outs(%0 : tensor<3x5xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    } -> tensor<3x5xbf16>
    return %1 : tensor<3x5xbf16>
  }
}