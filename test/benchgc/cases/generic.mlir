#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x3xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%0 : tensor<3x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}