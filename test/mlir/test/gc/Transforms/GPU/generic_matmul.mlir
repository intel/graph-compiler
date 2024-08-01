// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @generic_matmul(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf16>) {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>) outs(%arg2 : memref<8x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.mulf %in, %in_0 : f16
      %1 = arith.addf %out, %0 : f16
      linalg.yield %1 : f16
    }
    return
  }
}