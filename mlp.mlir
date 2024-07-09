// ./llvm-install/bin/mlir-opt --matmul-special-tile-and-fuse --mlir-print-ir-after-all ./mlp.mlir

func.func @tile_matmul(%arg0: tensor<64x32xf32>,
                       %arg1: tensor<32x128xf32>,
                       %arg2: tensor<64x128xf32>,
                       %arg3: tensor<128x16xf32>,
                       %arg4: tensor<64x16xf32>,
                       %arg5: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = tensor.empty() : tensor<64x128xf32>
  %3 = linalg.add ins(%1, %arg2 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%2 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %4 = tensor.empty() : tensor<64x128xf32>
  %5 = linalg.generic { indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"] } ins(%3 : tensor<64x128xf32>) outs(%4 : tensor<64x128xf32>) {
  ^bb0(%in_one : f32, %out_one : f32):
    %c0 = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %in_one, %c0 : f32
    %sel = arith.select %cmp, %in_one, %c0 : f32
    linalg.yield %sel : f32 
  } -> tensor<64x128xf32>
  %6 = tensor.empty() : tensor<64x16xf32>
  %7 = linalg.matmul {matmul_special_tiling = true} ins(%5, %arg3 : tensor<64x128xf32>, tensor<128x16xf32>) outs(%6 : tensor<64x16xf32>) -> tensor<64x16xf32>
  %8 = linalg.sub ins(%7, %arg4 : tensor<64x16xf32>, tensor<64x16xf32>) outs(%arg5 : tensor<64x16xf32>) -> tensor<64x16xf32>
  return %8 : tensor<64x16xf32>
}

