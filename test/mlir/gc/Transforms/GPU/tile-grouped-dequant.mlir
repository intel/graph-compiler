// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-grouped-matmul --split-input-file | FileCheck %s --check-prefix=TILE
// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-grouped-matmul --simplify-indexing-maps --vectorize --split-input-file | FileCheck %s --check-prefix=VECTOR

// Dequantization must happen on a K tile, not on the full weight tensor.
// TILE-LABEL: func.func @matmul
// TILE-NOT: linalg.generic
// TILE: scf.forall
// TILE-NOT: linalg.generic
// TILE: scf.for %
// TILE: %[[DEQUANT:.+]] = linalg.generic
// TILE-SAME: tensor<[[#N:]]x[[#K:]]xi4>
// TILE-SAME: tensor<[[#N]]x1xf16>, tensor<[[#N]]x1xi4>
// TILE-SAME: gc.tiling.wg_tile_sizes = array<i64: [[#N]], [[#K]]>
// TILE: linalg.matmul
// TILE-SAME: ins(%{{.*}}, %[[DEQUANT]]
// TILE-NOT: linalg.generic
// TILE: return
// VECTOR-LABEL: func.func @matmul
// VECTOR-NOT: linalg.generic
// VECTOR: scf.for %
// VECTOR: vector.transfer_read {{.*}} : tensor<512x32xf16>, vector<64x1xf16>
// VECTOR: vector.transfer_read {{.*}} : tensor<512x32xi4>, vector<64x1xi4>
// VECTOR: vector.transfer_read {{.*}} : tensor<512x2048xi4>, vector<64x64xi4>
// VECTOR: arith.mulf {{.*}} : vector<64x64xf16>
// VECTOR-NOT: linalg.generic
// VECTOR: return
#weights = affine_map<(n, k) -> (n, k)>
#group = affine_map<(n, k) -> (n, k floordiv 64)>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @matmul(%tokens: tensor<?x2048xf16>, %weights: tensor<512x2048xi4>,
                  %scales: tensor<512x32xf16>, %zps: tensor<512x32xi4>)
    -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %tokens, %c0 : tensor<?x2048xf16>
  %winit = tensor.empty() : tensor<512x2048xf16>
  %w = linalg.generic {
    indexing_maps = [#weights, #group, #group, #weights],
    iterator_types = ["parallel", "parallel"]}
    ins(%weights, %scales, %zps : tensor<512x2048xi4>, tensor<512x32xf16>, tensor<512x32xi4>)
    outs(%winit : tensor<512x2048xf16>) {
  ^bb0(%q: i4, %s: f16, %z: i4, %o: f16):
    %qi = arith.extui %q : i4 to i32
    %zi = arith.extui %z : i4 to i32
    %diff = arith.subi %qi, %zi : i32
    %f = arith.sitofp %diff : i32 to f16
    %v = arith.mulf %f, %s : f16
    linalg.yield %v : f16
  } -> tensor<512x2048xf16>
  %empty = tensor.empty(%m) : tensor<?x512xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<?x512xf16>) -> tensor<?x512xf16>
  %result = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%tokens, %w : tensor<?x2048xf16>, tensor<512x2048xf16>)
    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %result : tensor<?x512xf16>
}

// -----

// The expert dimension is dropped between work-group and sub-group tiling.
// The producer retains that dimension, including its tile-size attribute.
// TILE-LABEL: func.func @grouped_matmul
// TILE-NOT: linalg.generic
// TILE: scf.forall
// TILE-NOT: linalg.generic
// TILE: scf.for %
// TILE: %[[DEQUANT:.+]] = linalg.generic
// TILE-SAME: tensor<1x[[#N:]]x[[#K:]]xi4>
// TILE-SAME: tensor<1x[[#N]]x1xf16>, tensor<1x[[#N]]x1xi4>
// TILE-SAME: gc.tiling.wg_tile_sizes = array<i64: 1, [[#N]], [[#K]]>
// TILE: %[[SLICE:.+]] = tensor.extract_slice %[[DEQUANT]]
// TILE-SAME: tensor<1x[[#N]]x[[#K]]xf16> to tensor<[[#N]]x[[#K]]xf16>
// TILE: linalg.matmul
// TILE-SAME: ins(%{{.*}}, %[[SLICE]]
// TILE-NOT: linalg.generic
// TILE: return
// VECTOR-LABEL: func.func @grouped_matmul
// VECTOR-NOT: linalg.generic
// VECTOR: scf.for %
// VECTOR: vector.transfer_read {{.*}} : tensor<256x512x32xf16>, vector<1x64x1xf16>
// VECTOR: vector.transfer_read {{.*}} : tensor<256x512x32xi4>, vector<1x64x1xi4>
// VECTOR: vector.transfer_read {{.*}} : tensor<256x512x2048xi4>, vector<1x64x64xi4>
// VECTOR: arith.mulf {{.*}} : vector<1x64x64xf16>
// VECTOR-NOT: linalg.generic
// VECTOR: return
#weights = affine_map<(e, n, k) -> (e, n, k)>
#group = affine_map<(e, n, k) -> (e, n, k floordiv 64)>
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#rhs = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#out = affine_map<(e, m, n, k) -> (m, n)>
func.func @grouped_matmul(%tokens: tensor<?x2048xf16>,
                          %weights: tensor<256x512x2048xi4>,
                          %scales: tensor<256x512x32xf16>,
                          %zps: tensor<256x512x32xi4>,
                          %offsets: tensor<256xi32>, %max_group_size: index)
    -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %tokens, %c0 : tensor<?x2048xf16>
  %winit = tensor.empty() : tensor<256x512x2048xf16>
  %w = linalg.generic {
    indexing_maps = [#weights, #group, #group, #weights],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%weights, %scales, %zps : tensor<256x512x2048xi4>, tensor<256x512x32xf16>, tensor<256x512x32xi4>)
    outs(%winit : tensor<256x512x2048xf16>) {
  ^bb0(%q: i4, %s: f16, %z: i4, %o: f16):
    %qi = arith.extui %q : i4 to i32
    %zi = arith.extui %z : i4 to i32
    %diff = arith.subi %qi, %zi : i32
    %f = arith.sitofp %diff : i32 to f16
    %v = arith.mulf %f, %s : f16
    linalg.yield %v : f16
  } -> tensor<256x512x2048xf16>
  %empty = tensor.empty(%m) : tensor<?x512xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<?x512xf16>) -> tensor<?x512xf16>
  %result = linalgx.grouped_matmul <{indexing_maps = [#lhs, #rhs, #offsets, #out]}>
    ins(%tokens, %w, %offsets : tensor<?x2048xf16>, tensor<256x512x2048xf16>, tensor<256xi32>)
    max_group_size(%max_group_size)
    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %result : tensor<?x512xf16>
}

// -----

// A reduction tile may span more than one quantization group.
// TILE-LABEL: func.func @multiple_groups
// TILE-NOT: linalg.generic
// TILE: scf.forall
// TILE-NOT: linalg.generic
// TILE: scf.for %
// TILE: %[[DEQUANT:.+]] = linalg.generic
// TILE-SAME: tensor<64x64xi4>, tensor<64x2xf16>
// TILE-SAME: gc.tiling.wg_tile_sizes = array<i64: 64, 64>
// TILE: linalg.matmul
// TILE-SAME: ins(%{{.*}}, %[[DEQUANT]]
// TILE-NOT: linalg.generic
// TILE: return
// VECTOR-LABEL: func.func @multiple_groups
// VECTOR-NOT: linalg.generic
// VECTOR: scf.for %
// VECTOR: vector.transfer_read {{.*}} : tensor<512x64xf16>, vector<64x2xf16>
// VECTOR: vector.transfer_read {{.*}} : tensor<512x2048xi4>, vector<64x64xi4>
// VECTOR: arith.mulf {{.*}} : vector<64x64xf16>
// VECTOR-NOT: linalg.generic
// VECTOR: return
#weights = affine_map<(n, k) -> (n, k)>
#group = affine_map<(n, k) -> (n, k floordiv 32)>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @multiple_groups(%tokens: tensor<?x2048xf16>, %weights: tensor<512x2048xi4>,
                           %scales: tensor<512x64xf16>) -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %tokens, %c0 : tensor<?x2048xf16>
  %winit = tensor.empty() : tensor<512x2048xf16>
  %w = linalg.generic {
    indexing_maps = [#weights, #group, #weights],
    iterator_types = ["parallel", "parallel"]}
    ins(%weights, %scales : tensor<512x2048xi4>, tensor<512x64xf16>)
    outs(%winit : tensor<512x2048xf16>) {
  ^bb0(%q: i4, %s: f16, %o: f16):
    %qi = arith.extui %q : i4 to i32
    %f = arith.sitofp %qi : i32 to f16
    %v = arith.mulf %f, %s : f16
    linalg.yield %v : f16
  } -> tensor<512x2048xf16>
  %empty = tensor.empty(%m) : tensor<?x512xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<?x512xf16>) -> tensor<?x512xf16>
  %result = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%tokens, %w : tensor<?x2048xf16>, tensor<512x2048xf16>)
    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %result : tensor<?x512xf16>
}

// -----

// Consumer fusion must still reject non-projected input maps.
// TILE-LABEL: func.func @grouped_consumer
// TILE: scf.forall
// TILE-NOT: linalg.generic
// TILE: scf.for %
// TILE: linalg.matmul
// TILE-NOT: linalg.generic
// TILE: scf.forall.in_parallel
// TILE: linalg.generic
// TILE: return
// VECTOR-LABEL: func.func @grouped_consumer
// VECTOR: return
func.func @grouped_consumer(%a: tensor<128x256xf16>, %b: tensor<256x128xf16>)
    -> tensor<128x256xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<128x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<128x128xf16>) -> tensor<128x128xf16>
  %mm = linalg.matmul ins(%a, %b : tensor<128x256xf16>, tensor<256x128xf16>)
    outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  %out = tensor.empty() : tensor<128x256xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n floordiv 2)>,
                     affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm : tensor<128x128xf16>) outs(%out : tensor<128x256xf16>) {
  ^bb0(%v: f16, %o: f16):
    linalg.yield %v : f16
  } -> tensor<128x256xf16>
  return %result : tensor<128x256xf16>
}
