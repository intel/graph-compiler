// RUN: gc-opt %s --gpu-dev-props --horizontal-fusion --tile-contract --tile-attention --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul_add
// CHECK: scf.forall
// CHECK:   linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM:]], [[#WGN:]]>
// CHECK:   scf.for
// CHECK:     linalg.matmul {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]], [[#WGK:]]>
// CHECK:   linalg.generic {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]]>
func.func @matmul_add(%arg0: tensor<1024x512xf16>, %arg1: tensor<512x1024xf16>) -> tensor<1024x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1024x1024xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x512xf16>, tensor<512x1024xf16>) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%2, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%0 : tensor<1024x1024xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %out: f16):
      %sum = arith.addf %lhs, %rhs : f16
      linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>
  return %3 : tensor<1024x1024xf16>
}

// -----

// CHECK-LABEL: func.func @transpose_matmul_add
// CHECK: scf.forall
// CHECK:   linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM:]], [[#WGN:]]>
// CHECK:   scf.for
// CHECK:     linalg.transpose {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGK:]], [[#WGN]]>
// CHECK:     linalg.matmul {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]], [[#WGK]]>
// CHECK:   linalg.generic {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#WGM]], [[#WGN]]>
func.func @transpose_matmul_add(%arg0: tensor<1024x512xf16>, %argB: tensor<1024x512xf16>) -> tensor<1024x1024xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1024x1024xf16>
  %bt_init = tensor.empty() : tensor<512x1024xf16>
  %bt = linalg.transpose ins(%argB : tensor<1024x512xf16>) outs(%bt_init : tensor<512x1024xf16>) permutation = [1, 0]
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %2 = linalg.matmul ins(%arg0, %bt : tensor<1024x512xf16>, tensor<512x1024xf16>) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%2, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%0 : tensor<1024x1024xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %out: f16):
      %sum = arith.addf %lhs, %rhs : f16
      linalg.yield %sum : f16
  } -> tensor<1024x1024xf16>
  return %3 : tensor<1024x1024xf16>
}

// -----

// CHECK-LABEL: func.func @attention
// CHECK: scf.forall
// CHECK:   linalgx.attention {{.*}}gc.tiling.wg_tile_sizes = array<i64: 1, 1, 128, 0, 0, 0>
#mapQ = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#mapK = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#mapV = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#mapS = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#mapO = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
func.func @attention(%q: tensor<2x16x1024x64xf16>, %k: tensor<2x16x1024x64xf16>, %v: tensor<2x16x1024x64xf16>) -> tensor<2x16x1024x64xf16> {
  %scale = arith.constant 1.250000e-01 : f16
  %cst = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<2x16x1024x64xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<2x16x1024x64xf16>) -> tensor<2x16x1024x64xf16>
  %2 = linalgx.attention
       <{ indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO] }>
       ins(%q, %k, %v, %scale : tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, tensor<2x16x1024x64xf16>, f16)
       outs(%1 : tensor<2x16x1024x64xf16>) -> tensor<2x16x1024x64xf16>
  return %2 : tensor<2x16x1024x64xf16>
}

// -----

// Regular fused generics belong to tile-contract, not tile-grouped-matmul.
// Preserve the regular marker and create one shared reduction loop.
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @fused_matmul_shared_lhs
// CHECK:         %[[OUTER:.+]]:2 = scf.forall
// CHECK:           %[[KLOOP:.+]]:2 = scf.for
// CHECK-SAME:        iter_args(%[[ACC0:[^ ]+]] = %{{.*}}, %[[ACC1:[^ ]+]] = %{{.*}})
// CHECK:             %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:          gc.fused_matmul,
// CHECK-SAME:          gc.tiling.wg_tile_sizes
// CHECK-NEXT:        ^bb0(%[[A:[^:]+]]: f16, %[[B0:[^:]+]]: f16, %[[B1:[^:]+]]: f16, %[[C0:[^:]+]]: f32, %[[C1:[^:]+]]: f32):
// CHECK-DAG:           %[[AF:.+]] = arith.extf %[[A]] : f16 to f32
// CHECK-DAG:           %[[BF0:.+]] = arith.extf %[[B0]] : f16 to f32
// CHECK-DAG:           %[[BF1:.+]] = arith.extf %[[B1]] : f16 to f32
// CHECK-DAG:           %[[P0:.+]] = arith.mulf %[[AF]], %[[BF0]] : f32
// CHECK-DAG:           %[[S0:.+]] = arith.addf %[[C0]], %[[P0]] : f32
// CHECK-DAG:           %[[P1:.+]] = arith.mulf %[[AF]], %[[BF1]] : f32
// CHECK-DAG:           %[[S1:.+]] = arith.addf %[[C1]], %[[P1]] : f32
// CHECK-NEXT:          linalg.yield %[[S0]], %[[S1]] : f32, f32
// CHECK:           scf.forall.in_parallel {
// CHECK-DAG:         tensor.parallel_insert_slice %[[KLOOP]]#0
// CHECK-DAG:         tensor.parallel_insert_slice %[[KLOOP]]#1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[OUTER]]#0, %[[OUTER]]#1
func.func @fused_matmul_shared_lhs(%a: tensor<64x256xf16>, %b0: tensor<256x64xf16>,
                                  %b1: tensor<256x64xf16>, %init0: tensor<64x64xf32>,
                                  %init1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r:2 = linalg.generic
      {indexing_maps = [#lhs, #rhs, #rhs, #out, #out],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b0, %b1 : tensor<64x256xf16>, tensor<256x64xf16>, tensor<256x64xf16>)
      outs(%init0, %init1 : tensor<64x64xf32>, tensor<64x64xf32>)
      attrs = {gc.fused_matmul} {
    ^bb0(%av: f16, %b0v: f16, %b1v: f16, %cv0: f32, %cv1: f32):
      %af = arith.extf %av : f16 to f32
      %b0f = arith.extf %b0v : f16 to f32
      %b1f = arith.extf %b1v : f16 to f32
      %p0 = arith.mulf %af, %b0f : f32
      %s0 = arith.addf %cv0, %p0 : f32
      %p1 = arith.mulf %af, %b1f : f32
      %s1 = arith.addf %cv1, %p1 : f32
      linalg.yield %s0, %s1 : f32, f32
  } -> (tensor<64x64xf32>, tensor<64x64xf32>)
  return %r#0, %r#1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Like a plain matmul, a fused matmul must not pull in a producer that depends
// on a previous matmul: the tanh stays with the first matmul's loop instead of
// being recomputed for every tile of the fused matmul.
#id = affine_map<(d0, d1) -> (d0, d1)>
#fa = affine_map<(m, n, k) -> (m, k)>
#fb = affine_map<(m, n, k) -> (k, n)>
#fc = affine_map<(m, n, k) -> (m, n)>
// CHECK-LABEL: func.func @fused_matmul_no_producer_recompute
// CHECK:         scf.forall
// CHECK:           linalg.matmul
// CHECK:           math.tanh
// CHECK:         scf.forall
// CHECK-NOT:       math.tanh
// CHECK:           gc.fused_matmul
// CHECK-NOT:       math.tanh
// CHECK:         return
func.func @fused_matmul_no_producer_recompute(%x: tensor<1024x1024xf16>, %w0: tensor<1024x1024xf16>,
                                             %w1: tensor<1024x1024xf16>, %w2: tensor<1024x1024xf16>)
    -> (tensor<1024x1024xf16>, tensor<1024x1024xf16>) {
  %cst = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<1024x1024xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %h = linalg.matmul ins(%x, %w0 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%f : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
  %g = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%h : tensor<1024x1024xf16>) outs(%e : tensor<1024x1024xf16>) {
    ^bb0(%a: f16, %o: f16):
      %t = math.tanh %a : f16
      linalg.yield %t : f16
  } -> tensor<1024x1024xf16>
  %r:2 = linalg.generic {indexing_maps = [#fa, #fb, #fb, #fc, #fc],
                         iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%g, %w1, %w2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>, tensor<1024x1024xf16>)
      outs(%f, %f : tensor<1024x1024xf16>, tensor<1024x1024xf16>) attrs = {gc.fused_matmul} {
    ^bb0(%av: f16, %b0v: f16, %b1v: f16, %c0v: f16, %c1v: f16):
      %p0 = arith.mulf %av, %b0v : f16
      %s0 = arith.addf %c0v, %p0 : f16
      %p1 = arith.mulf %av, %b1v : f16
      %s1 = arith.addf %c1v, %p1 : f16
      linalg.yield %s0, %s1 : f16, f16
  } -> (tensor<1024x1024xf16>, tensor<1024x1024xf16>)
  return %r#0, %r#1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>
}

// -----

// Both outputs share the same Z fill, which is fused once per destination.
// A is also a fill with the same indexing-map structure, but its (M, K) tile
// sizes differ from Z's (M, N) sizes. Each copy must inherit its own producer's
// annotation rather than relying on positional or structural matching.
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @fused_matmul_duplicated_z
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[OUTER:.+]]:2 = scf.forall
// CHECK-DAG:       %[[Z0:.+]] = linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#TM:]], [[#TN:]]>{{.*}}ins(%[[ZERO]] : f32)
// CHECK-DAG:       %[[Z1:.+]] = linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#TM]], [[#TN]]>{{.*}}ins(%[[ZERO]] : f32)
// CHECK:           %[[KLOOP:.+]]:2 = scf.for
// CHECK-SAME:        iter_args(%[[ACC0:[^ ]+]] = %[[Z0]], %[[ACC1:[^ ]+]] = %[[Z1]])
// CHECK:             %[[A:.+]] = linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#TM]], [[#TK:]]>
// CHECK:             %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:          ins(%[[A]],
// CHECK-SAME:          gc.fused_matmul
// CHECK-SAME:          gc.tiling.wg_tile_sizes = array<i64: [[#TM]], [[#TN]], [[#TK]]>
// CHECK:           scf.forall.in_parallel {
// CHECK-DAG:         tensor.parallel_insert_slice %[[KLOOP]]#0
// CHECK-DAG:         tensor.parallel_insert_slice %[[KLOOP]]#1
// CHECK:         return %[[OUTER]]#0, %[[OUTER]]#1
func.func @fused_matmul_duplicated_z(%b0: tensor<256x128xf16>,
                                    %b1: tensor<256x128xf16>)
    -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %one = arith.constant 1.0 : f16
  %zero = arith.constant 0.0 : f32
  %a_empty = tensor.empty() : tensor<128x256xf16>
  %a = linalg.fill ins(%one : f16) outs(%a_empty : tensor<128x256xf16>) -> tensor<128x256xf16>
  %z_empty = tensor.empty() : tensor<128x128xf32>
  %z = linalg.fill ins(%zero : f32) outs(%z_empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %r:2 = linalg.generic
      {indexing_maps = [#lhs, #rhs, #rhs, #out, #out],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b0, %b1 : tensor<128x256xf16>, tensor<256x128xf16>, tensor<256x128xf16>)
      outs(%z, %z : tensor<128x128xf32>, tensor<128x128xf32>)
      attrs = {gc.fused_matmul} {
    ^bb0(%av: f16, %b0v: f16, %b1v: f16, %cv0: f32, %cv1: f32):
      %af = arith.extf %av : f16 to f32
      %b0f = arith.extf %b0v : f16 to f32
      %b1f = arith.extf %b1v : f16 to f32
      %p0 = arith.mulf %af, %b0f : f32
      %s0 = arith.addf %cv0, %p0 : f32
      %p1 = arith.mulf %af, %b1f : f32
      %s1 = arith.addf %cv1, %p1 : f32
      linalg.yield %s0, %s1 : f32, f32
  } -> (tensor<128x128xf32>, tensor<128x128xf32>)
  return %r#0, %r#1 : tensor<128x128xf32>, tensor<128x128xf32>
}

// -----

// The sibling matmuls share one tile loop. The pointwise consumer needs a
// transposed slice of the second result, so it must read the completed results
// outside that loop.
#id = affine_map<(m, n) -> (m, n)>
#transpose = affine_map<(m, n) -> (n, m)>
// CHECK-LABEL: func.func @fused_matmul_transposed_consumer
// CHECK:         %[[OUTER:.+]]:2 = scf.forall
// CHECK:           %[[KLOOP:.+]]:2 = scf.for
// CHECK:             %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:          gc.fused_matmul
// CHECK:           scf.forall.in_parallel {
// CHECK-DAG:         tensor.parallel_insert_slice %[[KLOOP]]#0
// CHECK-DAG:         tensor.parallel_insert_slice %[[KLOOP]]#1
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[OUTER]]#0, %[[OUTER]]#1 :
// CHECK:         return %[[ADD]] : tensor<64x64xf32>
func.func @fused_matmul_transposed_consumer(
    %a: tensor<64x256xf16>, %b0: tensor<256x64xf16>,
    %b1: tensor<256x64xf16>, %i0: tensor<64x64xf32>,
    %i1: tensor<64x64xf32>, %sum_init: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %r0 = linalg.matmul ins(%a, %b0 : tensor<64x256xf16>, tensor<256x64xf16>)
      outs(%i0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<64x256xf16>, tensor<256x64xf16>)
      outs(%i1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %sum = linalg.generic
      {indexing_maps = [#id, #transpose, #id],
       iterator_types = ["parallel", "parallel"]}
      ins(%r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%sum_init : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %add = arith.addf %x, %y : f32
      linalg.yield %add : f32
  } -> tensor<64x64xf32>
  return %sum : tensor<64x64xf32>
}
