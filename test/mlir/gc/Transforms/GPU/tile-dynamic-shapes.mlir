// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-attention --split-input-file | FileCheck %s


// CHECK-LABEL: func.func @entry
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[M:.+]] = tensor.dim
// CHECK:         %[[K:.+]] = tensor.dim
// CHECK:         %[[N:.+]] = tensor.dim
// CHECK:         scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (%[[M]], %[[N]]) step (32, 64)
// CHECK:           scf.for %{{.+}} = %[[C0]] to %[[K]] step %[[C64]]
// CHECK:             linalg.matmul {gc.tiling.level = 1
// CHECK:           linalg.generic {{.*}}gc.tiling.level = 0
func.func @entry(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<?x?xf16> to tensor<?x?xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<?x?xf16> to tensor<?x?xf16>
  %dim = memref.dim %arg1, %c0 : memref<?x?xf16>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf16>
  %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %4 = linalg.matmul ins(%1, %0 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%4, %1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %out: f16):
      %sum = arith.addf %lhs, %rhs : f16
      linalg.yield %sum : f16
  } -> tensor<?x?xf16>
  bufferization.materialize_in_destination %5 in restrict writable %arg2 : (tensor<?x?xf16>, memref<?x?xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @entry
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[DB:.+]] = tensor.dim %{{.+}}, %[[C0]]
// CHECK:         %[[DS:.+]] = tensor.dim %{{.+}}, %[[C2]]
// CHECK:         scf.forall (%{{.+}}, %{{.+}}, %{{.+}}) = (0, 0, 0) to (%[[DB]], 16, %[[DS]]) step (1, 1, 128)
// CHECK:           linalgx.attention {gc.tiling.level = 1
// CHECK-SAME:        tensor<1x1x?x64xf16>
func.func @entry(%arg0: memref<?x16x?x64xf16>, %arg1: memref<?x16x?x64xf16>, %arg2: memref<?x16x?x64xf16>, %arg3: memref<?x16x?x64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 1.250000e-01 : f16
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<?x16x?x64xf16> to tensor<?x16x?x64xf16>
  %dim = memref.dim %arg0, %c0 : memref<?x16x?x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<?x16x?x64xf16> to tensor<?x16x?x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<?x16x?x64xf16> to tensor<?x16x?x64xf16>
  %dim_1 = memref.dim %arg2, %c2 : memref<?x16x?x64xf16>
  %3 = tensor.empty(%dim, %dim_1) : tensor<?x16x?x64xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<?x16x?x64xf16>) -> tensor<?x16x?x64xf16>
  %5 = linalgx.attention <{indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]}> ins(%2, %1, %0, %cst_0 : tensor<?x16x?x64xf16>, tensor<?x16x?x64xf16>, tensor<?x16x?x64xf16>, f16) outs(%4 : tensor<?x16x?x64xf16>) -> tensor<?x16x?x64xf16>
  bufferization.materialize_in_destination %5 in restrict writable %arg3 : (tensor<?x16x?x64xf16>, memref<?x16x?x64xf16>) -> ()
  return
}

// -----

// The pointwise consumer uses both dynamic results of one tiled contraction.
// Consumer fusion temporarily replaces one loop result with tensor.empty.
// That placeholder needs a dynamic-size operand from its forall initializer;
// constructing it without one creates invalid IR.
// CHECK-LABEL: func.func @dynamic_multi_result_consumer
// CHECK:         %[[FORALL:.+]] = scf.forall
// CHECK:           %[[KLOOP:.+]]:2 = scf.for
// CHECK:             %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:          gc.fused_matmul
// CHECK:           %[[ADD:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[KLOOP]]#0, %[[KLOOP]]#1 :
// CHECK:             tensor.parallel_insert_slice %[[ADD]]
// CHECK:         return %[[FORALL]] : tensor<?x64xf32>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>
#id = affine_map<(m, n) -> (m, n)>
func.func @dynamic_multi_result_consumer(
    %a: tensor<?x256xf16>, %b0: tensor<256x64xf16>,
    %b1: tensor<256x64xf16>, %i0: tensor<?x64xf32>,
    %i1: tensor<?x64xf32>, %sum_init: tensor<?x64xf32>)
    -> tensor<?x64xf32> {
  %r:2 = linalg.generic
      {indexing_maps = [#lhs, #rhs, #rhs, #out, #out],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b0, %b1 : tensor<?x256xf16>, tensor<256x64xf16>, tensor<256x64xf16>)
      outs(%i0, %i1 : tensor<?x64xf32>, tensor<?x64xf32>)
      attrs = {gc.fused_matmul} {
    ^bb0(%av: f16, %b0v: f16, %b1v: f16, %c0v: f32, %c1v: f32):
      %af = arith.extf %av : f16 to f32
      %b0f = arith.extf %b0v : f16 to f32
      %b1f = arith.extf %b1v : f16 to f32
      %p0 = arith.mulf %af, %b0f : f32
      %s0 = arith.addf %c0v, %p0 : f32
      %p1 = arith.mulf %af, %b1f : f32
      %s1 = arith.addf %c1v, %p1 : f32
      linalg.yield %s0, %s1 : f32, f32
  } -> (tensor<?x64xf32>, tensor<?x64xf32>)
  %sum = linalg.generic
      {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%r#0, %r#1 : tensor<?x64xf32>, tensor<?x64xf32>)
      outs(%sum_init : tensor<?x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %add = arith.addf %x, %y : f32
      linalg.yield %add : f32
  } -> tensor<?x64xf32>
  return %sum : tensor<?x64xf32>
}
