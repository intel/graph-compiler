// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-attention --split-input-file | FileCheck %s

// Tiling of a matmul + add with fully dynamic input shapes. The work-group
// tile sizes stay static while the loop bounds are dynamic (tensor.dim).

// CHECK-LABEL: func.func @entry
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//   The iteration domain is built from dynamic dims of the tensor operands:
//   the M and N parallel dims and the K reduction dim.
// CHECK:         %[[M:.+]] = tensor.dim
// CHECK:         %[[K:.+]] = tensor.dim
// CHECK:         %[[N:.+]] = tensor.dim
//   WG loop: static step sizes over dynamic upper bounds.
// CHECK:         scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (%[[M]], %[[N]]) step (256, 512)
//   SG reduction loop with the static k-tile of 16 over the dynamic K dim.
// CHECK:           scf.for %{{.+}} = %[[C0]] to %[[K]] step %[[C16]]
// CHECK:             linalg.matmul {gc.tiling.level = 1
// CHECK:         linalg.add {gc.tiling.level = 0
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
  %5 = linalg.add ins(%4, %1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  bufferization.materialize_in_destination %5 in restrict writable %arg2 : (tensor<?x?xf16>, memref<?x?xf16>) -> ()
  return
}

// -----

// Tiling of attention (SDPA) with dynamic batch and sequence-length dims.
// The head dim (16) and feature dim (64) are static; the parallel WG loop
// uses a static step of 128 over the dynamic sequence length.

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
  %5 = linalgx.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%2, %1, %0, %cst_0 : tensor<?x16x?x64xf16>, tensor<?x16x?x64xf16>, tensor<?x16x?x64xf16>, f16) outs(%4 : tensor<?x16x?x64xf16>) -> tensor<?x16x?x64xf16>
  bufferization.materialize_in_destination %5 in restrict writable %arg3 : (tensor<?x16x?x64xf16>, memref<?x16x?x64xf16>) -> ()
  return
}
