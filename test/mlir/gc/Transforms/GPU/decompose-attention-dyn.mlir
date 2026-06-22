// RUN: gc-opt %s --decomposition --split-input-file | FileCheck %s

// Decompose a tiled, dynamic-shape linalgx.attention and check that the
// generated ops carry the propagated gc.tiling.wg_tile_sizes attribute with
// the reduction (k-loop) step folded into the K2 dimension.

#map = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK-LABEL: func.func @entry
// The accumulator-init fills before the k-loop must NOT get tiling attributes.
// CHECK:         linalg.fill ins
// CHECK-NOT:       gc.tiling
// CHECK:         linalg.fill ins
// CHECK-NOT:       gc.tiling
// CHECK:         linalg.fill ins
// CHECK-NOT:       gc.tiling

// The k-loop with the original 6D wg tile sizes preserved verbatim.
// CHECK:         scf.for
// CHECK:           linalg.transpose
// CHECK-SAME:        gc.tiling.wg_tile_sizes = array<i64: 0, 64>
// QK matmul: iteration (m, k2, k1) -> [128, 64, 0]
// CHECK:           linalg.matmul {{.*}}gc.tiling.wg_tile_sizes = array<i64: 128, 64, 0>
// CHECK:           linalg.reduce
// CHECK-SAME:        gc.tiling.wg_tile_sizes = array<i64: 128, 64>
// CHECK:           linalg.max {{.*}}gc.tiling.wg_tile_sizes = array<i64: 128>
// PV matmul: iteration (m, n, k2) -> [128, 0, 64]
// CHECK:           linalg.matmul {{.*}}gc.tiling.wg_tile_sizes = array<i64: 128, 0, 64>
// CHECK:         } {gc.tiling.level = 1 : i8, gc.tiling.wg_tile_sizes = array<i64: 1, 1, 128, 0, 0, 0>}

// Suffix ops (after the loop) use the 2D (m, n) tile space.
// CHECK:         linalg.broadcast {{.*}}gc.tiling.wg_tile_sizes = array<i64: 128, 0>
// CHECK:         linalg.div {{.*}}gc.tiling.wg_tile_sizes = array<i64: 128, 0>

module @fragment_name attributes {gc.module = {device = {arch = "bmg", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}, kernels = {entry_kernel = {threads = [128, 1, 1]}}}} {
  func.func @entry(%arg0: memref<?x16x?x64xf16>, %arg1: memref<?x16x?x64xf16>, %arg2: memref<?x16x?x64xf16>, %arg3: memref<?x16x?x64xf16>) attributes {gc.num_kernels = 1 : i32} {
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
    %dim_2 = memref.dim %arg2, %c0 : memref<?x16x?x64xf16>
    %dim_3 = memref.dim %arg2, %c2 : memref<?x16x?x64xf16>
    %dim_4 = memref.dim %arg1, %c2 : memref<?x16x?x64xf16>
    %4 = scf.forall (%arg4, %arg5, %arg6) = (0, 0, 0) to (%dim_2, 16, %dim_3) step (1, 1, 128) shared_outs(%arg7 = %3) -> (tensor<?x16x?x64xf16>) {
      %5 = affine.min #map(%arg6)[%dim_3]
      %extracted_slice = tensor.extract_slice %2[%arg4, %arg5, %arg6, 0] [1, 1, %5, 64] [1, 1, 1, 1] : tensor<?x16x?x64xf16> to tensor<1x1x?x64xf16>
      %extracted_slice_5 = tensor.extract_slice %1[%arg4, %arg5, 0, 0] [1, 1, %dim_4, 64] [1, 1, 1, 1] : tensor<?x16x?x64xf16> to tensor<1x1x?x64xf16>
      %extracted_slice_6 = tensor.extract_slice %0[%arg4, %arg5, 0, 0] [1, 1, %dim_4, 64] [1, 1, 1, 1] : tensor<?x16x?x64xf16> to tensor<1x1x?x64xf16>
      %extracted_slice_7 = tensor.extract_slice %arg7[%arg4, %arg5, %arg6, 0] [1, 1, %5, 64] [1, 1, 1, 1] : tensor<?x16x?x64xf16> to tensor<1x1x?x64xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%extracted_slice_7 : tensor<1x1x?x64xf16>) -> tensor<1x1x?x64xf16>
      %7 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.wg_tile_sizes = array<i64: 1, 1, 128, 0, 0, 0>, indexing_maps = [#map1, #map2, #map3, #map4, #map5]} ins(%extracted_slice, %extracted_slice_5, %extracted_slice_6, %cst_0 : tensor<1x1x?x64xf16>, tensor<1x1x?x64xf16>, tensor<1x1x?x64xf16>, f16) outs(%6 : tensor<1x1x?x64xf16>) -> tensor<1x1x?x64xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %arg7[%arg4, %arg5, %arg6, 0] [1, 1, %5, 64] [1, 1, 1, 1] : tensor<1x1x?x64xf16> into tensor<?x16x?x64xf16>
      }
    } {gc.kernel_name = "entry_kernel", gc.tiling.stamp = 1 : i64}
    bufferization.materialize_in_destination %4 in restrict writable %arg3 : (tensor<?x16x?x64xf16>, memref<?x16x?x64xf16>) -> ()
    return
  }
}
