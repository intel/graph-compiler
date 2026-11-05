// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-grouped-matmul --split-input-file | FileCheck %s

// Without 'max_group_size' the row loop runs over all rows of 'lhs'. Every
// tile covers one expert: its rows are clamped to the expert's group, which is
// read from 'offsets', and the tile itself is a plain linalg.matmul with the
// expert's weight matrix.

#mapL = affine_map<(e, m, n, k) -> (m, k)>
#mapW = affine_map<(e, m, n, k) -> (e, n, k)>
#mapO = affine_map<(e, m, n, k) -> (e)>
#mapR = affine_map<(e, m, n, k) -> (m, n)>

//                Maps of the plain matmul: out[m, n] += lhs[m, k] * wei[n, k].
// CHECK-DAG:   #[[MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @grouped_matmul
// CHECK-SAME:    %[[LHS:[^:]+]]: tensor<?x256xf16>, %[[WEI:[^:]+]]: tensor<8x512x256xf16>, %[[OFFS:[^:]+]]: tensor<8xi32>
// CHECK:         scf.forall (%[[E:[^,]+]], %[[M:[^,]+]], %[[N:[^)]+]]) = (0, 0, 0) to (8, %[[TOTAL:[^,]+]], 512) step (1, [[#TM:]], [[#TN:]])
// CHECK-SAME:      shared_outs(%[[OUT:[^ ]+]] = %{{[^)]+}})
//                  The tile's rows, clamped to the bounds of the row loop.
// CHECK:           %[[TILE:.+]] = affine.min {{.*}}(%[[M]])[%[[TOTAL]]]
//                  Group range: start = e == 0 ? 0 : offsets[e - 1], end = offsets[e].
// CHECK:           %[[FIRST:.+]] = arith.cmpi eq, %[[E]], %{{.+}}
// CHECK:           %[[PREV_IDX:.+]] = arith.select %[[FIRST]], %{{.+}}, %{{.+}}
// CHECK:           %[[PREV_I32:.+]] = tensor.extract %[[OFFS]][%[[PREV_IDX]]]
// CHECK:           %[[PREV:.+]] = arith.index_cast %[[PREV_I32]]
// CHECK:           %[[START:.+]] = arith.select %[[FIRST]], %{{.+}}, %[[PREV]]
// CHECK:           %[[END_I32:.+]] = tensor.extract %[[OFFS]][%[[E]]]
// CHECK:           %[[END:.+]] = arith.index_cast %[[END_I32]]
//                  row0 = min(start + m, end), size = clamp(tile, 0, end - row0).
// CHECK:           %[[ABS:.+]] = arith.addi %[[START]], %[[M]]
// CHECK:           %[[ROW0:.+]] = arith.minsi %[[ABS]], %[[END]]
// CHECK:           %[[AVAIL:.+]] = arith.subi %[[END]], %[[ROW0]]
// CHECK:           %[[CLAMPED:.+]] = arith.minsi %[[TILE]], %[[AVAIL]]
// CHECK:           %[[SIZE:.+]] = arith.maxsi {{.*}}%[[CLAMPED]]
// CHECK:           %[[LHS_T:.+]] = tensor.extract_slice %[[LHS]][%[[ROW0]], 0] [%[[SIZE]], 256] [1, 1]
// CHECK:           %[[OUT_T:.+]] = tensor.extract_slice %[[OUT]][%[[ROW0]], %[[N]]] [%[[SIZE]], [[#TN]]] [1, 1]
// CHECK:           %[[INIT:.+]] = linalg.fill {{.*}}gc.tiling.wg_tile_sizes = array<i64: [[#TM]], [[#TN]]>{{.*}} outs(%[[OUT_T]]
//                  The expert dim of the weights is sliced to 1 and dropped.
// CHECK:           %[[WEI_T:.+]] = tensor.extract_slice %[[WEI]][%[[E]], %[[N]], 0] [1, [[#TN]], 256] [1, 1, 1]
// CHECK-SAME:        : tensor<8x512x256xf16> to tensor<[[#TN]]x256xf16>
//                  The K loop of the plain matmul.
// CHECK:           %[[RES:.+]] = scf.for %[[K:[^ ]+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{[^ ]+}} = %[[INIT]])
// CHECK:             tensor.extract_slice %[[LHS_T]][0, %[[K]]] [%[[SIZE]], 64]
// CHECK:             tensor.extract_slice %[[WEI_T]][0, %[[K]]] {{\[}}[[#TN]], 64]
// CHECK:             linalg.matmul indexing_maps = [#[[MAP_A]], #[[MAP_B]], #[[MAP_C]]]
// CHECK-SAME:          gc.tiling.level = 1
// CHECK-SAME:          gc.tiling.wg_tile_sizes = array<i64: [[#TM]], [[#TN]], 64>
// CHECK:           scf.forall.in_parallel
// CHECK:             tensor.parallel_insert_slice %[[RES]] into %[[OUT]][%{{.+}}, %[[N]]] [%{{.+}}, [[#TN]]] [1, 1]
// CHECK-NOT:     linalgx.grouped_matmul
func.func @grouped_matmul(%lhs: tensor<?x256xf16>, %wei: tensor<8x512x256xf16>, %offs: tensor<8xi32>) -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %total = tensor.dim %lhs, %c0 : tensor<?x256xf16>
  %0 = tensor.empty(%total) : tensor<?x512xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %2 = linalgx.grouped_matmul <{indexing_maps = [#mapL, #mapW, #mapO, #mapR]}>
       ins(%lhs, %wei, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
       outs(%1 : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %2 : tensor<?x512xf16>
}

// -----

// With 'max_group_size' the row loop only runs up to the hint.

#mapL = affine_map<(e, m, n, k) -> (m, k)>
#mapW = affine_map<(e, m, n, k) -> (e, n, k)>
#mapO = affine_map<(e, m, n, k) -> (e)>
#mapR = affine_map<(e, m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @grouped_matmul_max_group_size
// CHECK-SAME:    %{{[^:]+}}: tensor<?x256xf16>, %{{[^:]+}}: tensor<8x512x256xf16>, %{{[^:]+}}: tensor<8xi32>, %[[MAX:[^:]+]]: index
// CHECK:         scf.forall (%{{[^,]+}}, %[[M:[^,]+]], %{{[^)]+}}) = (0, 0, 0) to (8, %[[MAX]], 512) step (1, {{[0-9]+}}, {{[0-9]+}})
// CHECK:           affine.min {{.*}}(%[[M]])[%[[MAX]]]
// CHECK:           linalg.matmul
// CHECK-NOT:     linalgx.grouped_matmul
func.func @grouped_matmul_max_group_size(%lhs: tensor<?x256xf16>, %wei: tensor<8x512x256xf16>, %offs: tensor<8xi32>, %max: index) -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %total = tensor.dim %lhs, %c0 : tensor<?x256xf16>
  %0 = tensor.empty(%total) : tensor<?x512xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %2 = linalgx.grouped_matmul <{indexing_maps = [#mapL, #mapW, #mapO, #mapR]}>
       ins(%lhs, %wei, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
       max_group_size(%max)
       outs(%1 : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %2 : tensor<?x512xf16>
}
