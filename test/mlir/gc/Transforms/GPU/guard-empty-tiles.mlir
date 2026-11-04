// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-grouped-matmul --guard-empty-tiles --split-input-file --verify-diagnostics | FileCheck %s

// A grouped matmul tile has the rows of its expert's group, which may be
// empty. The whole tile, including the fused fill and the fused relu consumer,
// is skipped when there are no rows, and the output slice is left untouched.

#mapL = affine_map<(e, m, n, k) -> (m, k)>
#mapW = affine_map<(e, m, n, k) -> (e, n, k)>
#mapO = affine_map<(e, m, n, k) -> (e)>
#mapR = affine_map<(e, m, n, k) -> (m, n)>
#id = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @grouped_matmul_relu
// CHECK:         scf.forall (%[[E:[^,]+]], %[[M:[^,]+]], %[[N:[^)]+]]) = (0, 0, 0) to (8, %{{.+}}, 512) step (1, 32, 64)
// CHECK-SAME:      shared_outs(%[[OUT:[^ ]+]] = %{{[^)]+}})
//                  The tile position stays ahead of the guard.
// CHECK:           %[[ROW0:.+]] = arith.minsi
// CHECK:           %[[SIZE:.+]] = arith.maxsi
// CHECK:           %[[HAS_ROWS:.+]] = arith.cmpi sgt, %[[SIZE]], %{{.+}}
// CHECK:           %[[RES:.+]] = scf.if %[[HAS_ROWS]] -> (tensor<?x64xf16>) {
// CHECK:             linalg.fill
// CHECK:             scf.for
// CHECK:               linalg.matmul
// CHECK:             %[[RELU:.+]] = linalg.generic
// CHECK:               arith.maximumf
// CHECK:             scf.yield %[[RELU]]
// CHECK:           } else {
// CHECK:             %[[SAME:.+]] = tensor.extract_slice %[[OUT]][%[[ROW0]], %[[N]]] [%[[SIZE]], 64] [1, 1]
// CHECK:             scf.yield %[[SAME]]
// CHECK:           }
// CHECK:           scf.forall.in_parallel
// CHECK-NEXT:        tensor.parallel_insert_slice %[[RES]] into %[[OUT]][%[[ROW0]], %[[N]]] [%[[SIZE]], 64] [1, 1]
func.func @grouped_matmul_relu(%lhs: tensor<?x256xf16>, %wei: tensor<8x512x256xf16>, %offs: tensor<8xi32>) -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %total = tensor.dim %lhs, %c0 : tensor<?x256xf16>
  %0 = tensor.empty(%total) : tensor<?x512xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %2 = linalgx.grouped_matmul <{indexing_maps = [#mapL, #mapW, #mapO, #mapR]}>
       ins(%lhs, %wei, %offs : tensor<?x256xf16>, tensor<8x512x256xf16>, tensor<8xi32>)
       outs(%1 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %3 = tensor.empty(%total) : tensor<?x512xf16>
  %4 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
       ins(%2 : tensor<?x512xf16>) outs(%3 : tensor<?x512xf16>) {
  ^bb0(%in: f16, %out: f16):
    %m = arith.maximumf %in, %cst : f16
    linalg.yield %m : f16
  } -> tensor<?x512xf16>
  return %4 : tensor<?x512xf16>
}

// -----

// A plain matmul with a dynamic M: the tile's rows are min(32, M - m), which is
// at least 1 inside the loop. No guard.

// CHECK-LABEL: func.func @matmul_dynamic_m
// CHECK:         scf.forall
// CHECK-NOT:       scf.if
// CHECK:           linalg.matmul
// CHECK-NOT:       scf.if
func.func @matmul_dynamic_m(%a: tensor<?x256xf16>, %b: tensor<256x512xf16>) -> tensor<?x512xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %m = tensor.dim %a, %c0 : tensor<?x256xf16>
  %0 = tensor.empty(%m) : tensor<?x512xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x512xf16>) -> tensor<?x512xf16>
  %2 = linalg.matmul ins(%a, %b : tensor<?x256xf16>, tensor<256x512xf16>) outs(%1 : tensor<?x512xf16>) -> tensor<?x512xf16>
  return %2 : tensor<?x512xf16>
}

// -----

// Two outputs that are empty together: one guard covers both.

// CHECK-LABEL: func.func @two_outputs_same_size
// CHECK:         %[[HAS:.+]] = arith.cmpi sgt, %[[N:.+]], %{{.+}}
// CHECK-NOT:     arith.andi
// CHECK:         %[[RES:.+]]:2 = scf.if %[[HAS]] -> (tensor<?xf32>, tensor<?xf32>)
// CHECK:           linalg.fill
// CHECK:           linalg.fill
// CHECK:         } else {
// CHECK:           tensor.extract_slice
// CHECK:           tensor.extract_slice
// CHECK:         scf.forall.in_parallel
// CHECK-NEXT:      tensor.parallel_insert_slice %[[RES]]#0
// CHECK-NEXT:      tensor.parallel_insert_slice %[[RES]]#1
func.func @two_outputs_same_size(%n: index, %o1: tensor<64xf32>, %o2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %c1 = arith.constant 1.0 : f32
  %r:2 = scf.forall (%i) in (1) shared_outs(%s1 = %o1, %s2 = %o2) -> (tensor<64xf32>, tensor<64xf32>) {
    %e1 = tensor.empty(%n) : tensor<?xf32>
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<?xf32>) -> tensor<?xf32>
    %e2 = tensor.empty(%n) : tensor<?xf32>
    %f2 = linalg.fill ins(%c1 : f32) outs(%e2 : tensor<?xf32>) -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %f1 into %s1[0] [%n] [1] : tensor<?xf32> into tensor<64xf32>
      tensor.parallel_insert_slice %f2 into %s2[0] [%n] [1] : tensor<?xf32> into tensor<64xf32>
    }
  }
  return %r#0, %r#1 : tensor<64xf32>, tensor<64xf32>
}

// -----

// Two outputs that are empty under different conditions: skipping the body
// when one is empty would drop the other's result. No guard.

// CHECK-LABEL: func.func @two_outputs_different_sizes
// CHECK-NOT:     scf.if
func.func @two_outputs_different_sizes(%n1: index, %n2: index, %o1: tensor<64xf32>, %o2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %c1 = arith.constant 1.0 : f32
  // expected-warning @below {{output tiles are empty under different conditions, not guarded}}
  %r:2 = scf.forall (%i) in (1) shared_outs(%s1 = %o1, %s2 = %o2) -> (tensor<64xf32>, tensor<64xf32>) {
    %e1 = tensor.empty(%n1) : tensor<?xf32>
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<?xf32>) -> tensor<?xf32>
    %e2 = tensor.empty(%n2) : tensor<?xf32>
    %f2 = linalg.fill ins(%c1 : f32) outs(%e2 : tensor<?xf32>) -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %f1 into %s1[0] [%n1] [1] : tensor<?xf32> into tensor<64xf32>
      tensor.parallel_insert_slice %f2 into %s2[0] [%n2] [1] : tensor<?xf32> into tensor<64xf32>
    }
  }
  return %r#0, %r#1 : tensor<64xf32>, tensor<64xf32>
}

// -----

// A side effect in the body has to run even when the tile is empty. No guard.

// CHECK-LABEL: func.func @side_effect_in_body
// CHECK-NOT:     scf.if
func.func @side_effect_in_body(%n: index, %o: tensor<64xf32>, %flag: memref<i1>) -> tensor<64xf32> {
  %c1 = arith.constant 1.0 : f32
  %true = arith.constant true
  // expected-warning @below {{tile body has side effects, not guarded}}
  %r = scf.forall (%i) in (1) shared_outs(%s = %o) -> (tensor<64xf32>) {
    memref.store %true, %flag[] : memref<i1>
    %e = tensor.empty(%n) : tensor<?xf32>
    %f = linalg.fill ins(%c1 : f32) outs(%e : tensor<?xf32>) -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %f into %s[0] [%n] [1] : tensor<?xf32> into tensor<64xf32>
    }
  }
  return %r : tensor<64xf32>
}

// -----

// Two sizes read from memory with a store in between: the same op on the same
// operands, but not the same value. No guard.

// CHECK-LABEL: func.func @sizes_from_loads
// CHECK-NOT:     scf.if
func.func @sizes_from_loads(%m: memref<index>, %o1: tensor<64xf32>, %o2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %c1 = arith.constant 1.0 : f32
  %c1i = arith.constant 1 : index
  %n1 = memref.load %m[] : memref<index>
  memref.store %c1i, %m[] : memref<index>
  %n2 = memref.load %m[] : memref<index>
  // expected-warning @below {{output tiles are empty under different conditions, not guarded}}
  %r:2 = scf.forall (%i) in (1) shared_outs(%s1 = %o1, %s2 = %o2) -> (tensor<64xf32>, tensor<64xf32>) {
    %e1 = tensor.empty(%n1) : tensor<?xf32>
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<?xf32>) -> tensor<?xf32>
    %e2 = tensor.empty(%n2) : tensor<?xf32>
    %f2 = linalg.fill ins(%c1 : f32) outs(%e2 : tensor<?xf32>) -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %f1 into %s1[0] [%n1] [1] : tensor<?xf32> into tensor<64xf32>
      tensor.parallel_insert_slice %f2 into %s2[0] [%n2] [1] : tensor<?xf32> into tensor<64xf32>
    }
  }
  return %r#0, %r#1 : tensor<64xf32>, tensor<64xf32>
}
