// RUN: gc-opt %s -convert-linalg-to-microkernel -split-input-file | FileCheck %s

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    linalg.batch_reduce_matmul ins(%subview, %subview_11 : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>)
    %subview_12 = memref.subview %alloc_5[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_10, %subview_12 : memref<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %0 = arith.addf %in, %in_14 : f32
      linalg.yield %0 : f32
    }
    %subview_13 = memref.subview %alloc_6[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_10 : memref<32x32xf32>) outs(%subview_13 : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
    }
    memref.dealloc %alloc_10 : memref<32x32xf32>
  }
  return
}

// CHECK-LABEL: simple_brgemm
// CHECK: %[[CST0:.+]] = arith.constant 0 : i64
// CHECK: %[[CST16:.+]] = arith.constant 16 : i64
// CHECK: %[[C:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32> 
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (f32, f32)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----
