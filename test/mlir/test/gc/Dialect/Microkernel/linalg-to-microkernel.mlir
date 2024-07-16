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

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    linalgx.batch_reduce_matmul_vnni ins(%subview, %subview_11 : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>)
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
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
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
// CHECK-NOT: linalg.fill
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0, stride) data_type = (f32, f32)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
    linalgx.batch_reduce_matmul_vnni ins(%subview, %subview_11 : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>)
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
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CHECK-NOT: linalg.fill
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0, stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x32x256xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x512xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xbf16>
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x2xbf16>
    %subview = memref.subview %alloc_1[%arg7, 0, 0] [1, 32, 256] [1, 1, 1] : memref<4x32x256xbf16> to memref<32x256xbf16, strided<[256, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0] [1, 16, 512] [1, 1, 1] : memref<8x16x512xbf16> to memref<16x512xbf16, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0], [1, 2]] output_shape [32, 8, 32] : memref<32x256xbf16, strided<[256, 1], offset: ?>> into memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>
    linalg.transpose ins(%expand_shape : memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>) outs(%alloc_101 : memref<8x32x32xbf16>) permutation = [1, 0, 2]
    %expand_shape_2 = memref.expand_shape %subview_11 [[0], [1, 2, 3]] output_shape [16, 8, 32, 2] : memref<16x512xbf16, strided<[512, 1], offset: ?>> into memref<16x8x32x2xbf16, strided<[512, 64, 2, 1], offset: ?>>
    linalg.transpose ins(%expand_shape_2 : memref<16x8x32x2xbf16, strided<[512, 64, 2, 1], offset: ?>>) outs(%alloc_102 : memref<8x16x32x2xbf16>) permutation = [1, 0, 2, 3]
    
    linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
    linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
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
// CHECK: scf.forall (%[[arg0:.+]], %[[arg1:.+]]) in (4, 8)
// CHECK: %[[subviewA:.+]] = memref.subview %[[allocA:.+]][%[[arg0]], 0, 0] [1, 32, 256] [1, 1, 1]
// CHECK: %[[subviewB:.+]] = memref.subview %[[allocB:.+]][%[[arg1]], 0, 0] [1, 16, 512] [1, 1, 1]
// CHECK: %[[exsA:.+]] = memref.expand_shape %[[subviewA]]
// CHECK-NOT: linalg.transpose
// CHECK: %[[exsB:.+]] = memref.expand_shape %[[subviewB]]
// CHECK-NOT: linalg.transpose

// CHECK-NOT: linalg.fill
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 256, 256, 32, 32, 64] flags = (beta_0, stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[exsA]], %[[exsB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x32x256xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x512xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xbf16>
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x2xbf16>
    %subview = memref.subview %alloc_1[%arg7, 0, 0] [1, 32, 256] [1, 1, 1] : memref<4x32x256xbf16> to memref<32x256xbf16, strided<[256, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0] [1, 16, 512] [1, 1, 1] : memref<8x16x512xbf16> to memref<16x512xbf16, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0], [1, 2]] output_shape [32, 8, 32] : memref<32x256xbf16, strided<[256, 1], offset: ?>> into memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>
    linalg.transpose ins(%expand_shape : memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>) outs(%alloc_101 : memref<8x32x32xbf16>) permutation = [1, 0, 2]
    %expand_shape_2 = memref.expand_shape %subview_11 [[0], [1, 2, 3]] output_shape [16, 8, 32, 2] : memref<16x512xbf16, strided<[512, 1], offset: ?>> into memref<16x8x32x2xbf16, strided<[512, 64, 2, 1], offset: ?>>
    linalg.transpose ins(%expand_shape_2 : memref<16x8x32x2xbf16, strided<[512, 64, 2, 1], offset: ?>>) outs(%alloc_102 : memref<8x16x32x2xbf16>) permutation = [1, 0, 2, 3]
    %first = arith.cmpi eq, %arg7, %c0 : index
    scf.if %first {    
      linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
      linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
    } else {
      linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
    }
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
// CHECK: scf.forall (%[[arg0:.+]], %[[arg1:.+]]) in (4, 8)
// CHECK: %[[subviewA:.+]] = memref.subview %[[allocA:.+]][%[[arg0]], 0, 0] [1, 32, 256] [1, 1, 1]
// CHECK: %[[subviewB:.+]] = memref.subview %[[allocB:.+]][%[[arg1]], 0, 0] [1, 16, 512] [1, 1, 1]
// CHECK: %[[exsA:.+]] = memref.expand_shape %[[subviewA]]
// CHECK-NOT: linalg.transpose
// CHECK: %[[exsB:.+]] = memref.expand_shape %[[subviewB]]
// CHECK-NOT: linalg.transpose

// CHECK: scf.if
// CHECK-NOT: linalg.fill
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 256, 256, 32, 32, 64] flags = (beta_0, stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[exsA]], %[[exsB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()
// CHECK: else
// CHECK: %[[DIS2:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 256, 256, 32, 32, 64] flags = (stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS2]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS2]], %[[exsA]], %[[exsB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS2]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_bf16 = arith.constant 0.000000e+00 : bf16
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x32x256xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x512xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xbf16>
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x2xbf16>
    linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_10 : memref<32x32xf32>)
    linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_10 : memref<32x32xf32>)
    %subview = memref.subview %alloc_1[%arg7, 0, 0] [1, 32, 256] [1, 1, 1] : memref<4x32x256xbf16> to memref<32x256xbf16, strided<[256, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0] [1, 16, 512] [1, 1, 1] : memref<8x16x512xbf16> to memref<16x512xbf16, strided<[512, 1], offset: ?>>
    %first = arith.cmpi eq, %arg7, %c0 : index
    scf.if %first {    
      %expand_shape = memref.expand_shape %subview [[0], [1, 2]] output_shape [32, 8, 32] : memref<32x256xbf16, strided<[256, 1], offset: ?>> into memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>
      linalg.transpose ins(%expand_shape : memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>) outs(%alloc_101 : memref<8x32x32xbf16>) permutation = [1, 0, 2]
      linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
      linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
    } else {
      linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
    }
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
// CHECK: scf.forall (%[[arg0:.+]], %[[arg1:.+]]) in (4, 8)
// CHECK: %[[allocA:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xbf16>
// CHECK: %[[allocB:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x2xbf16>
// CHECK: %[[subviewA:.+]] = memref.subview %[[allocTransA:.+]][%[[arg0]], 0, 0] [1, 32, 256] [1, 1, 1]

// CHECK: scf.if
// CHECK: %[[exsA:.+]] = memref.expand_shape %[[subviewA]]
// CHECK: linalg.transpose
// CHECK-NOT: linalg.fill
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0, stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[allocA]], %[[allocB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()
// CHECK: else
// CHECK: %[[DIS2:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS2]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS2]], %[[allocA]], %[[allocB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS2]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_brgemm() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_bf16 = arith.constant 0.000000e+00 : bf16
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x32x256xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x512xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xbf16>
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x2xbf16>
    %subview = memref.subview %alloc_1[%arg7, 0, 0] [1, 32, 256] [1, 1, 1] : memref<4x32x256xbf16> to memref<32x256xbf16, strided<[256, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0] [1, 16, 512] [1, 1, 1] : memref<8x16x512xbf16> to memref<16x512xbf16, strided<[512, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0], [1, 2]] output_shape [32, 8, 32] : memref<32x256xbf16, strided<[256, 1], offset: ?>> into memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>
    linalg.transpose ins(%expand_shape : memref<32x8x32xbf16, strided<[256, 32, 1], offset: ?>>) outs(%alloc_101 : memref<8x32x32xbf16>) permutation = [1, 0, 2]
    %expand_shape_2 = memref.expand_shape %subview_11 [[0], [1, 2, 3]] output_shape [16, 8, 32, 2] : memref<16x512xbf16, strided<[512, 1], offset: ?>> into memref<16x8x32x2xbf16, strided<[512, 64, 2, 1], offset: ?>>
    linalg.transpose ins(%expand_shape_2 : memref<16x8x32x2xbf16, strided<[512, 64, 2, 1], offset: ?>>) outs(%alloc_102 : memref<8x16x32x2xbf16>) permutation = [1, 0, 2, 3]
    %first = arith.cmpi eq, %arg7, %c0 : index
    scf.if %first {    
      linalg.fill ins(%cst : f32) outs(%alloc_10 : memref<32x32xf32>)
      linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
    } else {
      linalgx.batch_reduce_matmul_vnni ins(%alloc_101, %alloc_102 : memref<8x32x32xbf16>, memref<8x16x32x2xbf16>) outs(%alloc_10 : memref<32x32xf32>)
    }
    linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_101 : memref<8x32x32xbf16>)
    linalg.fill ins(%cst_bf16 : bf16) outs(%alloc_102 : memref<8x16x32x2xbf16>)
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
// CHECK: scf.forall (%[[arg0:.+]], %[[arg1:.+]]) in (4, 8)
// CHECK: %[[allocTransA:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32xbf16>
// CHECK: %[[allocTransB:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x2xbf16>
// CHECK: %[[subviewA:.+]] = memref.subview %[[allocA:.+]][%[[arg0]], 0, 0] [1, 32, 256] [1, 1, 1]
// CHECK: %[[subviewB:.+]] = memref.subview %[[allocB:.+]][%[[arg1]], 0, 0] [1, 16, 512] [1, 1, 1]
// CHECK: %[[exsA:.+]] = memref.expand_shape %[[subviewA]]
// CHECK: linalg.transpose
// CHECK: %[[exsB:.+]] = memref.expand_shape %[[subviewB]]
// CHECK: linalg.transpose

// CHECK: scf.if
// CHECK-NOT: linalg.fill
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0, stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS]], %[[allocTransA]], %[[allocTransB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()
// CHECK: else
// CHECK: %[[DIS2:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS2]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm(%[[DIS2]], %[[allocTransA]], %[[allocTransB]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS2]]) : (i64) -> ()

// -----
