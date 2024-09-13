// RUN: gc-opt %s -expand-microkernel -split-input-file | FileCheck %s

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @basic_expand_microkernel_non_init() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    microkernel.brgemm ins(%subview, %subview_11 : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>) batch_dims(0, 0) leading_dims(1, 1) flags()
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

// CHECK-LABEL: basic_expand_microkernel_non_init
// CHECK: %[[CST0:.+]] = arith.constant 0 : i64
// CHECK: %[[CST16:.+]] = arith.constant 16 : i64
// CHECK: %[[C:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32> 
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(f32, f32)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm.execute(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @basic_expand_microkernel_init() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    microkernel.brgemm ins(%subview, %subview_11 : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>) batch_dims(0, 0) leading_dims(1, 1) flags(beta_0)
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

// CHECK-LABEL: basic_expand_microkernel_init
// CHECK: %[[CST0:.+]] = arith.constant 0 : i64
// CHECK: %[[CST16:.+]] = arith.constant 16 : i64
// CHECK: %[[C:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32> 
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(beta_0, stride) data_type(f32, f32)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm.execute(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_expand_microkernel_init() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x32x16x32xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x32x16x32xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<4x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<8x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
    microkernel.brgemm ins(%subview, %subview_11 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>) batch_dims(1, 1) leading_dims(0, 0) flags(beta_0)
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

// CHECK-LABEL: transpose_expand_microkernel_init
// CHECK: %[[CST0:.+]] = arith.constant 0 : i64
// CHECK: %[[CST16:.+]] = arith.constant 16 : i64
// CHECK: %[[C:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32> 
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<4x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<8x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 512, 512, 32, 32, 32] flags(beta_0, stride) data_type(f32, f32)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm.execute(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_expand_microkernel_init_vnni() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x32x16x32xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
  scf.forall (%arg7, %arg8) in (4, 8) {
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    %subview = memref.subview %alloc_1[%arg7, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<4x32x16x32xbf16> to memref<32x16x32xbf16, strided<[512, 32, 1], offset: ?>>
    %subview_11 = memref.subview %alloc_4[%arg8, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    microkernel.brgemm ins(%subview, %subview_11 : memref<32x16x32xbf16, strided<[512, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%alloc_10 : memref<32x32xf32>) batch_dims(1, 1) leading_dims(0, 0) flags(beta_0)
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

// CHECK-LABEL: transpose_expand_microkernel_init_vnni
// CHECK: %[[CST0:.+]] = arith.constant 0 : i64
// CHECK: %[[CST16:.+]] = arith.constant 16 : i64
// CHECK: %[[C:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32> 
// CHECK: %[[A:.+]] = memref.subview %[[TMP1:.+]][%arg0, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<4x32x16x32xbf16> to memref<32x16x32xbf16, strided<[512, 32, 1], offset: ?>>
// CHECK: %[[B:.+]] = memref.subview %[[TMP2:.+]][%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [32, 32, 32, 512, 512, 32, 32, 64] flags(beta_0, stride) data_type(bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm.execute(%[[DIS]], %[[A]], %[[B]], %[[C]], %[[CST16]], %[[CST0]]) : (i64, memref<32x16x32xbf16, strided<[512, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()

// -----

#map = affine_map<(d0) -> (-d0 + 344, 11)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @expand_microkernel_with_dynamic(%arg0: memref<1x128x1x32xbf16>, %arg1: memref<344x128x16x32x2xbf16>, %arg2: memref<1x344x1x32xbf16>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg3) = (0) to (344) step (11) {
      %0 = affine.min #map(%arg3)
      %subview = memref.subview %arg2[0, %arg3, 0, 0] [1, %0, 1, 32] [1, 1, 1, 1] : memref<1x344x1x32xbf16> to memref<1x?x1x32xbf16, strided<[11008, 32, 32, 1], offset: ?>>
      scf.for %arg4 = %c0 to %0 step %c8 {
        %1 = affine.min #map1(%arg4)[%0]
        %subview_0 = memref.subview %subview[0, %arg4, 0, 0] [1, %1, 1, 32] [1, 1, 1, 1] : memref<1x?x1x32xbf16, strided<[11008, 32, 32, 1], offset: ?>> to memref<1x?x1x32xbf16, strided<[11008, 32, 32, 1], offset: ?>>
        %alloc = memref.alloc(%1) {alignment = 64 : i64} : memref<1x?x1x32xf32>
        scf.for %arg5 = %c0 to %c128 step %c64 {
          %subview_1 = memref.subview %alloc[0, 0, 0, 0] [1, %1, 1, 32] [1, 1, 1, 1] : memref<1x?x1x32xf32> to memref<1x?x1x32xf32, strided<[?, 32, 32, 1]>>
          %subview_2 = memref.subview %arg0[0, %arg5, 0, 0] [1, 64, 1, 32] [1, 1, 1, 1] : memref<1x128x1x32xbf16> to memref<64x1x32xbf16, strided<[32, 32, 1], offset: ?>>
          %2 = arith.cmpi eq, %arg5, %c0 : index
          %3 = arith.addi %arg5, %c64 : index
          %4 = arith.cmpi sge, %3, %c128 : index
          scf.for %arg6 = %c0 to %1 step %c1 {
            %5 = affine.apply #map2()[%arg3, %arg6, %arg4]
            %subview_3 = memref.subview %arg1[%5, %arg5, 0, 0, 0] [1, 64, 16, 32, 2] [1, 1, 1, 1, 1] : memref<344x128x16x32x2xbf16> to memref<64x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
            %subview_4 = memref.subview %subview_1[0, %arg6, 0, 0] [1, 1, 1, 32] [1, 1, 1, 1] : memref<1x?x1x32xf32, strided<[?, 32, 32, 1]>> to memref<1x32xf32, strided<[?, 1], offset: ?>>
            %subview_5 = memref.subview %subview_0[0, %arg6, 0, 0] [1, 1, 1, 32] [1, 1, 1, 1] : memref<1x?x1x32xbf16, strided<[11008, 32, 32, 1], offset: ?>> to memref<1x32xbf16, strided<[11008, 1], offset: ?>>
            scf.if %2 {
              microkernel.brgemm ins(%subview_2, %subview_3 : memref<64x1x32xbf16, strided<[32, 32, 1], offset: ?>>, memref<64x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%subview_4 : memref<1x32xf32, strided<[?, 1], offset: ?>>) batch_dims(0, 0) leading_dims(1, 1) flags(beta_0) 
            } else {
              microkernel.brgemm ins(%subview_2, %subview_3 : memref<64x1x32xbf16, strided<[32, 32, 1], offset: ?>>, memref<64x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%subview_4 : memref<1x32xf32, strided<[?, 1], offset: ?>>) batch_dims(0, 0) leading_dims(1, 1) flags() 
            }
            scf.if %4 {
              linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%subview_4 : memref<1x32xf32, strided<[?, 1], offset: ?>>) outs(%subview_5 : memref<1x32xbf16, strided<[11008, 1], offset: ?>>) {
              ^bb0(%in: f32, %out: bf16):
                %6 = arith.truncf %in : f32 to bf16
                linalg.yield %6 : bf16
              }
            }
          }
        }
        memref.dealloc %alloc : memref<1x?x1x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: expand_microkernel_with_dynamic
// CHECK: scf.forall (%[[ARG:.+]]) = (0) to (344) step (11)
// CHECK: scf.for %[[ARG2:.+]] = %[[CST0:.+]] to %[[AFF:.+]] step %[[CST8:.+]]
// CHECK: scf.for %[[ARG3:.+]] = %[[CST0]] to %[[CST128:.+]] step %[[CST64:.+]]
// CHECK: scf.for %[[ARG4:.+]] = %[[CST0]] to %[[AFF1:.+]] step %[[CST1:.+]]
// CHECK: scf.if
// CHECK: %[[DIS:.+]] = microkernel.brgemm.dispatch [1, 32, 32, 32, 32, 9223372036854775807, 32, 1024] flags(beta_0, stride) data_type(bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm.execute(%[[DIS]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS]]) : (i64) -> ()
// CHECK: else
// CHECK: %[[DIS2:.+]] = microkernel.brgemm.dispatch [1, 32, 32, 32, 32, 9223372036854775807, 32, 1024] flags(stride) data_type(bf16, bf16)
// CHECK-NEXT: microkernel.brgemm.prologue(%[[DIS2]]) : (i64) -> ()
// CHECK-NEXT: microkernel.brgemm.execute(%[[DIS]]
// CHECK-NEXT: microkernel.brgemm.epilogue(%[[DIS2]]) : (i64) -> ()

// -----
