// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file | FileCheck %s

module {
  func.func @matmul_transpose_b(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c1024, %c1024) step (%c16, %c64) {
      %subview_0 = memref.subview %arg2[%arg3, %arg4] [16, 64] [1, 1] : memref<1024x1024xf16> to memref<16x64xf16, strided<[1024, 1], offset: ?>>
      %subview_1 = memref.subview %arg0[%arg3, 0] [16, 1024] [1, 1] : memref<1024x1024xf16> to memref<16x1024xf16, strided<[1024, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0] [64, 1024] [1, 1] : memref<1024x1024xf16> to memref<64x1024xf16, strided<[1024, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%subview_1, %subview_2 : memref<16x1024xf16, strided<[1024, 1], offset: ?>>, memref<64x1024xf16, strided<[1024, 1], offset: ?>>) outs(%subview_0 : memref<16x64xf16, strided<[1024, 1], offset: ?>>)
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @matmul_transpose_b
// CHECK-SAME:  %[[Ap:.+]]: memref<1024x1024xf16>, %[[Bp:.+]]: memref<1024x1024xf16>, %[[Cp:.+]]: memref<1024x1024xf16>

// CHECK: scf.parallel (%[[iter1:.+]], %[[iter2:.+]]) = (%c0, %c0) to (%c1024, %c1024) step (%c16, %c64) {
// CHECK: %[[C:.+]] = memref.subview %[[Cp]][%[[iter1]], %[[iter2]]] {{.*}}
// CHECK: %[[A:.+]] = memref.subview %[[Ap]][%[[iter1]], 0] {{.*}}
// CHECK: %[[B:.+]] = memref.subview %[[Bp]][%[[iter2]], 0] {{.*}}

// Create output initial value load tiles.
// CHECK-DAG: %[[rootC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK: %[[tC:.+]] = xegpu.update_nd_offset %[[rootC]], [%c0, %c0]
// CHECK-COUNT-7: xegpu.update_nd_offset

// Load initial accumulator values.
// CHECK-DAG: %[[vC:.+]] = xegpu.load_nd %[[tC]]
// CHECK-COUNT-7: xegpu.load_nd

// Extend the type to match DPAS output precision.
// CHECK: %[[vC_f32:.+]] = arith.extf %[[vC]]
// CHECK-COUNT-7: arith.extf

// Create input load tiles.
// CHECK: %[[rootA:.+]] = xegpu.create_nd_tdesc %[[A]]
// CHECK: %[[tA:.+]] = xegpu.update_nd_offset %[[rootA]], [%c0, %c0]
// CHECK: %[[rootB:.+]] = xegpu.create_nd_tdesc %[[B]]
// CHECK: %[[tB:.+]] = xegpu.update_nd_offset %[[rootB]], [%c0, %c0]
// CHECK: %[[tB1:.+]] = xegpu.update_nd_offset %[[rootB]], [%c16, %c0]
// CHECK: %[[tB2:.+]] = xegpu.update_nd_offset %[[rootB]], [%c32, %c0]
// CHECK: %[[tB3:.+]] = xegpu.update_nd_offset %[[rootB]], [%c48, %c0]

// Create DPAS computation loop over tiled reduction dimension.
// CHECK: %[[res:.+]]:13 = scf.for{{.*}}%c0 to %c1024 step %c16
// CHECK-SAME: iter_args(%[[acc:.+]] = %[[vC_f32]],{{.*}}%[[iterA:.+]] = %[[tA]],{{.*}}%[[iterB:.+]] = %[[tB]],{{.*}}%[[iterB1:.+]] = %[[tB1]]
// CHECK-SAME: {

// Load input values and update the load tile position.
// CHECK:   %[[vA:.+]] = xegpu.load_nd %[[iterA]]
// CHECK:   %[[vB:.+]] = xegpu.load_nd %[[iterB]] {{.*}}transpose = array<i64: 1, 0>{{.*}}transpose_bit_width = 32 : i32{{.*}}
// CHECK:   %[[vB1:.+]] = xegpu.load_nd %[[iterB1]] {{.*}}transpose = array<i64: 1, 0>, transpose_bit_width = 32 : i32{{.*}}

// CHECK:   %[[new_tA:.+]] = xegpu.update_nd_offset %[[iterA]], [%c0, %c16]
// CHECK:   %[[new_tB:.+]] = xegpu.update_nd_offset %[[iterB]], [%c0, %c16]
// CHECK:   %[[new_tB1:.+]] = xegpu.update_nd_offset %[[iterB1]], [%c0, %c16]

// Apply simple prefetching scheme - start loading the next set of input
// tiles before computation is started.
// CHECK:   xegpu.prefetch_nd %[[new_tA]]
// CHECK:   xegpu.prefetch_nd %[[new_tB]]
// CHECK:   xegpu.prefetch_nd %[[new_tB1]]

// Extract DPAS-sized chunks from larger loaded tile A.
// Tile B is already in the correct shape.
// CHECK:   %[[vA_flat:.+]] = vector.shape_cast %[[vA]] : vector<16x16xf16> to vector<256xf16>
// CHECK:   %[[vA_dpas_flat:.+]] = vector.extract_strided_slice{{.*}}: vector<256xf16> to vector<128xf16>
// CHECK:   %[[vA_dpas:.+]] = vector.shape_cast %[[vA_dpas_flat]] : vector<128xf16> to vector<8x16xf16>
// CHECK-COUNT-1: vector.extract_strided_slice

// Perform DPAS computation.
// CHECK:   %[[dpas:.+]] = xegpu.dpas %[[vA_dpas]], %[[vB]], %[[acc]]
// CHECK-COUNT-7: xegpu.dpas
