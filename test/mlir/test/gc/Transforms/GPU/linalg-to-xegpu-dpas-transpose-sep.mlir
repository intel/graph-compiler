// RUN: gc-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 k-tile=16" -canonicalize -split-input-file | FileCheck %s

module {
  func.func @matmul_transpose_b_sep(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c1024, %c1024) step (%c32, %c32) {
      %subview_0 = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] : memref<1024x1024xf16> to memref<32x32xf16, strided<[1024, 1], offset: ?>>
      %subview_1 = memref.subview %arg0[%arg3, 0] [32, 1024] [1, 1] : memref<1024x1024xf16> to memref<32x1024xf16, strided<[1024, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0] [32, 1024] [1, 1] : memref<1024x1024xf16> to memref<32x1024xf16, strided<[1024, 1], offset: ?>>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x32xf16>
      linalg.transpose ins(%subview_2 : memref<32x1024xf16, strided<[1024, 1], offset: ?>>) outs(%alloc : memref<1024x32xf16>) permutation = [1, 0]
      linalg.matmul ins(%subview_1, %alloc : memref<32x1024xf16, strided<[1024, 1], offset: ?>>, memref<1024x32xf16>) outs(%subview_0 : memref<32x32xf16, strided<[1024, 1], offset: ?>>)
      memref.dealloc %alloc : memref<1024x32xf16>
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @matmul_transpose_b_sep
// CHECK-SAME:  %[[Ap:.+]]: memref<1024x1024xf16>, %[[Bp:.+]]: memref<1024x1024xf16>, %[[Cp:.+]]: memref<1024x1024xf16>

// CHECK: scf.parallel (%[[iter1:.+]], %[[iter2:.+]]) = (%c0, %c0) to (%c1024, %c1024) step (%c32, %c32) {
// CHECK: %[[C:.+]] = memref.subview %[[Cp]][%[[iter1]], %[[iter2]]] {{.*}}
// CHECK: %[[A:.+]] = memref.subview %[[Ap]][%[[iter1]], 0] {{.*}}
// CHECK: %[[B:.+]] = memref.subview %[[Bp]][%[[iter2]], 0] {{.*}}

// CHECK-NOT: memref.alloc()
// CHECK-NOT: linalg.transpose
// CHECK-NOT: memref.dealloc()

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

// Create DPAS computation loop over tiled reduction dimension.
// CHECK: %[[res:.+]]:11 = scf.for{{.*}}%c0 to %c1024 step %c16
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
// CHECK:   %[[vA_flat:.+]] = vector.shape_cast %[[vA]] : vector<32x16xf16> to vector<512xf16>
// CHECK:   %[[vA_dpas_flat:.+]] = vector.extract_strided_slice{{.*}}: vector<512xf16> to vector<128xf16>
// CHECK:   %[[vA_dpas:.+]] = vector.shape_cast %[[vA_dpas_flat]] : vector<128xf16> to vector<8x8x2xf16>
// CHECK-COUNT-3: vector.extract_strided_slice

// Perform DPAS computation.
// CHECK:   %[[dpas:.+]] = xegpu.dpas %[[vA_dpas]], %[[vB]], %[[acc]]
// CHECK-COUNT-7: xegpu.dpas
