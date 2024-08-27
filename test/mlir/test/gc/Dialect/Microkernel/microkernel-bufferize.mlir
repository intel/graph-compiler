// RUN: gc-opt %s -one-shot-bufferize -split-input-file | FileCheck %s

func.func @basic_microkernel_bufferize(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %2 = scf.forall (%arg1, %arg2) in (4, 8) shared_outs(%arg3 = %arg0) -> (tensor<4x8x32x32xf32>) {
    %extracted_slice = tensor.extract_slice %arg3[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %extracted_slice_0 = tensor.extract_slice %0[%arg1, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
    %extracted_slice_1 = tensor.extract_slice %1[%arg2, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
    %3 = microkernel.brgemm ins(%extracted_slice_0, %extracted_slice_1 : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) batch_dims(0, 0) leading_dims(1, 1) flags() -> tensor<32x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg3[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    }
  }
  return %2 : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: basic_microkernel_bufferize
// CHECK: %[[Abuf:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xf32>
// CHECK: %[[Bbuf:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32>
// CHECK: %[[Cbuf:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
// CHECK: scf.forall
// CHECK: %[[C:.+]] = memref.subview %[[Cbuf]]
// CHECK: %[[A:.+]] = memref.subview %[[Abuf]]
// CHECK: %[[B:.+]] = memref.subview %[[Bbuf]]
// CHECK: microkernel.brgemm ins(%[[A]], %[[B]] : [[TYPE:.+]]) outs(%[[C]] : [[TYPE2:.+]]) batch_dims(0, 0) leading_dims(1, 1) flags()

// -----

func.func @vnni_microkernel_bufferize(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %0 = tensor.empty() : tensor<4x16x32x32xbf16>
  %1 = tensor.empty() : tensor<8x16x16x32x2xbf16>
  %2 = scf.forall (%arg1, %arg2) in (4, 8) shared_outs(%arg3 = %arg0) -> (tensor<4x8x32x32xf32>) {
    %extracted_slice = tensor.extract_slice %arg3[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %extracted_slice_0 = tensor.extract_slice %0[%arg1, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
    %extracted_slice_1 = tensor.extract_slice %1[%arg2, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
    %3 = microkernel.brgemm ins(%extracted_slice_0, %extracted_slice_1 : tensor<16x32x32xbf16>, tensor<16x16x32x2xbf16>) outs(%extracted_slice : tensor<32x32xf32>) batch_dims(0, 0) leading_dims(1, 1) flags() -> tensor<32x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg3[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    }
  }
  return %2 : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: vnni_microkernel_bufferize
// CHECK: %[[Abuf:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
// CHECK: %[[Bbuf:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
// CHECK: %[[Cbuf:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
// CHECK: scf.forall
// CHECK: %[[C:.+]] = memref.subview %[[Cbuf]]
// CHECK: %[[A:.+]] = memref.subview %[[Abuf]]
// CHECK: %[[B:.+]] = memref.subview %[[Bbuf]]
// CHECK: microkernel.brgemm ins(%[[A]], %[[B]] : [[TYPE:.+]]) outs(%[[C]] : [[TYPE2:.+]]) batch_dims(0, 0) leading_dims(1, 1) flags()
