// RUN: gc-opt %s -convert-linalg-to-microkernel -split-input-file | FileCheck %s

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @basic_linalg_to_microkernel(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %alloc_1 = tensor.empty() : tensor<4x16x32x32xf32>
  %alloc_4 = tensor.empty() : tensor<8x16x32x32xf32>
  %ret = scf.forall (%arg7, %arg8) in (4, 8) shared_outs(%argp = %arg0) -> (tensor<4x8x32x32xf32>) {
    %alloc_10 = tensor.extract_slice %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %subview = tensor.extract_slice %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
    %subview_11 = tensor.extract_slice %alloc_4[%arg8, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
    %res = linalg.batch_reduce_matmul ins(%subview, %subview_11 : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%alloc_10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    } 
  }
  return %ret : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: basic_linalg_to_microkernel
// CHECK: scf.forall
// CHECK: %[[C:.+]] = tensor.extract_slice %[[Csrc:.+]][%[[arg1:.+]], %[[arg2:.+]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[A:.+]] = tensor.extract_slice %[[Asrc:.+]][%[[arg1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[B:.+]] = tensor.extract_slice %[[Bsrc:.+]][%[[arg2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[RES:.+]] = microkernel.brgemm ins(%[[A]], %[[B]]) outs(%[[C]]) batch_dims(0, 0) leading_dims(1, 1) flags()  -> tensor<32x32xf32>
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @vnni_linalg_to_microkernel(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %alloc_1 = tensor.empty() : tensor<4x16x32x32xbf16>
  %alloc_4 = tensor.empty() : tensor<8x16x16x32x2xbf16>
  %ret = scf.forall (%arg7, %arg8) in (4, 8) shared_outs(%argp = %arg0) -> (tensor<4x8x32x32xf32>) {
    %alloc_10 = tensor.extract_slice %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %subview = tensor.extract_slice %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
    %subview_11 = tensor.extract_slice %alloc_4[%arg8, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
    %res = linalgx.batch_reduce_matmul_vnni ins(%subview, %subview_11 : tensor<16x32x32xbf16>, tensor<16x16x32x2xbf16>) outs(%alloc_10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    } 
  }
  return %ret : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: vnni_linalg_to_microkernel
// CHECK: scf.forall
// CHECK: %[[C:.+]] = tensor.extract_slice %[[Csrc:.+]][%[[arg1:.+]], %[[arg2:.+]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[A:.+]] = tensor.extract_slice %[[Asrc:.+]][%[[arg1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[B:.+]] = tensor.extract_slice %[[Bsrc:.+]][%[[arg2]], 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK: %[[RES:.+]] = microkernel.brgemm ins(%[[A]], %[[B]]) outs(%[[C]]) batch_dims(0, 0) leading_dims(1, 1) flags()  -> tensor<32x32xf32>
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @basic_linalg_to_microkernel_fusing_fill(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = tensor.empty() : tensor<4x16x32x32xf32>
  %alloc_4 = tensor.empty() : tensor<8x16x32x32xf32>
  %ret = scf.forall (%arg7, %arg8) in (4, 8) shared_outs(%argp = %arg0) -> (tensor<4x8x32x32xf32>) {
    %alloc_10 = tensor.extract_slice %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %subview = tensor.extract_slice %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
    %subview_11 = tensor.extract_slice %alloc_4[%arg8, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%alloc_10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %res = linalg.batch_reduce_matmul ins(%subview, %subview_11 : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    } 
  }
  return %ret : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: basic_linalg_to_microkernel_fusing_fill
// CHECK: scf.forall
// CHECK: %[[C:.+]] = tensor.extract_slice %[[Csrc:.+]][%[[arg1:.+]], %[[arg2:.+]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[A:.+]] = tensor.extract_slice %[[Asrc:.+]][%[[arg1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[B:.+]] = tensor.extract_slice %[[Bsrc:.+]][%[[arg2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK-NOT: linalg.fill
// CHECK: %[[RES:.+]] = microkernel.brgemm ins(%[[A]], %[[B]]) outs(%[[C]]) batch_dims(0, 0) leading_dims(1, 1) flags(beta_0)  -> tensor<32x32xf32>
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @vnni_linalg_to_microkernel_fusing_fill(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = tensor.empty() : tensor<4x16x32x32xbf16>
  %alloc_4 = tensor.empty() : tensor<8x16x16x32x2xbf16>
  %ret = scf.forall (%arg7, %arg8) in (4, 8) shared_outs(%argp = %arg0) -> (tensor<4x8x32x32xf32>) {
    %alloc_10 = tensor.extract_slice %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %subview = tensor.extract_slice %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
    %subview_11 = tensor.extract_slice %alloc_4[%arg8, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
    %11 = linalg.fill ins(%cst : f32) outs(%alloc_10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %res = linalgx.batch_reduce_matmul_vnni ins(%subview, %subview_11 : tensor<16x32x32xbf16>, tensor<16x16x32x2xbf16>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    } 
  }
  return %ret : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: vnni_linalg_to_microkernel_fusing_fill
// CHECK: scf.forall
// CHECK: %[[C:.+]] = tensor.extract_slice %[[Csrc:.+]][%[[arg1:.+]], %[[arg2:.+]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[A:.+]] = tensor.extract_slice %[[Asrc:.+]][%[[arg1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[B:.+]] = tensor.extract_slice %[[Bsrc:.+]][%[[arg2]], 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK-NOT: linalg.fill
// CHECK: %[[RES:.+]] = microkernel.brgemm ins(%[[A]], %[[B]]) outs(%[[C]]) batch_dims(0, 0) leading_dims(1, 1) flags(beta_0)  -> tensor<32x32xf32>
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @basic_linalg_to_microkernel_fusing_transpose(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = tensor.empty() : tensor<4x16x32x32xf32>
  %alloc_4 = tensor.empty() : tensor<8x32x16x32xf32>
  %trans_base = tensor.empty() : tensor<16x32x32xf32>
  %ret = scf.forall (%arg7, %arg8) in (4, 8) shared_outs(%argp = %arg0) -> (tensor<4x8x32x32xf32>) {
    %alloc_10 = tensor.extract_slice %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %subview = tensor.extract_slice %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
    %subview_11 = tensor.extract_slice %alloc_4[%arg8, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : tensor<8x32x16x32xf32> to tensor<32x16x32xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%alloc_10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %transposed = linalg.transpose ins(%subview_11 : tensor<32x16x32xf32>) outs(%trans_base : tensor<16x32x32xf32>) permutation = [1, 0, 2]
    %res = linalg.batch_reduce_matmul ins(%subview, %transposed : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    } 
  }
  return %ret : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: basic_linalg_to_microkernel_fusing_transpose
// CHECK: scf.forall
// CHECK: %[[C:.+]] = tensor.extract_slice %[[Csrc:.+]][%[[arg1:.+]], %[[arg2:.+]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[A:.+]] = tensor.extract_slice %[[Asrc:.+]][%[[arg1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[B:.+]] = tensor.extract_slice %[[Bsrc:.+]][%[[arg2]], 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : tensor<8x32x16x32xf32> to tensor<32x16x32xf32>
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.transpose
// CHECK: %[[RES:.+]] = microkernel.brgemm ins(%[[A]], %[[B]]) outs(%[[C]]) batch_dims(0, 1) leading_dims(1, 0) flags(beta_0)  -> tensor<32x32xf32>
// CHECK-NEXT: scf.forall.in_parallel

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @vnni_linalg_to_microkernel_fusing_transpose(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_1 = tensor.empty() : tensor<4x16x32x32xbf16>
  %alloc_4 = tensor.empty() : tensor<8x16x16x32x2xbf16>
  %trans_base = tensor.empty() : tensor<16x16x32x2xbf16>
  %ret = scf.forall (%arg7, %arg8) in (4, 8) shared_outs(%argp = %arg0) -> (tensor<4x8x32x32xf32>) {
    %alloc_10 = tensor.extract_slice %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
    %subview = tensor.extract_slice %alloc_1[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
    %subview_11 = tensor.extract_slice %alloc_4[%arg8, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
    %11 = linalg.fill ins(%cst : f32) outs(%alloc_10 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %transposed = linalg.transpose ins(%subview_11 : tensor<16x16x32x2xbf16>) outs(%trans_base : tensor<16x16x32x2xbf16>) permutation = [1, 0, 2, 3]
    %res = linalgx.batch_reduce_matmul_vnni ins(%subview, %transposed : tensor<16x32x32xbf16>, tensor<16x16x32x2xbf16>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %argp[%arg7, %arg8, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
    } 
  }
  return %ret : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: vnni_linalg_to_microkernel_fusing_transpose
// CHECK: scf.forall
// CHECK: %[[C:.+]] = tensor.extract_slice %[[Csrc:.+]][%[[arg1:.+]], %[[arg2:.+]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[A:.+]] = tensor.extract_slice %[[Asrc:.+]][%[[arg1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[B:.+]] = tensor.extract_slice %[[Bsrc:.+]][%[[arg2]], 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<8x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.transpose
// CHECK: %[[RES:.+]] = microkernel.brgemm ins(%[[A]], %[[B]]) outs(%[[C]]) batch_dims(0, 1) leading_dims(1, 0) flags(beta_0)  -> tensor<32x32xf32>
// CHECK-NEXT: scf.forall.in_parallel

// -----
