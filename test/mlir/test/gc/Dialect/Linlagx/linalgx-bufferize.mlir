// RUN: mlir-opt --one-shot-bufferize="dialect-filter=linalgx,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: @batch_reduce_matmul_vnni
func.func @batch_reduce_matmul_vnni(%arg0: tensor<512x32x64xbf16>, %arg1: tensor<512x32x128x2xbf16>, 
                      %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
  // CHECK: bufferization.to_memref
  // CHECK: bufferization.to_memref
  // CHECK: bufferization.to_memref
  // CHECK: memref.alloc()
  // CHECK: memref.copy
  // CHECK: linalgx.batch_reduce_matmul_vnni
  // CHECK: bufferization.to_tensor
  %0 = linalgx.batch_reduce_matmul_vnni ins(%arg0, %arg1 : tensor<512x32x64xbf16>, tensor<512x32x128x2xbf16>) 
                          outs(%arg2 : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}
