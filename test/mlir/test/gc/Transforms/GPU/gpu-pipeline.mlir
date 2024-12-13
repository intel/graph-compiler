// RUN: gc-opt %s --gc-gpu-pipeline -split-input-file | FileCheck %s

// Regression check for infinite loop in the linalg-to-xegpu pass
// CHECK-DAG: @gcGpuOclKernel_corner_shape_matmul_f16_kernel_SPIRV
// CHECK-DAG: @createGcGpuOclKernel_corner_shape_matmul_f16_kernel
// CHECK-DAG: @corner_shape_matmul_f16
module @fragment_name attributes {} {
  func.func @corner_shape_matmul_f16(%arg0: memref<521x521xf16>, %arg1: memref<521x521xf16>, %arg2: memref<521x521xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<521x521xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<521x521xf16>
    %2 = tensor.empty() : tensor<521x521xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<521x521xf16>) -> tensor<521x521xf16>
    %4 = linalg.matmul_transpose_b ins(%0, %1 : tensor<521x521xf16>, tensor<521x521xf16>) outs(%3 : tensor<521x521xf16>) -> tensor<521x521xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<521x521xf16>, memref<521x521xf16>) -> ()
    return
  }
}
