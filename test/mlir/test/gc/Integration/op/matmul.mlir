// RUN: gc-opt %s --gc-gpu-pipeline -split-input-file | FileCheck %s

// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @matmul_f16(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16>
    %2 = tensor.empty() : tensor<4096x4096xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
    %4 = linalg.matmul_transpose_b ins(%0, %1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%3 : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4096x4096xf16>, memref<4096x4096xf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
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

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>}{
  func.func @dynamic_matmul_f16(%arg0: memref<?x?xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<?x1024xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<?x?xf16>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %0, %c0 : tensor<?x?xf16>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %0, %c1 : tensor<?x?xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf16>
    %2 = tensor.empty(%dim) : tensor<?x1024xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x1024xf16>) -> tensor<?x1024xf16>
    %4 = linalg.matmul_transpose_b ins(%0, %1 : tensor<?x?xf16>, tensor<1024x1024xf16>) outs(%3 : tensor<?x1024xf16>) -> tensor<?x1024xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<?x1024xf16>, memref<?x1024xf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @matmul_bf16(%arg0: memref<4096x4096xbf16>, %arg1: memref<4096x4096xbf16>, %arg2: memref<4096x4096xbf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xbf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xbf16>
    %2 = tensor.empty() : tensor<4096x4096xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %3 = linalg.fill ins(%cst : bf16) outs(%2 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %4 = linalg.matmul_transpose_b ins(%0, %1 : tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) outs(%3 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4096x4096xbf16>, memref<4096x4096xbf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @matmul_f32(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf32>
    %2 = tensor.empty() : tensor<4096x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %4 = linalg.matmul_transpose_b ins(%0, %1 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%3 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
}
