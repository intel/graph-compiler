// RUN: gc-opt %s --gc-gpu-pipeline -split-input-file | FileCheck %s

// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @multiply(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>, %arg3: memref<1024x1024xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1024x1024xf16>
    %3 = tensor.empty() : tensor<1024x1024xf16>
    %4 = linalg.mul ins(%0, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%3 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg3 : (tensor<1024x1024xf16>, memref<1024x1024xf16>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @add(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>, %arg3: memref<1024x1024xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1024x1024xf16>
    %3 = tensor.empty() : tensor<1024x1024xf16>
    %4 = linalg.add ins(%0, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%3 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg3 : (tensor<1024x1024xf16>, memref<1024x1024xf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @subtract(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>, %arg3: memref<1024x1024xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1024x1024xf16>
    %3 = tensor.empty() : tensor<1024x1024xf16>
    %4 = linalg.sub ins(%0, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%3 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg3 : (tensor<1024x1024xf16>, memref<1024x1024xf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"num_exec_units", 448 : i32>, #dlti.dl_entry<"num_exec_units_per_slice", 32 : i32>, #dlti.dl_entry<"num_threads_per_eu", 8 : i32>, #dlti.dl_entry<"L1_cache_size_in_bytes", 67108864 : i32>, #dlti.dl_entry<"max_vector_op_width", 256 : i32>, #dlti.dl_entry<"max_work_group_size", 1024 : i32>>>} {
  func.func @divide(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>, %arg3: memref<1024x1024xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<1024x1024xf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<1024x1024xf16>
    %3 = tensor.empty() : tensor<1024x1024xf16>
    %4 = linalg.div ins(%0, %1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%3 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    bufferization.materialize_in_destination %4 in restrict writable %arg3 : (tensor<1024x1024xf16>, memref<1024x1024xf16>) -> ()
    return
  }
}
