// RUN: gc-opt %s --gc-gpu-pipeline -split-input-file | FileCheck %s


// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>>>} {
  func.func @relu_f16(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf16>
    %1 = tensor.empty() : tensor<1024x1024xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %3 = linalg.max ins(%0, %2 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    bufferization.materialize_in_destination %3 in restrict writable %arg1 : (tensor<1024x1024xf16>, memref<1024x1024xf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>>>} {
  func.func @dynamic_relu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<?x?xf16>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %0, %c0 : tensor<?x?xf16>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %0, %c1 : tensor<?x?xf16>
    %1 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %2 = linalg.fill ins(%cst : f16) outs(%1 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %3 = linalg.max ins(%0, %2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%1 : tensor<?x?xf16>) -> tensor<?x?xf16>
    bufferization.materialize_in_destination %3 in restrict writable %arg1 : (tensor<?x?xf16>, memref<?x?xf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>>>} {
  func.func @relu_bf16(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xbf16>
    %1 = tensor.empty() : tensor<1024x1024xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %2 = linalg.fill ins(%cst : bf16) outs(%1 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    %3 = linalg.max ins(%0, %2 : tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) outs(%1 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    bufferization.materialize_in_destination %3 in restrict writable %arg1 : (tensor<1024x1024xbf16>, memref<1024x1024xbf16>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>>>} {
  func.func @relu_f32(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1024x1024xf32>
    %1 = tensor.empty() : tensor<1024x1024xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %3 = linalg.max ins(%0, %2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg1 : (tensor<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: llvm
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"CPU" : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 32 : i32>>>} {
  func.func @relu_f32_corner_shape(%arg0: memref<1061x1061xf32>, %arg1: memref<1061x1061xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<1061x1061xf32>
    %1 = tensor.empty() : tensor<1061x1061xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1061x1061xf32>) -> tensor<1061x1061xf32>
    %3 = linalg.max ins(%0, %2 : tensor<1061x1061xf32>, tensor<1061x1061xf32>) outs(%1 : tensor<1061x1061xf32>) -> tensor<1061x1061xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg1 : (tensor<1061x1061xf32>, memref<1061x1061xf32>) -> ()
    return
  }
}

