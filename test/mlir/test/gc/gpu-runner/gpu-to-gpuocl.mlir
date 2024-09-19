// RUN: gc-opt %s --gc-gpu-pipeline | FileCheck %s

module @test {
  func.func @entry(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<32x32xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<32x32xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.add ins(%1, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
    return
  }
}

// CHECK: llvm.mlir.global internal constant @gcGpuOclKernel_entry_kernel_SPIRV
// CHECK: llvm.mlir.global internal constant @gcGpuOclKernel_entry_kernel_Name
// CHECK: llvm.mlir.global internal @gcGpuOclKernel_entry_kernel_Ptr

// CHECK: llvm.func internal @createGcGpuOclKernel_entry_kernel([[CTX:%.+]]: !llvm.ptr) -> !llvm.ptr
// CHECK: [[PTR_ADDR:%.+]] = llvm.mlir.addressof @gcGpuOclKernel_entry_kernel_Ptr
// CHECK: [[ZERO:%.+]] = llvm.mlir.zero
// CHECK: [[ONE:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[NEW_PTR:%.+]] = llvm.call @gcGpuOclKernelCreate([[CTX]]
// CHECK: [[CMPXCHG:%.+]] = llvm.cmpxchg [[PTR_ADDR]], [[ZERO]], [[NEW_PTR]]
// CHECK: [[FLAG:%.+]] = llvm.extractvalue [[CMPXCHG]][1]
// CHECK: llvm.cond_br [[FLAG]], [[BB1:\^.+]], [[BB2:\^.+]]
// CHECK: [[BB1]]:
// CHECK: llvm.return [[NEW_PTR]]
// CHECK: [[BB2]]:
// CHECK: [[ARRAY:%.+]] = llvm.alloca [[ONE]]
// CHECK: llvm.store [[NEW_PTR]], [[ARRAY]]
// CHECK: llvm.call @gcGpuOclKernelDestroy([[ONE]], [[ARRAY]])
// CHECK: [[OLD_PTR:%.+]] = llvm.extractvalue [[CMPXCHG]][0]
// CHECK: llvm.return [[OLD_PTR]]

// CHECK: llvm.func internal @getGcGpuOclKernel_entry_kernel([[CTX:%.+]]: !llvm.ptr) -> !llvm.ptr attributes {always_inline}
// CHECK: [[ZERO:%.+]] = llvm.mlir.zero
// CHECK: [[PTR_ADDR:%.+]] = llvm.mlir.addressof @gcGpuOclKernel_entry_kernel_Ptr
// CHECK: [[PTR:%.+]] = llvm.load [[PTR_ADDR]]
// CHECK: [[ICMP:%.+]] = llvm.icmp "eq" [[PTR]], [[ZERO]]
// CHECK: llvm.cond_br [[ICMP]], [[BB1:\^.+]], [[BB2:\^.+]]
// CHECK: [[BB1]]:
// CHECK: [[NEW_PTR:%.+]] = llvm.call @createGcGpuOclKernel_entry_kernel([[CTX]])
// CHECK: llvm.return [[NEW_PTR]]
// CHECK: [[BB2]]:
// CHECK: llvm.return [[PTR]]

// CHECK: llvm.func @entry
// CHECK: [[KERNEL:%.+]] = llvm.call @getGcGpuOclKernel_entry_kernel([[CTX:%.+]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @gcGpuOclKernelLaunch([[CTX]], [[KERNEL]],

// CHECK: llvm.func @gcGpuOclKernelCreate
// CHECK: llvm.func @gcGpuOclKernelDestroy
// CHECK: llvm.func @gcGpuOclKernelLaunch


// CHECK: llvm.func @gcGpuOclModuleDestructor()
// CHECK: [[ONE:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[PTR_ADDR:%.+]] = llvm.mlir.addressof @gcGpuOclKernel_entry_kernel_Ptr
// CHECK: llvm.fence acquire
// CHECK: [[PTR:%.+]] = llvm.load [[PTR_ADDR]]
// CHECK: [[ARRAY:%.+]] = llvm.alloca [[ONE]]
// CHECK: llvm.store [[PTR]], [[ARRAY]]
// CHECK: llvm.call @gcGpuOclKernelDestroy([[ONE]], [[ARRAY]])
