// RUN: gc-opt %s --gpu-to-gpuocl | FileCheck %s

module @test attributes {gpu.container_module} {
  llvm.func @entry(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<64x64xf32>
    %gpu_mem = gpu.alloc  host_shared () : memref<64x64xf32>
    gpu.memcpy  %gpu_mem, %8 : memref<64x64xf32>, memref<64x64xf32>
    %9 = llvm.mlir.constant(32 : index) : i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%12, %12, %14) threads in (%14, %14, %14)  args(%10 : index, %gpu_mem : memref<64x64xf32>)
    gpu.memcpy  %8, %gpu_mem : memref<64x64xf32>, memref<64x64xf32>
    gpu.dealloc  %gpu_mem : memref<64x64xf32>
    llvm.return
  }

  gpu.module @entry_kernel attributes {gpu.binary = "Some SPIRV here \00"} {
    gpu.func @entry_kernel(%arg0: index, %arg1: memref<64x64xf32>) kernel attributes {} {
      gpu.return
    }
  }
}

// CHECK: llvm.mlir.global internal constant @gcGpuOclKernel_entry_kernel_SPIRV
// CHECK: llvm.mlir.global internal constant @gcGpuOclKernel_entry_kernel_Name
// CHECK: llvm.mlir.global internal @gcGpuOclKernel_entry_kernel_Ptr

// CHECK: llvm.func @createGcGpuOclKernel_entry_kernel([[CTX:%.+]]: !llvm.ptr) -> !llvm.ptr
// CHECK: [[NEW_PTR:%.+]] = llvm.call @gcGpuOclKernelCreate([[CTX]]
// CHECK: [[ZERO:%.+]] = llvm.mlir.zero
// CHECK: [[PTR_ADDR:%.+]] = llvm.mlir.addressof @gcGpuOclKernel_entry_kernel_Ptr
// CHECK: [[CMPXCHG:%.+]] = llvm.cmpxchg [[PTR_ADDR]], [[ZERO]], [[NEW_PTR]]
// CHECK: [[FLAG:%.+]] = llvm.extractvalue [[CMPXCHG]][1]
// CHECK: llvm.cond_br [[FLAG]], [[BB1:\^.+]], [[BB2:\^.+]]
// CHECK: [[BB1]]:
// CHECK: llvm.return [[NEW_PTR]]
// CHECK: [[BB2]]:
// CHECK: [[ONE:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[ARRAY:%.+]] = llvm.alloca [[ONE]]
// CHECK: [[ADDR:%.+]] = llvm.getelementptr [[ARRAY]]
// CHECK: llvm.store [[NEW_PTR]], [[ADDR]]
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

// CHECK: llvm.func @entry(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, [[CTX:%.+]]: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64)
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(16384 : i64) : i64
// CHECK: llvm.call @gcGpuOclMallocShared([[CTX]], [[SIZE]])
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(16384 : i64) : i64
// CHECK: [[SRC:%.+]] = llvm.extractvalue
// CHECK: [[DST:%.+]] = llvm.extractvalue [[GPU_MEMREF:%.+]][1]
// CHECK: llvm.call @gcGpuOclMemcpy([[CTX]], [[SRC]], [[DST]], [[SIZE]])
// CHECK: [[KERNEL:%.+]] = llvm.call @getGcGpuOclKernel_entry_kernel([[CTX:%.+]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @gcGpuOclKernelLaunch([[CTX]], [[KERNEL]],
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(16384 : i64) : i64
// CHECK: [[SRC:%.+]] = llvm.extractvalue [[GPU_MEMREF:%.+]][1]
// CHECK: [[DST:%.+]] = llvm.extractvalue
// CHECK: llvm.call @gcGpuOclMemcpy([[CTX]], [[SRC]], [[DST]], [[SIZE]])
// CHECK: [[GPU_PTR:%.+]] = llvm.extractvalue [[GPU_MEMREF:%.+]][0]
// CHECK: llvm.call @gcGpuOclDealloc([[CTX]], [[GPU_PTR]])

// CHECK: llvm.func @gcGpuOclKernelCreate
// CHECK: llvm.func @gcGpuOclKernelDestroy
// CHECK: llvm.func @gcGpuOclKernelLaunch


// CHECK: llvm.func @gcGpuOclModuleDestructor()
// CHECK: llvm.fence acquire
// CHECK: [[PTR_ADDR:%.+]] = llvm.mlir.addressof @gcGpuOclKernel_entry_kernel_Ptr
// CHECK: [[PTR:%.+]] = llvm.load [[PTR_ADDR]]
// CHECK: [[ONE:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[ARRAY:%.+]] = llvm.alloca [[ONE]]
// CHECK: [[ADDR:%.+]] = llvm.getelementptr [[ARRAY]]
// CHECK: llvm.store [[PTR]], [[ADDR]]
// CHECK: llvm.call @gcGpuOclKernelDestroy([[ONE]], [[ARRAY]])
