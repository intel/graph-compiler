//===-- GpuOclRuntime.h - GPU OpenCL runtime --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_GPUOCLRUNTIME_H
#define GC_GPUOCLRUNTIME_H

namespace mlir::gc::gpu {
constexpr char GPU_OCL_MALLOC[] = "gcGpuOclMalloc";
constexpr char GPU_OCL_DEALLOC[] = "gcGpuOclDealloc";
constexpr char GPU_OCL_MEMCPY[] = "gcGpuOclMemcpy";
constexpr char GPU_OCL_KERNEL_CREATE[] = "gcGpuOclKernelCreate";
constexpr char GPU_OCL_KERNEL_DESTROY[] = "gcGpuOclKernelDestroy";
constexpr char GPU_OCL_KERNEL_LAUNCH[] = "gcGpuOclKernelLaunch";
constexpr char GPU_OCL_MOD_DESTRUCTOR[] = "gcGpuOclModuleDestructor";
} // namespace mlir::gc::gpu

#ifndef GC_GPU_OCL_CONST_ONLY

// TBD

#else
#undef GC_GPU_OCL_CONST_ONLY
#endif
#endif
