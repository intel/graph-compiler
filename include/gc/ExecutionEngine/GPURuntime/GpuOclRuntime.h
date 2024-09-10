//===-- GpuOclRuntime.h - GPU OpenCL runtime --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_GPUOCLMODULE_H
#define GC_GPUOCLMODULE_H

#define GC_GPU_OCL_MALLOC "gcGpuOclMaloc"
#define GC_GPU_OCL_DEALLOC "gcGpuOclDealloc"
#define GC_GPU_OCL_MEMCPY "gcGpuOclMemcpy"
#define GC_GPU_OCL_KERNEL_CREATE "gcGpuOclKernelCreate"
#define GC_GPU_OCL_KERNEL_DESTROY "gcGpuOclKernelDestroy"
#define GC_GPU_OCL_KERNEL_LAUNCH "gcGpuOclKernelLaunch"
#define GC_GPU_OCL_MOD_DESTRUCTOR "gcGpuOclModuleDestructor"

#ifndef GC_GPU_OCL_DEF_ONLY

// TBD

#else
#undef GC_GPU_OCL_DEF_ONLY
#endif
#endif
