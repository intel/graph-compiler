//===-- Utils.h - Utils for CPURuntime --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_CPURUNTIME_UTILS_H
#define GC_EXECUTIONENGINE_CPURUNTIME_UTILS_H

#if defined _WIN32 || defined __CYGWIN__
#define GC_DLL_EXPORT __declspec(dllexport)
#else
#define GC_DLL_EXPORT __attribute__((visibility("default")))
#endif

#endif
