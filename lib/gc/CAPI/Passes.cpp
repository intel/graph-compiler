//===-- Passes.cpp - DESC ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"

// #include "gc/Dialect/Linalgx/Transforms/AllInterfaces.h"
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.capi.h.inc"
#include "gc/Transforms/Passes.capi.h.inc"
using namespace mlir::gc;
using namespace mlir::cpuruntime;

#ifdef __cplusplus
extern "C" {
#endif

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.capi.cpp.inc"
#include "gc/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif