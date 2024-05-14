//===- CPURuntimePasses.h - CPU Runtime Passes ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CPURUNTIME_CPURUNTIMEPASSES_H
#define CPURUNTIME_CPURUNTIMEPASSES_H

#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace cpuruntime {
void registerConvertCPURuntimeToLLVMInterface(DialectRegistry &registry);

#define GEN_PASS_DECL
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h.inc"
} // namespace cpuruntime
} // namespace mlir

#endif
