//===- CPURuntimeOps.cpp - CPU Runtime Ops ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"

#define GET_OP_CLASSES
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.cpp.inc"

#include <llvm/Support/Debug.h>

namespace mlir {
namespace cpuruntime {} // namespace cpuruntime
} // namespace mlir