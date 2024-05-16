//===- CPURuntimeDialect.cpp - CPU Runtime Dialect --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"

using namespace mlir;
using namespace mlir::cpuruntime;

#include "gc/Dialect/CPURuntime/IR/CPURuntimeOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CPURuntime dialect.
//===----------------------------------------------------------------------===//

void CPURuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.cpp.inc"
      >();
}
