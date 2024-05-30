//===-- MicrokernelDialect.cpp - microkernel dialect ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelEnum.h"
#include "gc/Dialect/Microkernel/MicrokernelOps.h"

using namespace mlir;
using namespace mlir::microkernel;

#include "gc/Dialect/Microkernel/MicrokernelOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Microkernel dialect.
//===----------------------------------------------------------------------===//

void MicrokernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/Microkernel/MicrokernelOps.cpp.inc"
      >();
}
