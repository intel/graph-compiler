//===- MicrokernelDialect.h - microkernel dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc-dialects/Microkernel/MicrokernelDialect.h"
#include "gc-dialects/Microkernel/MicrokernelOps.h"

using namespace mlir;
using namespace mlir::microkernel;

void MicrokernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc-dialects/Microkernel/MicrokernelOps.cpp.inc"
      >();
}
