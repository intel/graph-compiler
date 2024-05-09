//===- LinalgxDialect.h - linalgx dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

using namespace mlir;
using namespace mlir::linalgx;

void LinalgxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"
      >();
}
