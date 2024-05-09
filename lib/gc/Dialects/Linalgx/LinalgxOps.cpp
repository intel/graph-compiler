//===- LinalgxOps.h - linalgx dialect ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialects/Linalgx/LinalgxOps.h"
#include "gc/Dialects/Linalgx/LinalgxDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "gc/Dialects/Linalgx/LinalgxOps.cpp.inc"
