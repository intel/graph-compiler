//===- MicrokernelOps.h - microkernel dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialects/Microkernel/MicrokernelOps.h"
#include "gc/Dialects/Microkernel/MicrokernelDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "gc/Dialects/Microkernel/MicrokernelOps.cpp.inc"
