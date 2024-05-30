//===- MicrokernelPasses.h - Graph Compiler microkerenl passes --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_MICROKERNELPASSES_H
#define GC_MICROKERNELPASSES_H

#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace microkernel {
#define GEN_PASS_DECL
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"
} // namespace microkernel
} // namespace mlir

#endif // GC_MICROKERNELPASSES_H
