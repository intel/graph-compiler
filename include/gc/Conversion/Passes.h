//===-- Passes.h - Conversion Passes  ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_CONVERSION_PASSES_H
#define GC_CONVERSION_PASSES_H

#include "gc/Conversion/XeVMToLLVM/XeVMToLLVM.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "gc/Conversion/Passes.h.inc"

} // namespace mlir

#endif // GC_CONVERSION_PASSES_H
