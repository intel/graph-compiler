//===- Passes.h - Graph Compiler passes -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_PASSES_H
#define GC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gc {

#define GEN_PASS_DECL
#define GEN_PASS_DECL_CSA
#define GEN_PASS_DECL_CST
#include "gc/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createCSAPass();
std::unique_ptr<Pass> createCSTPass();

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
