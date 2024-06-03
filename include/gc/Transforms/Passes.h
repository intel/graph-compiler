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
#define GEN_PASS_DECL_CONSTANTSUBGRAPHANALYSIS
#define GEN_PASS_DECL_CONSTANTTENSORFOLDING
#include "gc/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createConstantSubgraphAnalysisPass();
std::unique_ptr<Pass> createConstantTensorFoldingPass();

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
