//===- Passes.h - Graph Compiler passes -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_PASSES_H
#define GC_PASSES_H

#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace vector {
#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"
/// Creates an instance of the `vector.multi_reduction` lowering pass.
std::unique_ptr<Pass> createLowerVectorMultiReductionPass(
    VectorMultiReductionLowering option =
        VectorMultiReductionLowering::InnerParallel);
} // namespace vector
namespace gc {

#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
