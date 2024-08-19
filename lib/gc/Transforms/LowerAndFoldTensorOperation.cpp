//===-- FoldTensorOperation.cpp - fold tensor op ----------------*-C++//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace gc {

#define GEN_PASS_DEF_FOLDTENSOROPERATION
#include "gc/Transforms/Passes.h.inc"
namespace {

/// LowerAndFoldTensorOperation is a pass that fold some useless tensor
/// operation.
struct FoldTensorOperationPass
    : public impl::FoldTensorOperationBase<FoldTensorOperationPass> {
  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet pattern(ctx);

    tensor::ControlFoldFn defaultControlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer && producer->hasOneUse();
    };
    // Some operation convert as constant, this pattern can help us to improve
    // the performance.
    tensor::populateRewriteAsConstantPatterns(pattern, defaultControlFn);
    // Remove unnessary operation like extract slice and insert slice
    tensor::populateReassociativeReshapeFoldingPatterns(pattern);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(pattern);
    tensor::populateFoldTensorSubsetOpPatterns(pattern);

    GreedyRewriteConfig configInit;
    // Use to remove useless tensor operation like extract or
    // insert slice.
    configInit.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(pattern),
                                       configInit);
  }
};
} // namespace
} // namespace gc
} // namespace mlir