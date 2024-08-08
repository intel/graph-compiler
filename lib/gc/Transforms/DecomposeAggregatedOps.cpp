//===-- DecomposeAggregatedOps.cpp - Decompose Aggregated Ops ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace mlir {
namespace gc {
#define GEN_PASS_DEF_DECOMPOSEAGGREGATEDOPS
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {

struct DecomposeAggregateOpsImpl : public OpRewritePattern<linalg::SoftmaxOp> {
  using OpRewritePattern<linalg::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::SoftmaxOp softmaxOp,
                                PatternRewriter &rewriter) const override {
    auto decomposableOp =
        cast<linalg::AggregatedOpInterface>(softmaxOp.getOperation());
    FailureOr<SmallVector<Value>> maybeNewResult =
        decomposableOp.decomposeOperation(rewriter);
    if (failed(maybeNewResult))
      return failure();
    rewriter.replaceOp(softmaxOp, *maybeNewResult);
    return success();
  }
};

struct DecomposeAggregatedOps
    : public gc::impl::DecomposeAggregatedOpsBase<DecomposeAggregatedOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DecomposeAggregateOpsImpl>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace