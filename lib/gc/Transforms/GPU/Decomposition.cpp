//===--------- Decomposition.cpp - Decompose aggregated ops ------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_DECOMPOSITION
#define GEN_PASS_DEF_DECOMPOSITION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Decomposes any operation that implements AggregatedOpInterface by calling
// its decomposeOperation method.
struct DecomposeAggregatedOp : public RewritePattern {
  explicit DecomposeAggregatedOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto decomposableOp = dyn_cast<linalg::AggregatedOpInterface>(op);
    if (!decomposableOp || !isa<linalgx::AttentionOp>(op))
      return failure();

    FailureOr<SmallVector<Value>> maybeNewResults =
        decomposableOp.decomposeOperation(rewriter);
    if (failed(maybeNewResults))
      return failure();

    rewriter.replaceOp(op, maybeNewResults.value()[0]);
    return success();
  }
};

struct Decomposition final : gc::impl::DecompositionBase<Decomposition> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<DecomposeAggregatedOp>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns),
                                     GreedyRewriteConfig()))) {
      signalPassFailure();
    }
  }
};

} // namespace
