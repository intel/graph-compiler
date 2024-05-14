//===- CPURuntimePasses.cpp - CPU Runtime Passes ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"

namespace mlir::cpuruntime {
#define GEN_PASS_DEF_CPURUNTIMEATEXITTOOMP
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h.inc"

namespace {

class CPURuntimeAtExitToOmpRewriter
    : public OpRewritePattern<AtParallelExitOp> {
public:
  using OpRewritePattern<AtParallelExitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtParallelExitOp op,
                                PatternRewriter &rewriter) const final {
    auto parent = op->getParentOp();
    Operation *secondLast = nullptr;
    while (parent && (llvm::isa<memref::AllocaScopeOp>(parent) ||
                      llvm::isa<omp::WsloopOp>(parent))) {
      secondLast = parent;
      parent = parent->getParentOp();
    }
    auto parallel = llvm::dyn_cast<omp::ParallelOp>(parent);
    if (!parallel) {
      return failure();
    }
    assert(secondLast->getBlock());
    auto itr = secondLast->getBlock()->end();
    --itr;
    rewriter.inlineBlockBefore(&op->getRegion(0).getBlocks().front(),
                               secondLast->getBlock(), itr);
    rewriter.eraseOp(op);
    return success();
  }
};

class CPURuntimeExitReturnRewriter
    : public OpRewritePattern<ParallelExitReturnOp> {
public:
  using OpRewritePattern<ParallelExitReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ParallelExitReturnOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

class CPURuntimeAtExitToOmp
    : public impl::CPURuntimeAtExitToOmpBase<CPURuntimeAtExitToOmp> {
public:
  using impl::CPURuntimeAtExitToOmpBase<
      CPURuntimeAtExitToOmp>::CPURuntimeAtExitToOmpBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CPURuntimeExitReturnRewriter>(&getContext());
    patterns.add<CPURuntimeAtExitToOmpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cpuruntime
