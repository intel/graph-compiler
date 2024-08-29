//===-- MemRefToCPURuntime.cpp - MemRef To CPURuntime Lowering --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <vector>

#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONVERTMEMREFTOCPURUNTIME
#include "gc/Transforms/Passes.h.inc"

namespace {

bool hasParallelParent(Operation *op) {
  // Check if the parent contains a forall / parallel loop
  for (Operation *parentOp = op->getParentOp(); parentOp != nullptr;
       parentOp = parentOp->getParentOp()) {
    if (parentOp->hasTrait<OpTrait::HasParallelRegion>()) {
      return true;
    }
  }
  return false;
}

struct AlignedAllocLowering : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    MemRefType type = op.getMemref().getType();
    ValueRange symbolOperands = op.getSymbolOperands();
    ValueRange dynamicSizes = op.getDynamicSizes();
    cpuruntime::AllocOp newAllocOp = rewriter.create<cpuruntime::AllocOp>(
        loc, type, dynamicSizes, symbolOperands);
    if (hasParallelParent(op))
      newAllocOp.setThreadLocal(true);
    rewriter.replaceOp(op, newAllocOp.getResult());
    return success();
  }
};
struct ConvertMemRefToCPURuntime
    : public impl::ConvertMemRefToCPURuntimeBase<ConvertMemRefToCPURuntime> {

  void runOnOperation() final {
    auto *ctx = &getContext();

    // Create deallocOp accoresponding to the alloca's localtion
    getOperation()->walk([&](func::FuncOp funcOp) {
      OpBuilder builder(funcOp.getContext());
      funcOp.walk([&](memref::AllocaOp op) {
        Region *parentRegion = op->getParentRegion();
        Block &lastBlock = parentRegion->back();
        builder.setInsertionPointToEnd(&lastBlock);
        if (!lastBlock.empty() &&
            lastBlock.back().hasTrait<OpTrait::IsTerminator>()) {
          builder.setInsertionPoint(&lastBlock.back());
        }
        auto deallocOp =
            builder.create<cpuruntime::DeallocOp>(op.getLoc(), op.getResult());
        if (hasParallelParent(op))
          deallocOp.setThreadLocal(true);
      });
    });

    // add lowering target
    ConversionTarget target(getContext());
    // Make all operations legal by default.
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalOp<memref::AllocaOp>();
    // set pattern
    RewritePatternSet patterns(ctx);
    patterns.add<AlignedAllocLowering>(ctx);
    // perform conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir
