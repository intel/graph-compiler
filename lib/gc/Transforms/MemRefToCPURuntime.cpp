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
#include "llvm/ADT/SmallSet.h"

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

template <typename OpTy>
struct AlignedAllocLowering : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
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

struct AlignedDeallocLowering : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    Value memref = op.getMemref();
    cpuruntime::DeallocOp newDeallocOp =
        rewriter.create<cpuruntime::DeallocOp>(loc, memref);
    if (hasParallelParent(op))
      newDeallocOp.setThreadLocal(true);
    rewriter.eraseOp(op);
    return success();
  }
};

using LowerAlloca = AlignedAllocLowering<memref::AllocaOp>;
using LowerAlloc = AlignedAllocLowering<memref::AllocOp>;

struct ConvertMemRefToCPURuntime
    : public impl::ConvertMemRefToCPURuntimeBase<ConvertMemRefToCPURuntime> {

  void runOnOperation() final {
    auto *ctx = &getContext();
    // Create a local set to store operations that should not be transformed.
    llvm::SmallSet<Operation *, 16> noTransformOps;

    // Walk through the module to find func::FuncOp instances.
    getOperation()->walk([&](func::FuncOp funcOp) {
      BufferViewFlowAnalysis analysis(funcOp);
      // Now walk through the operations within the func::FuncOp.
      funcOp.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::ReturnLike>()) {
          for (Value operand : op->getOperands()) {
            if (isa<MemRefType>(operand.getType())) {
              auto aliases = analysis.resolveReverse(operand);
              // Check if any of the returned memref is allocated within scope.
              for (auto &&alias : aliases) {
                if (Operation *allocOp =
                        alias.getDefiningOp<memref::AllocOp>()) {
                  noTransformOps.insert(allocOp);
                  UnitAttr unitAttr = UnitAttr::get(ctx);
                  allocOp->setAttr("leak", unitAttr);
                }
              }
            }
          }
        }
      });
    });
    getOperation()->walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        SmallVector<Operation *, 4> allocStack;
        region.walk([&](Operation *nestedOp) {
          Region *parentRegion = nestedOp->getParentRegion();
          if (parentRegion == &region) {
            if (isa<memref::AllocOp>(nestedOp)) {
              if (nestedOp->getAttr("leak") == nullptr) {
                allocStack.push_back(nestedOp);
              }
            } else if (isa<memref::DeallocOp>(nestedOp)) {
              Value deallocMemref = nestedOp->getOperands().front();
              if (!allocStack.empty()) {
                Value topAllocMemref = allocStack.back()->getResults().front();
                if (deallocMemref == topAllocMemref) {
                  allocStack.pop_back();
                } else {
                  noTransformOps.insert(nestedOp);
                  for (int i = allocStack.size() - 1; i >= 0; --i) {
                    Operation *curAlloc = allocStack[i];
                    if (deallocMemref == curAlloc->getResults().front()) {
                      noTransformOps.insert(curAlloc);
                      allocStack.erase(allocStack.begin() + i);
                      break;
                    }
                  }
                }
              } else {
                noTransformOps.insert(nestedOp);
              }
            }
          }
        });
        while (!allocStack.empty()) {
          noTransformOps.insert(allocStack.back());
          allocStack.pop_back();
        }
      }
    });

    // add lowering target
    ConversionTarget target(getContext());
    // Make all operations legal by default.
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addDynamicallyLegalOp<memref::AllocOp, memref::DeallocOp>(
        [&](Operation *op) {
          // Return true if the operation is in the noTransformOps set, making
          // it dynamically legal.
          return noTransformOps.find(op) != noTransformOps.end();
        });
    target.addIllegalOp<memref::AllocaOp>();
    // set pattern
    RewritePatternSet patterns(ctx);
    patterns.add<LowerAlloca>(ctx);
    patterns.add<LowerAlloc>(ctx);
    patterns.add<AlignedDeallocLowering>(ctx);
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
