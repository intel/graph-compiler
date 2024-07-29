//===-- MemRefToCPURuntime.cpp - MemRef To CPURuntime Lowering --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"
#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include <numeric>
#include <vector>

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
struct AlignedAllocLowering : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    MemRefType type = op.getMemref().getType();
    ValueRange symbolOperands = op.getSymbolOperands();
    ValueRange dynamicSizes = op.getDynamicSizes();
    Operation *newAllocOp;
    if (hasParallelParent(op)) {
      newAllocOp = rewriter.create<cpuruntime::ThreadAllocOp>(
          loc, type, dynamicSizes, symbolOperands);
    } else {
      newAllocOp = rewriter.create<cpuruntime::AllocOp>(loc, type, dynamicSizes,
                                                        symbolOperands);
    }
    rewriter.replaceOp(op, newAllocOp->getResults());
    return success();
  }
};

struct AlignedDeallocLowering : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    Value memref = op.getMemref();
    Operation *newDeallocOp;
    if (hasParallelParent(op)) {
      newDeallocOp = rewriter.create<cpuruntime::ThreadDeallocOp>(loc, memref);
    } else {
      newDeallocOp = rewriter.create<cpuruntime::DeallocOp>(loc, memref);
    }
    rewriter.replaceOp(op, newDeallocOp->getResults());
    return success();
  }
};

/// Given a memref value, return the "base" value by skipping over all
/// ViewLikeOpInterface ops (if any) in the reverse use-def chain.
static Value getViewBase(Value value) {
  while (auto viewLikeOp = value.getDefiningOp<ViewLikeOpInterface>())
    value = viewLikeOp.getViewSource();
  return value;
}

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
              Value v = getViewBase(operand);
              auto aliases = analysis.resolveReverse(v);
              // Check if any of the returned memref is allocated within scope.
              for (auto &&alias : aliases) {
                if (Operation *allocOp =
                        alias.getDefiningOp<memref::AllocOp>()) {
                  noTransformOps.insert(allocOp);
                }
              }
            }
          }
        }
      });
    });

    // add lowering target
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<memref::AllocOp, memref::DeallocOp>(
        [&](Operation *op) {
          // Return true if the operation is in the noTransformOps set, making
          // it dynamically legal.
          return noTransformOps.find(op) != noTransformOps.end();
        });
    target.addLegalDialect<
        // clang-format off
        BuiltinDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        cpuruntime::CPURuntimeDialect,
        arith::ArithDialect,
        affine::AffineDialect,
        microkernel::MicrokernelDialect,
        LLVM::LLVMDialect,
        scf::SCFDialect
        // clang-format on
        >();
    target.addLegalOp<LLVM::GlobalCtorsOp>();
    target.addLegalOp<LLVM::GlobalOp>();
    // set pattern
    RewritePatternSet patterns(ctx);
    patterns.add<AlignedAllocLowering>(ctx);
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