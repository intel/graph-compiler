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
        scf::SCFDialect
        // clang-format on
        >();
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
