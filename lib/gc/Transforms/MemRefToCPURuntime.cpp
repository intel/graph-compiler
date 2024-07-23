//===- MemRefToCPURuntime.cpp -MemRef To CPURuntime Lowering ----*- C++ -*-===//
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

using namespace mlir::cpuruntime;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONVERTMEMREFTOCPURUNTIME
#include "gc/Transforms/Passes.h.inc"

namespace {
struct AlignedAllocLowering : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    MemRefType type = op.getMemref().getType();
    ValueRange symbolOperands = op.getSymbolOperands();
    ValueRange dynamicSizes = op.getDynamicSizes();
    // Check if the parent contains a forall / parallel loop
    bool hasParallelParent = false;
    for (Operation *parentOp = op->getParentOp(); parentOp != nullptr;
         parentOp = parentOp->getParentOp()) {
      if (isa<scf::ForallOp, scf::ParallelOp>(parentOp)) {
        hasParallelParent = true;
        break;
      }
    }
    Operation *newAllocOp;
    if (hasParallelParent) {
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
    // Check if the parent contains a forall / parallel loop
    bool hasParallelParent = false;
    for (Operation *parentOp = op->getParentOp(); parentOp != nullptr;
         parentOp = parentOp->getParentOp()) {
      if (isa<scf::ForallOp, scf::ParallelOp>(parentOp)) {
        hasParallelParent = true;
        break;
      }
    }
    Operation *newDeallocOp;
    if (hasParallelParent) {
      newDeallocOp = rewriter.create<cpuruntime::ThreadDeallocOp>(loc, memref);
    } else {
      newDeallocOp = rewriter.create<cpuruntime::DeallocOp>(loc, memref);
    }
    rewriter.replaceOp(op, newDeallocOp->getResults());
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
              SmallPtrSet<Value, 16> aliases = analysis.resolve(operand);
              // Check if any of the returned memref is allocated within scope.
              for (Value alias : aliases) {
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
