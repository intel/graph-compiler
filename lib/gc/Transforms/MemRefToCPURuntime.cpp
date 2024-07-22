//===- MemRefToCPURuntime.cpp -MemRef To CPURuntime Lowering --*- C++ -*-=//
//-*-===//
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir::cpuruntime;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONVERTMEMREFTOCPURUNTIME
#include "gc/Transforms/Passes.h.inc"

static Value getViewBase(Value value) {
  while (auto viewLikeOp = value.getDefiningOp<ViewLikeOpInterface>())
    value = viewLikeOp.getViewSource();
  return value;
}
namespace {
struct AlignedAllocLowering : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {
    auto attr = op->getAttr("no_trans");
    if (attr && attr.isa<mlir::BoolAttr>() &&
        attr.cast<mlir::BoolAttr>().getValue()) {
      success();
    }
    // The operation has a "no_trans" attribute set to true
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
    llvm::outs() << "get into memreftocpuruntime conversoin\n";
    auto *ctx = &getContext();
    // add lowering target
    ConversionTarget target(getContext());
    target.addLegalDialect<
        // clang-format off
        BuiltinDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        cpuruntime::CPURuntimeDialect,
        math::MathDialect,
        arith::ArithDialect,
        scf::SCFDialect,
        tensor::TensorDialect,
        linalg::LinalgDialect,
        affine::AffineDialect
        // clang-format on
        >();
    target.addIllegalOp<memref::AllocOp, memref::DeallocOp>();
    // set pattern
    RewritePatternSet patterns(ctx);
    patterns.add<AlignedAllocLowering>(ctx);
    patterns.add<AlignedDeallocLowering>(ctx);

    // Traverse the entire graph to find returnlikeOp operations
    getOperation()->walk([this, ctx](Operation *op) {
      if (op->hasTrait<OpTrait::ReturnLike>()) {
        for (Value operand : op->getOperands()) {
          if (operand.getType().isa<MemRefType>()) {
            // Use buffer analysis to trace the source of the memref
            Value memrefSrc = getViewBase(operand);
            if (auto allocOp = memrefSrc.getDefiningOp<memref::AllocOp>()) {
              // If the source is an alloc, mark the allocOp with a no_trans
              // attribute
              allocOp->setAttr("no_trans", mlir::BoolAttr::get(ctx, true));
            }
          }
        }
      }
    });
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