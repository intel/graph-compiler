//===--------- Vectorize.cpp - Vectorize structured ops ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "gc/Utils/Transform.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_VECTORIZE
#define GEN_PASS_DEF_VECTORIZE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rw) const override {
    if (!linalg::hasVectorizationImpl(op))
      return rw.notifyMatchFailure(op, "Unsupported Op, cannot vectorize");

    FailureOr<linalg::VectorizationResult> vectorResults =
        linalg::vectorize(rw, op, /*inputVectorSizes=*/{},
                          /*inputScalableVecDims=*/{});
    if (failed(vectorResults)) return failure();

    rw.replaceOp(op, vectorResults->replacements);
    return success();
  }
};

/// Rewrites a full, in-bounds, identity transfer_read of a
/// tensor.collapse_shape or tensor.expand_shape into a read of the reshape
/// source followed by vector.shape_cast, so the read can fold with a
/// transfer_write of the same source.
struct TransferReadOfReshape : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rw) const override {
    Operation *def = readOp.getBase().getDefiningOp();
    Value src;
    if (auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(def))
      src = collapse.getSrc();
    else if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(def))
      src = expand.getSrc();
    else return rw.notifyMatchFailure(readOp, "base is not a tensor reshape");

    // Only when the reshape source is itself a vector write, so the new read
    // forwards to that write. A reshape of anything else (e.g. a collapsed
    // tensor slice) is left alone: the shape_cast this rewrite introduces
    // would sit on a vector.contract operand and block the upstream
    // broadcast/transpose folds for that contraction.
    if (!src.getDefiningOp<vector::TransferWriteOp>())
      return rw.notifyMatchFailure(readOp,
                                   "reshape source is not a vector write");

    if (readOp.getMask() || readOp.hasOutOfBoundsDim() ||
        !readOp.getPermutationMap().isIdentity())
      return rw.notifyMatchFailure(readOp, "not a plain in-bounds read");
    if (!llvm::all_of(readOp.getIndices(),
                      [](Value idx) { return isZeroInteger(idx); }))
      return rw.notifyMatchFailure(readOp, "non-zero indices");

    auto baseTy = cast<RankedTensorType>(readOp.getBase().getType());
    auto srcTy = cast<RankedTensorType>(src.getType());
    VectorType vecTy = readOp.getVectorType();
    if (!baseTy.hasStaticShape() || !srcTy.hasStaticShape() ||
        vecTy.getShape() != baseTy.getShape() || vecTy.isScalable())
      return rw.notifyMatchFailure(readOp, "not a full static read");

    Location loc = readOp.getLoc();
    Value zero = arith::ConstantIndexOp::create(rw, loc, 0);
    SmallVector<Value> indices(srcTy.getRank(), zero);
    SmallVector<bool> inBounds(srcTy.getRank(), true);
    auto newVecTy = VectorType::get(srcTy.getShape(), vecTy.getElementType());
    Value newRead = vector::TransferReadOp::create(
        rw, loc, newVecTy, src, indices, readOp.getPadding(), inBounds);
    rw.replaceOpWithNewOp<vector::ShapeCastOp>(readOp, vecTy, newRead);
    return success();
  }
};

struct Vectorize final : ::mlir::gc::impl::VectorizeBase<Vectorize> {

  void runOnOperation() override {
    auto fn = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<VectorizationPattern, TransferReadOfReshape>(ctx);

    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);

    patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                 linalg::LinalgCopyVTWForwardingPattern>(ctx, /*benefit=*/2);

    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

    patterns.add<linalg::CopyVectorizationPattern>(ctx);

    vector::populateFoldArithExtensionPatterns(patterns);

    linalg::populatePadOpVectorizationPatterns(patterns);
    linalg::populateDecomposePadPatterns(patterns);

    vector::populateVectorStepLoweringPatterns(patterns);

    if (failed(applyPatternsGreedily(fn, std::move(patterns),
                                     GreedyRewriteConfig()))) {
      signalPassFailure();
      return;
    }

    // Inner loops hoisting
    auto isKernelLoop = [](LoopLikeOpInterface loop) {
      return isa<scf::ForallOp>(loop) && loop->hasAttr(GC_ATTR_KERNEL_NAME);
    };
    fn.walk([&](LoopLikeOpInterface loop) {
      if (!isKernelLoop(loop)) moveLoopInvariantCode(loop);
    });
    OpRewriter rw(fn);
    fn.walk([&](LoopLikeOpInterface loop) {
      if (!isKernelLoop(loop)) (void)hoistLoopInvariantSubsets(rw, loop);
    });
  }
};

} // namespace
