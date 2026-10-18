//===-- FoldPadRoundtrip.cpp -------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_FOLDPADROUNDTRIP
#define GEN_PASS_DEF_FOLDPADROUNDTRIP
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

static bool allZero(ArrayRef<OpFoldResult> ofrs) {
  return llvm::all_of(ofrs, [](OpFoldResult ofr) {
    return getConstantIntValue(ofr) == static_cast<int64_t>(0);
  });
}

static bool allOne(ArrayRef<OpFoldResult> ofrs) {
  return llvm::all_of(ofrs, [](OpFoldResult ofr) {
    return getConstantIntValue(ofr) == static_cast<int64_t>(1);
  });
}

struct FoldPadOfExtractSlice final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType padType = padOp.getResultType();
    if (!padType.hasStaticShape()) return failure();
    if (!allZero(padOp.getMixedLowPad())) return failure();

    auto sliceOp = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp) return failure();
    if (!allZero(sliceOp.getMixedOffsets()) ||
        !allOne(sliceOp.getMixedStrides()))
      return failure();

    auto srcType = dyn_cast<RankedTensorType>(sliceOp.getSource().getType());
    if (!srcType || srcType != padType) return failure();

    rewriter.replaceOp(padOp, sliceOp.getSource());
    return success();
  }
};

struct FoldPadOfDynamicEmpty final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType padType = padOp.getResultType();
    if (!padType.hasStaticShape()) return failure();
    if (!allZero(padOp.getMixedLowPad())) return failure();
    if (!padOp.getSource().getDefiningOp<tensor::EmptyOp>()) return failure();

    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(padOp, padType.getShape(),
                                                 padType.getElementType());
    return success();
  }
};

struct FoldPadRoundtrip final
    : gc::impl::FoldPadRoundtripBase<FoldPadRoundtrip> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FoldPadOfExtractSlice, FoldPadOfDynamicEmpty>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
