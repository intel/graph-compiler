//===-- PostProcessPackUnpack.cpp - Simplify pack unpack --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Transforms/Passes.h"
#include "gc/Transforms/Transforms.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_POSTPROCESSPACKUNPACK
#include "gc/Transforms/Passes.h.inc"

using namespace mlir;

// copied from tpp - lower tensor.pack operations that pack constants.
struct LowerConstantPacking : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto constOp = packOp.getSource().getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return failure();
    // Must be a dense constant.
    auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!denseAttr)
      return failure();

    // Bail out if the pack is used as a writing operation i.e., the destination
    // is not a tensor.empty.
    if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(packOp,
                                         "expects empty tensor destination");
    // Pack destination must have static shape.
    if (!packOp.getDestType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          packOp, "expects destination with static shape");

    // If it is a splat constant, skip and let tensor.pack folder to handle this
    // case.
    if (denseAttr.isSplat())
      return rewriter.notifyMatchFailure(
          packOp, "skip pack - existing folder covers constant splats");

    return linalg::lowerPack(rewriter, packOp);
  }
};

static void populateConstantFoldPacking(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<LowerConstantPacking>(ctx);
  linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateRewriteAsConstantPatterns(
      patterns, [](OpOperand *) -> bool { return true; });
  linalg::populateConstantFoldLinalgOperations(
      patterns, [](OpOperand *) -> bool { return true; });
}

struct EliminateDummyPack : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getStaticInnerTiles().empty() &&
        packOp.getInnerTiles().empty()) {
      auto outerPerm = packOp.getOuterDimsPerm();
      for (int64_t i = 0; i < static_cast<int64_t>(outerPerm.size()); ++i) {
        if (outerPerm[i] != i) {
          return rewriter.notifyMatchFailure(packOp, "Not dummy");
        }
      }
      auto source = packOp.getSource();
      rewriter.replaceAllOpUsesWith(packOp, source);
      packOp->erase();
      return success();
    } else {
      return rewriter.notifyMatchFailure(packOp, "Not dummy");
    }
  }
};

struct EliminateDummyUnpack : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    if (unpackOp.getStaticInnerTiles().empty() &&
        unpackOp.getInnerTiles().empty()) {
      auto outerPerm = unpackOp.getOuterDimsPerm();
      for (int64_t i = 0; i < static_cast<int64_t>(outerPerm.size()); ++i) {
        if (outerPerm[i] != i) {
          return rewriter.notifyMatchFailure(unpackOp, "Not dummy");
        }
      }
      auto source = unpackOp.getSource();
      rewriter.replaceAllOpUsesWith(unpackOp, source);
      unpackOp->erase();
      return success();
    } else {
      return rewriter.notifyMatchFailure(unpackOp, "Not dummy");
    }
  }
};

static void populateEliminateDummyPackUnpack(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<EliminateDummyPack, EliminateDummyUnpack>(ctx);
}

class PostProcessPackUnpack
    : public impl::PostProcessPackUnpackBase<PostProcessPackUnpack> {
public:
  using impl::PostProcessPackUnpackBase<
      PostProcessPackUnpack>::PostProcessPackUnpackBase;
  void runOnOperation() final;
};

static void populateSimplifyPacking(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  tensor::PackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ParallelInsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
  ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
      patterns);
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}

void PostProcessPackUnpack::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  // constant fold packing and transpose
  populateConstantFoldPacking(patterns);
  // simplify packing
  populateSimplifyPacking(patterns);
  populateEliminateDummyPackUnpack(patterns);
  // simplify transpose inserted to perform packing
  linalg::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace gc
} // namespace mlir
