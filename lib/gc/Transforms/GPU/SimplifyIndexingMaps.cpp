//===-- SimplifyIndexingMaps.cpp ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SIMPLIFYINDEXINGMAPS
#define GEN_PASS_DEF_SIMPLIFYINDEXINGMAPS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

/// Matches `expr` against `d floordiv c`. For the static range R of `d`,
/// returns the number of distinct index values (1 if R <= c, R / c if c
/// divides R) and sets groupSize to the number of iterations per value.
static std::optional<int64_t> getGroupCount(linalg::GenericOp op,
                                            AffineExpr expr, int64_t &groupSize,
                                            unsigned &dimPos) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin || bin.getKind() != AffineExprKind::FloorDiv) return std::nullopt;
  auto dim = dyn_cast<AffineDimExpr>(bin.getLHS());
  auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
  if (!dim || !cst || cst.getValue() <= 0) return std::nullopt;
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  int64_t range = ranges[dim.getPosition()];
  if (ShapedType::isDynamic(range)) return std::nullopt;
  dimPos = dim.getPosition();
  if (range <= cst.getValue()) {
    groupSize = range;
    return 1;
  }
  if (range % cst.getValue() != 0) return std::nullopt;
  groupSize = cst.getValue();
  return range / groupSize;
}

/// Rewrites an input map result `d floordiv c` to `d`. The operand is
/// broadcast along a new dimension of size groupSize inserted after the
/// indexed dimension, then collapsed back so each value repeats groupSize
/// times along that dimension.
struct ExpandGroupedOperands final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) return failure();
    MLIRContext *ctx = op.getContext();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();

    for (OpOperand *operand : op.getDpsInputOperands()) {
      auto srcTy = dyn_cast<RankedTensorType>(operand->get().getType());
      if (!srcTy || !srcTy.hasStaticShape()) continue;
      AffineMap map = maps[operand->getOperandNumber()];
      for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
        int64_t groupSize = 0;
        unsigned dimPos = 0;
        std::optional<int64_t> groups =
            getGroupCount(op, expr, groupSize, dimPos);
        // The operand must hold exactly one value per group.
        if (!groups || srcTy.getDimSize(idx) != *groups) continue;

        Location loc = op.getLoc();
        rewriter.setInsertionPoint(op);
        SmallVector<int64_t> expandedShape(srcTy.getShape());
        expandedShape.insert(expandedShape.begin() + idx + 1, groupSize);
        Value empty = tensor::EmptyOp::create(rewriter, loc, expandedShape,
                                              srcTy.getElementType());
        auto bcast = linalg::BroadcastOp::create(
            rewriter, loc, operand->get(), empty,
            ArrayRef<int64_t>{static_cast<int64_t>(idx) + 1});
        SmallVector<ReassociationIndices> reassociation;
        for (int64_t i = 0, e = srcTy.getRank(); i < e; ++i) {
          if (i == static_cast<int64_t>(idx))
            reassociation.push_back({i, i + 1});
          else
            reassociation.push_back(
                {i < static_cast<int64_t>(idx) ? i : i + 1});
        }
        Value collapsed = tensor::CollapseShapeOp::create(
            rewriter, loc, bcast.getResult().front(), reassociation);

        SmallVector<AffineExpr> results(map.getResults());
        results[idx] = getAffineDimExpr(dimPos, ctx);
        maps[operand->getOperandNumber()] =
            AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, ctx);
        rewriter.modifyOpInPlace(op, [&] {
          op->setOperand(operand->getOperandNumber(), collapsed);
          op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
        });
        return success();
      }
    }
    return failure();
  }
};

struct SimplifyIndexingMaps final
    : gc::impl::SimplifyIndexingMapsBase<SimplifyIndexingMaps> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ExpandGroupedOperands>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
