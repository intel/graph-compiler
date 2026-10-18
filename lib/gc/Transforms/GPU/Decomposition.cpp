//===--------- Decomposition.cpp - Decompose aggregated ops ------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_DECOMPOSITION
#define GEN_PASS_DEF_DECOMPOSITION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Decomposes any operation that implements AggregatedOpInterface by calling
// its decomposeOperation method.
struct DecomposeAggregatedOp : public RewritePattern {
  explicit DecomposeAggregatedOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto decomposableOp = dyn_cast<linalg::AggregatedOpInterface>(op);
    if (!decomposableOp || !isa<linalgx::AttentionOp>(op)) return failure();

    FailureOr<SmallVector<Value>> maybeNewResults =
        decomposableOp.decomposeOperation(rewriter);
    if (failed(maybeNewResults)) return failure();

    rewriter.replaceOp(op, maybeNewResults.value()[0]);
    return success();
  }
};

// pack src into dst inner_dims_pos=[gi] inner_tiles=[B] (no outer perm)
// → expand_shape src [[0,..,gi-1, gi, gi+1], [gi+2,..]] (trailing 1:2 split)
struct PackToExpandShape : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::PackOp op,
                                PatternRewriter &rw) const override {
    if (!op.getOuterDimsPerm().empty()) return failure();
    if (op.getPaddingValue()) return failure();
    auto innerDims = op.getInnerDimsPos();
    auto innerTiles = op.getMixedTiles();
    if (innerDims.size() != 1) return failure();
    auto tileSize = getConstantIntValue(innerTiles[0]);
    if (!tileSize) return failure();
    int64_t gi = innerDims[0];
    auto srcType = op.getSourceType();
    auto dstType = op.getDestType();
    // Only trailing dim: tile is appended at dst end, expand_shape needs
    // contiguous indices.
    if (gi != srcType.getRank() - 1) return failure();
    // Verify dst shape matches expected pack layout.
    if (dstType.getDimSize(gi) != srcType.getDimSize(gi) / *tileSize)
      return failure();
    if (dstType.getDimSize(dstType.getRank() - 1) != *tileSize)
      return failure();

    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0, r = srcType.getRank(); i < r; ++i) {
      if (i == gi) reassoc.push_back({i, r});
      else reassoc.push_back({i});
    }
    rw.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, dstType, op.getSource(),
                                                 reassoc);
    return success();
  }
};

// unpack src into dst inner_dims_pos=[gi] inner_tiles=[B] (no outer perm)
// → collapse_shape src (trailing 2:1 merge)
struct UnPackToCollapseShape : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::UnPackOp op,
                                PatternRewriter &rw) const override {
    if (!op.getOuterDimsPerm().empty()) return failure();
    auto innerDims = op.getInnerDimsPos();
    auto innerTiles = op.getMixedTiles();
    if (innerDims.size() != 1) return failure();
    auto tileSize = getConstantIntValue(innerTiles[0]);
    if (!tileSize) return failure();
    int64_t gi = innerDims[0];
    auto srcType = op.getSourceType();
    auto dstType = op.getDestType();
    // Compute expected collapsed dim size from source.
    int64_t collapsedDim = srcType.getDimSize(gi) * *tileSize;
    if (ShapedType::isDynamic(collapsedDim)) return failure();
    // dst[gi] must match or be dynamic.
    int64_t dstDim = dstType.getDimSize(gi);
    if (!ShapedType::isDynamic(dstDim) && dstDim != collapsedDim)
      return failure();
    // Build static result type (may differ from dstType if dstType has dynamic
    SmallVector<int64_t> resultShape(dstType.getShape());
    resultShape[gi] = collapsedDim;
    auto resultType =
        RankedTensorType::get(resultShape, dstType.getElementType());
    // Build reassociation: merge gi with last src dim, all others trivial.
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < dstType.getRank(); ++i) {
      if (i == gi) reassoc.push_back({i, srcType.getRank() - 1});
      else reassoc.push_back({i});
    }
    rw.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, resultType,
                                                   op.getSource(), reassoc);
    return success();
  }
};

struct Decomposition final : gc::impl::DecompositionBase<Decomposition> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<DecomposeAggregatedOp>(ctx);
    patterns.add<PackToExpandShape, UnPackToCollapseShape>(ctx, /*benefit=*/2);
    linalg::populateDecomposePackUnpackPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns),
                                     GreedyRewriteConfig()))) {
      signalPassFailure();
    }
  }
};

} // namespace
