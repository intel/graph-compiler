//===- TensorReshapeToLinalg.cpp - Replace reshape with pack/unpack -*-C++-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>

namespace mlir::gc {
#define GEN_PASS_DECL_TENSORRESHAPETOLINALG
#define GEN_PASS_DEF_TENSORRESHAPETOLINALG
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

// Perm that moves last outer dim to position gi:
//   [0,..,gi-1, outerRank-1, gi,..,outerRank-2]
static SmallVector<int64_t> permForA1(size_t gi, int64_t outerRank) {
  SmallVector<int64_t> perm;
  for (int64_t i = 0; i < (int64_t)gi; ++i) perm.push_back(i);
  perm.push_back(outerRank - 1);
  for (int64_t i = (int64_t)gi; i < outerRank - 1; ++i) perm.push_back(i);
  return perm;
}

template <typename ReshapeOp>
struct ReshapeConst : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rw) const override {
    auto constOp = op.getSrc().template getDefiningOp<arith::ConstantOp>();
    if (!constOp) return failure();
    auto newAttr =
        DenseElementsAttr::get(op.getResultType(), constOp.getValue());
    rw.replaceOpWithNewOp<arith::ConstantOp>(op, op.getResultType(), newAttr);
    return success();
  }
};

// expand_shape A -> 1x...x1xAx1x...x1:
//    linalg.broadcast dims=[reassoc[gi][i]]
struct ExpandUnitsToBroadcast : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp op,
                                PatternRewriter &rw) const override {
    auto resultType = op.getResultType();
    auto reassoc = op.getReassociationIndices();
    SmallVector<int64_t> broadcastDims;
    for (auto group : reassoc) {
      if (group.size() < 2) continue;
      int64_t nonUnitCount = 0;
      for (int64_t dimIdx : group) {
        if (resultType.getDimSize(dimIdx) == 1) broadcastDims.push_back(dimIdx);
        else if (++nonUnitCount != 1) return failure();
      }
    }
    if (broadcastDims.empty()) return failure();

    Value dest = tensor::EmptyOp::create(rw, op.getLoc(), resultType.getShape(),
                                         resultType.getElementType());
    rw.replaceOpWithNewOp<linalg::BroadcastOp>(op, op.getSrc(), dest,
                                               broadcastDims);
    return success();
  }
};

// collapse_shape src<...x1xAx1x...> → dst<...xA...>
// Each non-trivial group must have exactly one non-unit src dim (i.e. the group
// only collapses away leading/trailing unit dims).
// Inverse of ExpandUnitsToBroadcast.
//
// For each non-trivial group at position gi where src[group[0]]==1:
//   linalg.unpack src into dst
//     inner_dims_pos=[R_dst - 1]
//     inner_tiles=[src.getDimSize(R_src - 1)]
//     outer_dims_perm=permForA1(gi, R_dst)
// This moves the last dim (the "inner tile") to position gi in the output.
struct CollapseUnitsToUnPack
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp op,
                                PatternRewriter &rw) const override {
    auto reassoc = op.getReassociationIndices();
    auto srcType = op.getSrcType();
    auto dstType = op.getResultType();

    // Find the single non-trivial group (size == 2).
    int64_t gi = 1;
    for (auto [i, group] : llvm::enumerate(reassoc)) {
      if (group.size() == 1) continue;
      if (group.size() != 2 || gi != 1) return failure();
      gi = i;
    }
    if (gi == -1 || srcType.getDimSize(reassoc[gi][0]) != 1) return failure();

    int64_t lastSrcDim = srcType.getDimSize(srcType.getRank() - 1);
    if (ShapedType::isDynamic(lastSrcDim)) return failure();

    int64_t r = dstType.getRank();
    SmallVector<int64_t> innerDimsPos{r - 1};
    SmallVector<OpFoldResult> innerTiles{rw.getIndexAttr(lastSrcDim)};
    SmallVector<int64_t> outerDimsPerm = permForA1(gi, r);

    Value dest = linalg::UnPackOp::createDestinationTensor(
        rw, op.getLoc(), op.getSrc(), innerTiles, innerDimsPos, outerDimsPerm);
    rw.replaceOpWithNewOp<linalg::UnPackOp>(op, op.getSrc(), dest, innerDimsPos,
                                            innerTiles, outerDimsPerm);
    return success();
  }
};

// expand_shape ...xA -> ...xAx(A/B)
//   linalg.pack inner_dims=[n - 1] tiles=[B]
// collapse_shape ...xAxB -> ...x(A*B)
//   linalg.unpack inner_dims=[n - 1] tiles=[B]
template <typename ReshapeOp,
          typename RepackOp = std::conditional_t<
              std::is_same_v<ReshapeOp, tensor::ExpandShapeOp>, linalg::PackOp,
              linalg::UnPackOp>>
struct InnerDimReshapeToRepack : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rw) const override {
    auto reassoc = op.getReassociationIndices();
    int64_t s = reassoc.size();
    int64_t tile = 0;
    for (auto [gi, group] : llvm::enumerate(reassoc)) {
      if (group.size() < 2) continue;
      if (group.size() != 2 || (int64_t)gi != s - 1 || group[0] != s - 1 ||
          group[1] != s)
        return failure();
      if constexpr (std::is_same_v<ReshapeOp, tensor::ExpandShapeOp>)
        tile = op.getResultType().getDimSize(s);
      else tile = op.getSrcType().getDimSize(s);
    }
    if (tile == 0 || ShapedType::isDynamic(tile)) return failure();

    SmallVector<int64_t> innerDimsPos{s - 1};
    SmallVector<OpFoldResult> innerTiles{rw.getIndexAttr(tile)};
    Value dest = RepackOp::createDestinationTensor(
        rw, op.getLoc(), op.getSrc(), innerTiles, innerDimsPos, {});
    rw.replaceOpWithNewOp<RepackOp>(op, op.getSrc(), dest, innerDimsPos,
                                    innerTiles);
    return success();
  }
};

// Fold collapse_shape(bufferization.to_tensor restrict writable memref)
// → bufferization.to_tensor(memref.reinterpret_cast memref to collapsed
// shape). Only for single non-trivial group with A=1 or trailing group,
// static shapes.
template <typename ReshapeOp>
struct FoldReshapeOfToTensor : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rw) const override {
    auto toTensor =
        op.getSrc().template getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensor || !toTensor.getRestrict() || !toTensor.getWritable())
      return failure();
    auto memrefVal = toTensor.getBuffer();
    auto memrefType = dyn_cast<MemRefType>(memrefVal.getType());
    if (!memrefType || !memrefType.hasStaticShape()) return failure();
    auto resType = op.getResultType();
    if (!resType.hasStaticShape()) return failure();

    // Build flat strides for the result shape (row-major).
    SmallVector<int64_t> sizes(resType.getShape());
    SmallVector<int64_t> strides(sizes.size());
    strides.back() = 1;
    for (int i = (int)strides.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * sizes[i + 1];

    auto castType =
        MemRefType::get(sizes, memrefType.getElementType(),
                        StridedLayoutAttr::get(rw.getContext(), 0, strides),
                        memrefType.getMemorySpace());
    auto cast = memref::ReinterpretCastOp::create(
        rw, op.getLoc(), castType, memrefVal, /*offset=*/int64_t(0),
        ArrayRef<int64_t>(sizes), ArrayRef<int64_t>(strides));
    rw.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, cast,
                                                     /*restrict=*/true,
                                                     /*writable=*/true);
    if (toTensor->use_empty()) rw.eraseOp(toTensor);
    return success();
  }
};

// Fold reshape(tensor.empty) → tensor.empty with result shape.
template <typename ReshapeOp>
struct FoldReshapeOfEmpty : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rw) const override {
    auto emptyOp = op.getSrc().template getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp || !emptyOp->hasOneUse()) return failure();
    auto resType = op.getResultType();
    if (!resType.hasStaticShape()) return failure();
    rw.replaceOpWithNewOp<tensor::EmptyOp>(op, resType.getShape(),
                                           resType.getElementType());
    rw.eraseOp(emptyOp);
    return success();
  }
};

// Fold reshape(linalg.fill(empty)) → linalg.fill(empty) with result shape.
template <typename ReshapeOp>
struct FoldReshapeOfFill : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rw) const override {
    auto fillOp = op.getSrc().template getDefiningOp<linalg::FillOp>();
    if (!fillOp || !fillOp->hasOneUse()) return failure();
    auto emptyOp =
        fillOp.getOutputs()[0].template getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp || !emptyOp->hasOneUse()) return failure();
    auto resType = op.getResultType();
    if (!resType.hasStaticShape()) return failure();
    Value newEmpty = tensor::EmptyOp::create(
        rw, op.getLoc(), resType.getShape(), resType.getElementType());
    Value newFill =
        linalg::FillOp::create(rw, op.getLoc(), fillOp.getInputs(), newEmpty)
            .getResult(0);
    rw.replaceOp(op, newFill);
    rw.eraseOp(fillOp);
    rw.eraseOp(emptyOp);
    return success();
  }
};

struct TensorReshapeToLinalg
    : mlir::gc::impl::TensorReshapeToLinalgBase<TensorReshapeToLinalg> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ReshapeConst<tensor::CollapseShapeOp>,
                 ReshapeConst<tensor::ExpandShapeOp>>(ctx, 3);
    patterns.add<FoldReshapeOfEmpty<tensor::CollapseShapeOp>,
                 FoldReshapeOfEmpty<tensor::ExpandShapeOp>,
                 FoldReshapeOfFill<tensor::CollapseShapeOp>,
                 FoldReshapeOfFill<tensor::ExpandShapeOp>,
                 FoldReshapeOfToTensor<tensor::CollapseShapeOp>,
                 FoldReshapeOfToTensor<tensor::ExpandShapeOp>>(ctx, 2);
    patterns.add<CollapseUnitsToUnPack, ExpandUnitsToBroadcast>(ctx, 1);
    patterns.add<InnerDimReshapeToRepack<tensor::CollapseShapeOp>,
                 InnerDimReshapeToRepack<tensor::ExpandShapeOp>>(ctx, 0);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
