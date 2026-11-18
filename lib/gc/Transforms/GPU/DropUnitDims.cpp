//===- DropUnitDims.cpp - Drop unit dims ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_DROPUNITDIMS
#define GEN_PASS_DEF_DROPUNITDIMS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
static VectorType dropNonScalableUnitDims(VectorType type,
                                          VectorType broadcastType) {
  SmallVector<int64_t> shape;
  SmallVector<bool> scalableDims;
  int64_t offset = broadcastType.getRank() - type.getRank();
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    int64_t broadcastDim = offset + dim;
    if (broadcastType.getDimSize(broadcastDim) == 1 &&
        !broadcastType.getScalableDims()[broadcastDim])
      continue;
    shape.push_back(type.getDimSize(dim));
    scalableDims.push_back(type.getScalableDims()[dim]);
  }
  return VectorType::get(shape, type.getElementType(), scalableDims);
}

struct DropTransferReadUnitDims : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto type = read.getVectorType();
    auto reducedType = dropNonScalableUnitDims(type, type);
    if (type == reducedType || reducedType.getRank() == 0 || read.getMask() ||
        read.hasOutOfBoundsDim() || read.getTransferRank() != type.getRank() ||
        isa<vector::MaskOp>(read->getParentOp()))
      return failure();

    SmallVector<AffineExpr> results;
    SmallVector<bool> inBounds;
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (type.getDimSize(dim) == 1 && !type.getScalableDims()[dim]) continue;
      results.push_back(read.getPermutationMap().getResult(dim));
      inBounds.push_back(true);
    }
    auto map = AffineMap::get(read.getPermutationMap().getNumDims(),
                              read.getPermutationMap().getNumSymbols(), results,
                              rewriter.getContext());
    Value source = read.getBase();
    SmallVector<Value> indices(read.getIndices());
    if (!map.isMinorIdentity()) {
      auto memrefType = dyn_cast<MemRefType>(read.getBase().getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      if (!memrefType ||
          failed(memrefType.getStridesAndOffset(strides, offset)))
        return failure();
      int64_t expectedStride = 1;
      for (AffineExpr result : llvm::reverse(results)) {
        auto dim = dyn_cast<AffineDimExpr>(result);
        if (!dim || strides[dim.getPosition()] != expectedStride ||
            memrefType.isDynamicDim(dim.getPosition()))
          return failure();
        expectedStride *= memrefType.getDimSize(dim.getPosition());
      }
      if (reducedType.getRank() > 1) {
        if (type.isScalable()) return failure();
        SmallVector<bool> retained(memrefType.getRank(), false);
        for (AffineExpr result : results)
          retained[cast<AffineDimExpr>(result).getPosition()] = true;
        auto zero = rewriter.getIndexAttr(0);
        auto one = rewriter.getIndexAttr(1);
        SmallVector<OpFoldResult> offsets(memrefType.getRank(), zero);
        auto sizes = memref::getMixedSizes(rewriter, read.getLoc(), source);
        SmallVector<OpFoldResult> steps(memrefType.getRank(), one);
        SmallVector<int64_t> shape;
        SmallVector<AffineExpr> replacements;
        indices.clear();
        for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
          if (retained[dim]) {
            replacements.push_back(rewriter.getAffineDimExpr(shape.size()));
            shape.push_back(memrefType.getDimSize(dim));
            indices.push_back(read.getIndices()[dim]);
          } else {
            replacements.push_back(rewriter.getAffineConstantExpr(0));
            offsets[dim] = read.getIndices()[dim];
            sizes[dim] = one;
          }
        }
        auto viewType = memref::SubViewOp::inferRankReducedResultType(
            shape, memrefType, offsets, sizes, steps);
        source = memref::SubViewOp::create(rewriter, read.getLoc(),
                                           cast<MemRefType>(viewType), source,
                                           offsets, sizes, steps);
        SmallVector<AffineExpr> viewResults;
        for (AffineExpr result : results)
          viewResults.push_back(result.replaceDims(replacements));
        map =
            AffineMap::get(shape.size(), 0, viewResults, rewriter.getContext());
      }
    }
    auto reduced = vector::TransferReadOp::create(
        rewriter, read.getLoc(), reducedType, source, indices,
        AffineMapAttr::get(map), read.getPadding(), Value{},
        rewriter.getBoolArrayAttr(inBounds));
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(read, type, reduced);
    return success();
  }
};

struct DropTransferWriteLeadingUnitDims
    : OpRewritePattern<vector::TransferWriteOp> {
  DropTransferWriteLeadingUnitDims(MLIRContext *context,
                                   std::unique_ptr<RewritePattern> pattern)
      : OpRewritePattern(context, pattern->getBenefit()),
        upstreamPattern(std::move(pattern)) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    Value value = write.getVector();
    while (true) {
      if (auto shapeCast = value.getDefiningOp<vector::ShapeCastOp>())
        value = shapeCast.getSource();
      else if (auto transpose = value.getDefiningOp<vector::TransposeOp>())
        value = transpose.getVector();
      else break;
    }
    if (value.getDefiningOp<vector::TransferReadOp>()) return failure();
    return upstreamPattern->matchAndRewrite(write, rewriter);
  }

private:
  std::unique_ptr<RewritePattern> upstreamPattern;
};

struct DropTransposeUnitDims : OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    if (transpose.getVector().getDefiningOp<vector::TransferReadOp>())
      return failure();
    auto type = transpose.getSourceVectorType();
    auto reducedType = dropNonScalableUnitDims(type, type);
    if (type == reducedType || reducedType.getRank() == 0) return failure();
    SmallVector<int64_t> dimIndices(type.getRank());
    int64_t nextDim = 0;
    for (int64_t dim = 0; dim < type.getRank(); ++dim)
      if (type.getDimSize(dim) != 1 || type.getScalableDims()[dim])
        dimIndices[dim] = nextDim++;
    SmallVector<int64_t> permutation;
    for (int64_t dim : transpose.getPermutation())
      if (type.getDimSize(dim) != 1 || type.getScalableDims()[dim])
        permutation.push_back(dimIndices[dim]);
    auto source = vector::ShapeCastOp::create(
        rewriter, transpose.getLoc(), reducedType, transpose.getVector());
    auto reduced = vector::TransposeOp::create(rewriter, transpose.getLoc(),
                                               source, permutation);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transpose, transpose.getResultVectorType(), reduced);
    return success();
  }
};

struct DropBroadcastUnitDims : OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcast,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<VectorType>(broadcast.getType());
    auto reducedType = dropNonScalableUnitDims(resultType, resultType);
    if (resultType == reducedType) return failure();

    Value source = broadcast.getSource();
    if (auto sourceType = dyn_cast<VectorType>(source.getType())) {
      auto reducedSourceType = dropNonScalableUnitDims(sourceType, resultType);
      if (sourceType != reducedSourceType)
        source = vector::ShapeCastOp::create(rewriter, broadcast.getLoc(),
                                             reducedSourceType, source);
    }
    auto reduced = vector::BroadcastOp::create(rewriter, broadcast.getLoc(),
                                               reducedType, source);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(broadcast, resultType,
                                                     reduced);
    return success();
  }
};

struct DropUnitDims final : gc::impl::DropUnitDimsBase<DropUnitDims> {

  void runOnOperation() override {
    auto fn = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<DropTransferReadUnitDims>(&getContext(), PatternBenefit(2));
    vector::populateDropInnerMostUnitDimsXferOpPatterns(patterns);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    RewritePatternSet writePatterns(&getContext());
    vector::populateCastAwayVectorLeadingOneDimPatterns(writePatterns);
    for (auto &pattern : writePatterns.getNativePatterns()) {
      auto root = pattern->getRootKind();
      if (root &&
          root->getStringRef() == vector::TransferWriteOp::getOperationName())
        patterns.add<DropTransferWriteLeadingUnitDims>(&getContext(),
                                                       std::move(pattern));
    }
    RewritePatternSet upstreamPatterns(&getContext());
    vector::populateDropUnitDimWithShapeCastPatterns(upstreamPatterns);
    for (auto &pattern : upstreamPatterns.getNativePatterns())
      if (pattern->getRootTraitID() == TypeID::get<OpTrait::Elementwise>())
        patterns.add(std::move(pattern));
    patterns.add<DropTransposeUnitDims>(&getContext());
    patterns.add<DropBroadcastUnitDims>(&getContext());
    vector::BroadcastOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(fn, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
