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
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
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
    vector::populateDropInnerMostUnitDimsXferOpPatterns(patterns);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);
    patterns.add<DropBroadcastUnitDims>(&getContext());
    vector::BroadcastOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(fn, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
