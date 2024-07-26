//===- TransformUtils.cpp - Transform utils ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "gc/Transforms/Utils/StructuredOpMatcher.h"
#include "gc/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"

namespace mlir {

namespace linalgx {

namespace utils {

bool isBlockedConvolution(Operation *op) {
  // clang-format off
  using namespace structured_match;
  
  auto isBlockedConv =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .operation(NumDpsInits(EqualsTo(1)))
      .operation(NumDpsInputs(EqualsTo(2)))
      .operation(NumAffineMaps(EqualsTo(3)))
      .operation(NumOfLoops(EqualsTo(9)))
      .operation(VerifyOpProperty(
            mlir::linalg::detail::verifyConvolutionInterface))
      .dim(MatchRange(/*lowerBound=*/0, /*upperBound=*/8),
          {mlir::utils::IteratorType::reduction, 
           mlir::utils::IteratorType::reduction,
           mlir::utils::IteratorType::reduction, 
           mlir::utils::IteratorType::reduction,
           mlir::utils::IteratorType::parallel, 
           mlir::utils::IteratorType::parallel,
           mlir::utils::IteratorType::parallel, 
           mlir::utils::IteratorType::parallel, 
           mlir::utils::IteratorType::parallel})
      .region(MatchOne(0),
            WithOpChain<KindMul, KindAdd>(/*captures=*/nullptr));
  // clang-format on
  return isBlockedConv.match(op);
}

FailureOr<linalg::ContractionDimensions>
isContraction(linalg::LinalgOp linalgOp) {
  using namespace structured_match;

  // clang-format off
  auto maybeContraction =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .operation(NumDpsInits(EqualsTo(1)))
      .operation(NumDpsInputs(EqualsTo(2)))
      .operation(NumAffineMaps(EqualsTo(3)))
      .region(MatchOne(0),
            WithOpChain<arith::MulFOp,
                        arith::AddFOp>(/*captures=*/nullptr));
  // clang-format on
  if (!maybeContraction.match(linalgOp))
    return failure();

  auto dims = linalg::inferContractionDims(linalgOp);
  if (failed(dims) ||
      (dims->m.size() < 1 || dims->n.size() < 1 || dims->k.size() < 1)) {
    return failure();
  }
  return dims;
}

static std::optional<int64_t> getConstantRange(const Range &range) {
  std::optional<int64_t> stride = getConstantIntValue(range.stride);
  if (!stride || *stride != 1)
    return std::nullopt;
  std::optional<int64_t> offset = getConstantIntValue(range.offset);
  if (!offset)
    return std::nullopt;
  std::optional<int64_t> size = getConstantIntValue(range.size);
  if (!size)
    return std::nullopt;
  return (*size - *offset);
}

static bool validateFullTilesOnDim(TilingInterface tileOp,
                                   const OpFoldResult &tile, size_t dim,
                                   int64_t minTileFactor) {
  OpBuilder builder(tileOp);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(tileOp.getOperation()).getIterationDomain(builder);
  if (dim >= iterationDomain.size())
    return false;

  auto tileSize = getConstantIntValue(tile);
  auto rangeOnDim = getConstantRange(iterationDomain[dim]);

  // If the tile factor or the range are non-constant, the tile size is
  // considered to be valid.
  if (!tileSize || !rangeOnDim)
    return true;

  // Corner case: Tiling with '0' along 'dim' is valid - no tiling.
  if (*tileSize == 0)
    return true;

  // Corner case: Tiling '1' with '1' is valid.
  if (*tileSize == 1 && *rangeOnDim == 1)
    return true;

  return (*rangeOnDim % *tileSize == 0) &&
         (*rangeOnDim / *tileSize >= minTileFactor);
}

bool validateFullTilesOnDims(TilingInterface tileOp,
                             ArrayRef<OpFoldResult> tiles,
                             ArrayRef<size_t> dims, int64_t minTileFactor) {
  if (!dims.empty() && dims.size() != tiles.size())
    return false;

  // If dims is empty we start from the outermost dim '0'.
  SmallVector<size_t> dimsToCheck;
  if (dims.empty())
    dimsToCheck = llvm::to_vector(llvm::seq<size_t>(0, tiles.size()));
  else
    dimsToCheck = llvm::to_vector(dims);
  assert(dimsToCheck.size() == tiles.size());

  for (auto dim : llvm::enumerate(dimsToCheck)) {
    if (!validateFullTilesOnDim(tileOp, tiles[dim.index()], dim.value(),
                                minTileFactor))
      return false;
  }
  return true;
}

namespace {

// Convert scf.for to scf.forall after fusion.
struct ConvertToForAll : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto metadata =
        forOp->getAttrOfType<StringAttr>(linalgx::utils::kLoopParallel);
    if (!metadata || metadata.getValue() != linalgx::utils::kLoopRoot)
      return failure();
    if (forOp.getNumRegionIterArgs() != 1)
      return failure();

    SmallVector<scf::ForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, forOp);
    if (nestedLoops.size() == 0)
      return failure();

    SmallVector<Value> loopArgs;
    SmallVector<OpFoldResult> lbs, ubs, steps;
    scf::ForOp innerMostLoop = nestedLoops[nestedLoops.size() - 1];
    for (scf::ForOp &currentLoop : nestedLoops) {
      if (currentLoop.getNumRegionIterArgs() != 1)
        return failure();
      loopArgs.push_back(currentLoop.getInductionVar());
      lbs.push_back(currentLoop.getLowerBound());
      ubs.push_back(currentLoop.getUpperBound());
      steps.push_back(currentLoop.getStep());
      if (currentLoop == innerMostLoop) {
        // We can only replace if the last operation before the terminator is
        // an insert slice.
        auto yieldOp =
            cast<scf::YieldOp>(currentLoop.getBody()->getTerminator());
        auto insertSlice =
            yieldOp.getOperands()[0].getDefiningOp<tensor::InsertSliceOp>();
        if (!insertSlice)
          return failure();
        loopArgs.push_back(currentLoop.getRegionIterArg(0));
      }
    }

    rewriter.replaceOpWithNewOp<scf::ForallOp>(
        forOp, lbs, ubs, steps, ValueRange{forOp.getInitArgs()},
        /*mapping=*/std::nullopt,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange regionArgs) {
          IRMapping mapping;
          assert(loopArgs.size() == regionArgs.size() &&
                 "expect same region args");
          mapping.map(loopArgs, regionArgs);
          Block *innerLoopBlock = nestedLoops[nestedLoops.size() - 1].getBody();
          auto yieldOp = cast<scf::YieldOp>(innerLoopBlock->getTerminator());
          auto insertSlice =
              yieldOp.getOperands()[0].getDefiningOp<tensor::InsertSliceOp>();
          assert(insertSlice && "must be an insert slice");
          for (auto &nestedOp : innerLoopBlock->without_terminator()) {
            if (&nestedOp == insertSlice.getOperation()) {
              auto term = nestedBuilder.create<scf::InParallelOp>(loc);
              nestedBuilder.setInsertionPointToStart(term.getBody());
              Value sourceVal = mapping.lookup(insertSlice.getSource());
              Value destVal = mapping.lookup(insertSlice.getDest());
              SmallVector<OpFoldResult> offsets;
              for (OpFoldResult offset : insertSlice.getMixedOffsets()) {
                if (auto valueOffset = dyn_cast<Value>(offset))
                  offsets.push_back(mapping.lookupOrDefault(valueOffset));
                else
                  offsets.push_back(offset);
              }
              SmallVector<OpFoldResult> sizes;
              for (OpFoldResult size : insertSlice.getMixedSizes()) {
                if (auto valueSize = dyn_cast<Value>(size))
                  sizes.push_back(mapping.lookupOrDefault(valueSize));
                else
                  sizes.push_back(size);
              }
              SmallVector<OpFoldResult> strides;
              for (OpFoldResult stride : insertSlice.getMixedStrides()) {
                if (auto valueStride = dyn_cast<Value>(stride))
                  strides.push_back(mapping.lookupOrDefault(valueStride));
                else
                  strides.push_back(stride);
              }
              assert(offsets.size() == sizes.size());
              assert(offsets.size() == strides.size());

              nestedBuilder.create<tensor::ParallelInsertSliceOp>(
                  loc, sourceVal, destVal, offsets, sizes, strides);
              continue;
            }
            Operation *clone = nestedBuilder.clone(nestedOp, mapping);
            mapping.map(nestedOp.getResults(), clone->getResults());
          }
        });
    return success();
  }
};

} // namespace

// Populate patterns to rewrite scf.for with scf.forall.
void populateScfForToForAllRewritePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertToForAll>(patterns.getContext());
}

} // namespace utils

} // namespace linalgx

} // namespace mlir
