//===-- TileNamed.cpp - Tile Named Linalg Ops -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_TILELINALGNAMED
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {

// scf::SCFTilingOptions &
// scf::SCFTilingOptions::setTileSizes(ArrayRef<OpFoldResult> ts) {
//   assert(!tileSizeComputationFunction && "tile sizes already set");
//   auto tileSizes = llvm::to_vector(ts);
//   tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
//     return tileSizes;
//   };
//   return *this;
// }

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

std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<LoopLikeOpInterface> loops) {
  std::optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = dyn_cast<BlockArgument>(source->get())) {
    auto loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = loop.getTiedLoopInit(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend())
    destinationIterArg = source;
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
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

// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  std::optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  std::optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  std::optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
}

/// Returns the bounded tile size given the current `iv`, `loopRange` and
/// `tileSize`, i.e., `min(tileSize, range.end() - iv)`.
static OpFoldResult getBoundedTileSize(OpBuilder &b, Location loc,
                                       Range loopRange, Value iv,
                                       OpFoldResult tileSize) {
  std::optional<int64_t> ts = getConstantIntValue(tileSize);
  if (ts && ts.value() == 1)
    return tileSize;

  if (tileDividesIterationDomain(
          Range{loopRange.offset, loopRange.size, tileSize}))
    return tileSize;

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable of the tiled
  // loop.
  AffineExpr s0, s1, d0;
  bindDims(b.getContext(), d0);
  bindSymbols(b.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, b.getContext());
  Value size = getValueOrCreateConstantIndexOp(b, loc, loopRange.size);
  return affine::makeComposedFoldedAffineMin(
      b, loc, minMap, SmallVector<OpFoldResult>{iv, tileSize, size});
}

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<int64_t>
fillInterchangeVector(ArrayRef<int64_t> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<int64_t> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<int64_t>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Clones the operation and updates the destination if the operation
/// implements the `DestinationStyleOpInterface`.
static Operation *cloneOpAndUpdateDestinationArgs(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange newDestArgs) {
  Operation *clonedOp = rewriter.clone(*op);
  if (newDestArgs.empty())
    return clonedOp;
  if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp))
    destinationStyleOp.getDpsInitsMutable().assign(newDestArgs);
  return clonedOp;
}

/// A function that allows returning additional yielded values during
/// `yieldTiledValuesAndReplace`.
/// - `ivs` induction variable for the loop.
/// - `newBbArgs` basic block arguments corresponding to newly added iter_args.
/// - `tiledValues` the tiled values to return. Must be of same size as
///   `newbbArgs`, each element of this array is inserted into the corresponding
///   element in `newbbArgs`.
/// - `resultOffsets` is of the same size as `tiledValues` and represents
///   the offsets to use when inserting corresponding element from `tiledValues`
///   into the element from `newBbArgs`.
/// - `resultSizes` is of the same size as `tiledValues` and represents
///   the size of the corresponding element from `tiledValues` inserted into
///   the element from `newBbArgs`.
/// In case the method needs to return `failure()` the method is expected
/// to clean up any inserted operations.
using YieldTiledValuesFn = std::function<LogicalResult(
    RewriterBase &rewriter, Location loc, ValueRange ivs, ValueRange newBbArgs,
    SmallVector<Value> &tiledValues,
    SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &resultSizes)>;

/// Generate the tile-loop nest using `scf.for` operation.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNestUsingForOp(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ValueRange destinationTensors,
    YieldTiledValuesFn yieldTiledValuesFn,
    SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<Value> ivs;

  for (auto [loopRange, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (isConstantIntValue(tileSize, 0))
      continue;

    Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
    Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
    Value step = getValueOrCreateConstantIndexOp(rewriter, loc, tileSize);
    auto loop =
        rewriter.create<scf::ForOp>(loc, lb, ub, step, destinationTensors,
                                    [](OpBuilder &bodyBuilder, Location bodyLoc,
                                       Value iv, ValueRange /*iterArgs*/) {});
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToEnd(loop.getBody());
    destinationTensors = loop.getRegionIterArgs();
  }

  SmallVector<Value> tiledResults;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  if (failed(yieldTiledValuesFn(rewriter, loc, ivs, destinationTensors,
                                tiledResults, resultOffsets, resultSizes))) {
    return rewriter.notifyMatchFailure(
        loc, "failed to generate inner tile loop body");
  }
  if (loops.empty())
    return success();

  // 6. Yield all the results of the tiled operation.
  SmallVector<Value> yieldedValues;
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    auto insertSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, tiledValue, destinationTensor, resultOffset, resultSize,
        resultStride);
    yieldedValues.push_back(insertSlice);
  }
  rewriter.create<scf::YieldOp>(loc, yieldedValues);

  // Add the scf.yield operations for all the outer loops.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(MutableArrayRef(loops).drop_back(),
                       MutableArrayRef(loops).drop_front())) {
    rewriter.setInsertionPointToEnd(
        cast<scf::ForOp>(outerLoop.getOperation()).getBody());
    rewriter.create<scf::YieldOp>(outerLoop.getLoc(), innerLoop->getResults());
  }
  return success();
}

/// Generate the tile-loop nest using `scf.forall` operation.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `mappingVector` is the mapping attributes to use for loop construction.
///   Can be empty.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNestUsingForallOp(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<Attribute> mappingVector,
    ValueRange destinationTensors, YieldTiledValuesFn tiledBodyFn,
    SmallVector<LoopLikeOpInterface> &loops) {
  SmallVector<OpFoldResult> lbs, ubs, steps;
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> offsets(loopRanges.size()),
      sizes(loopRanges.size());

  for (auto [tileSize, loopRange] : llvm::zip_equal(tileSizes, loopRanges)) {
    if (isConstantIntValue(tileSize, 0))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(tileSize);
  }
  assert(!lbs.empty() && "Expected at least one loop range");

  std::optional<ArrayAttr> mappingAttr;
  if (!mappingVector.empty())
    mappingAttr = rewriter.getArrayAttr(mappingVector);

  auto forallOp = rewriter.create<scf::ForallOp>(
      loc, lbs, ubs, steps, destinationTensors, mappingAttr);
  loops.push_back(forallOp);

  rewriter.setInsertionPoint(forallOp.getTerminator());
  destinationTensors = forallOp.getRegionOutArgs();

  SmallVector<Value> tiledResults;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  if (failed(tiledBodyFn(rewriter, loc, forallOp.getInductionVars(),
                         destinationTensors, tiledResults, resultOffsets,
                         resultSizes)))
    return rewriter.notifyMatchFailure(loc, "failed to generate loop body");

  rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));

    rewriter.create<tensor::ParallelInsertSliceOp>(
        loc, tiledValue, destinationTensor, resultOffset, resultSize,
        resultStride);
  }
  return success();
}

static LogicalResult generateLoopNest(RewriterBase &rewriter, Location loc,
                                      const scf::SCFTilingOptions &options,
                                      ArrayRef<Range> loopRanges,
                                      ArrayRef<OpFoldResult> tileSizes,
                                      ValueRange destinationTensors,
                                      YieldTiledValuesFn tiledBodyFn,
                                      SmallVector<LoopLikeOpInterface> &loops) {
  // If the tile sizes are all zero, no loops are generated. Just call the
  // callback function to handle untiled case.
  if (llvm::all_of(tileSizes, isZeroIndex)) {
    SmallVector<Value> tiledResults;
    SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
    return tiledBodyFn(rewriter, loc, ValueRange{}, destinationTensors,
                       tiledResults, resultOffsets, resultSizes);
  }
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForOp) {
    llvm::errs() << "SCF for Op\n";
    return generateLoopNestUsingForOp(rewriter, loc, loopRanges, tileSizes,
                                      destinationTensors, tiledBodyFn, loops);
  }
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForallOp) {
    llvm::errs() << "SCF for All Op\n";
    return generateLoopNestUsingForallOp(
        rewriter, loc, loopRanges, tileSizes, options.mappingVector,
        destinationTensors, tiledBodyFn, loops);
  }
  return rewriter.notifyMatchFailure(loc, "unhandled loop type");
}

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.
FailureOr<scf::SCFTilingResult>
myTileUsingSCF(RewriterBase &rewriter, TilingInterface op,
               const scf::SCFTilingOptions &options) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<OpFoldResult> tileSizes =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizes.size() < iterationDomain.size()) {
    auto zero = rewriter.getIndexAttr(0);
    tileSizes.append(numLoops - tileSizes.size(), zero);
  }

  // 3. If there is an interchange specified, permute the iteration domain and
  // the tile sizes.
  SmallVector<int64_t> interchangeVector;
  if (!options.interchangeVector.empty()) {
    interchangeVector = fillInterchangeVector(options.interchangeVector,
                                              iterationDomain.size());
  }
  if (!interchangeVector.empty()) {
    if (!isPermutationVector(interchangeVector)) {
      return rewriter.notifyMatchFailure(
          op, "invalid intechange vector, not a permutation of the entire "
              "iteration space");
    }

    applyPermutationToVector(iterationDomain, interchangeVector);
    applyPermutationToVector(tileSizes, interchangeVector);
  }

  FailureOr<TilingResult> tilingResult;
  // 4. Define the lambda function used later to generate the body of the
  // innermost tiled loop.
  YieldTiledValuesFn innerYieldTiledValuesFn =
      [&](RewriterBase &rewriter, Location loc, ValueRange ivs,
          ValueRange regionIterArgs, SmallVector<Value> &tiledResults,
          SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
          SmallVector<SmallVector<OpFoldResult>> &resultSizes)
      -> LogicalResult {
    // 4a. Compute the `offsets` and `sizes` to use for tiling.
    SmallVector<OpFoldResult> offsets, sizes;
    {
      int materializedLoopNum = 0;
      for (auto [tileSize, loopRange] :
           llvm::zip_equal(tileSizes, iterationDomain)) {
        if (isConstantIntValue(tileSize, 0)) {
          offsets.push_back(loopRange.offset);
          sizes.push_back(loopRange.size);
          continue;
        }
        Value iv = ivs[materializedLoopNum++];

        llvm::errs() << "iv dump: \n";
        iv.dump();
        llvm::errs() << "offsets iv: {";
        mlir::OpPrintingFlags printFlags;
        iv.printAsOperand(llvm::errs(), printFlags);
        llvm::errs() << " ";
        llvm::errs() << iv.getType();
        llvm::errs() << "} \n";

        offsets.push_back(iv);
        sizes.push_back(
            getBoundedTileSize(rewriter, loc, loopRange, iv, tileSize));
      }
    }
    llvm::errs() << "sizes: {\n";
    for (auto &sz : sizes) {
      llvm::errs() << "   ";
      sz.dump();
      // llvm::errs() << sz.dump()  "\n";
    }
    llvm::errs() << "}\n";
    llvm::errs() << "====================================================\n";

    // 4b. If interchange was provided, apply inverse of the interchange
    //     to get back the offsets/sizes in the order to be specified.
    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      applyPermutationToVector(offsets, inversePermutation);
      applyPermutationToVector(sizes, inversePermutation);
    }

    // 5. Generate the tiled implementation within the inner most loop.

    // 5a. Clone the operation within the loop body.
    auto clonedOp = cast<TilingInterface>(
        cloneOpAndUpdateDestinationArgs(rewriter, op, regionIterArgs));

    // 5b. Early return cloned op if tiling is not happening. We can not return
    // the original op because it could lead to
    // `rewriter.replaceOp(op, op->getResults())` and users would get crash.
    if (llvm::all_of(tileSizes, isZeroIndex)) {
      tiledResults.append(clonedOp->result_begin(), clonedOp->result_end());
      tilingResult =
          TilingResult{/*tiledOps=*/{clonedOp}, clonedOp->getResults()};
      return success();
    }

    // 5c. Tile the cloned operation.
    tilingResult = clonedOp.getTiledImplementation(rewriter, offsets, sizes);
    if (failed(tilingResult)) {
      rewriter.eraseOp(clonedOp);
      return op.emitOpError("faild to tile operation");
    }
    llvm::errs() << "tilingResult ops: {\n";
    for (auto tiledOp : tilingResult->tiledOps) {
      llvm::errs() << "   tile op:\n";
      tiledOp->dump();
    }

    llvm::errs() << "}\n";
    llvm::errs() << "tilingResult values: {\n";
    for (auto &tiledVal : tilingResult->tiledValues) {
      llvm::errs() << "   ";
      mlir::OpPrintingFlags printFlags;
      tiledVal.printAsOperand(llvm::errs(), printFlags);
      llvm::errs() << " ";
      llvm::errs() << tiledVal.getType();
      llvm::errs() << "\n";
    }
    llvm::errs() << "}\n";

    // 5d. Delete the cloned operation.
    rewriter.eraseOp(clonedOp);

    // 5e. Compute the offsets at which the result values are to be inserted
    //     back into its destinations.
    for (auto [index, tiledValue] :
         llvm::enumerate(tilingResult->tiledValues)) {
      tiledResults.push_back(tiledValue);
      SmallVector<OpFoldResult> resultOffset, resultSize;
      if (failed(op.getResultTilePosition(rewriter, index, offsets, sizes,
                                          resultOffset, resultSize))) {
        for (auto op : tilingResult->tiledOps) {
          rewriter.eraseOp(op);
        }
        return rewriter.notifyMatchFailure(
            op, "failed to get slice of result produced");
      }
      llvm::errs() << "resultOffset: {\n";
      for (auto &off : resultOffset) {
        llvm::errs() << "   ";
        off.dump();
      }
      llvm::errs() << "}\n\n";
      llvm::errs() << "resultSize: {\n";
      for (auto &sz : resultSize) {
        llvm::errs() << "   ";
        sz.dump();
      }
      llvm::errs() << "}\n\n";
      resultOffsets.emplace_back(std::move(resultOffset));
      resultSizes.emplace_back(std::move(resultSize));
    }

    return success();
  };

  // 6. Find the destination tensors to use for the operation.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(rewriter, op.getLoc(), op,
                                             destinationTensors))) {
    return rewriter.notifyMatchFailure(op,
                                       "unable to create destination tensors");
  }

  // 7. Generate the tiled loops nest using the callback defined above.
  SmallVector<LoopLikeOpInterface> loops;
  if (failed(generateLoopNest(rewriter, op.getLoc(), options, iterationDomain,
                              tileSizes, destinationTensors,
                              innerYieldTiledValuesFn, loops)))
    return op.emitOpError("failed to generate tiling loops");
  assert(succeeded(tilingResult) &&
         "expected tiling result to be computed after loop generation");

  // If loops are empty, the tiled op is used as the replacement for the untiled
  // op.
  if (loops.empty()) {
    return scf::SCFTilingResult{tilingResult->tiledOps, loops,
                                tilingResult->tiledValues};
  }

  SmallVector<Value> replacements = llvm::map_to_vector(
      loops.front()->getResults(), [](OpResult r) -> Value { return r; });
  return scf::SCFTilingResult{tilingResult->tiledOps, loops, replacements};
}

/// Implementation of tile consumer and fuse producer greedily.
FailureOr<scf::SCFTileAndFuseResult> myTileConsumerAndFuseProducersUsingSCF(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  SetVector<Operation *> fusedProducers, tiledAndFusedOps;
  llvm::SmallDenseMap<Value, size_t> origProducerToLoopResultNum;

  FailureOr<scf::SCFTilingResult> tilingResult =
      myTileUsingSCF(rewriter, consumer, options.tilingOptions);

  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
  for (auto *tiledOp : tilingResult->tiledOps)
    tiledAndFusedOps.insert(tiledOp);

  // If there are no loops generated, fusion is immaterial.
  auto &loops = tilingResult->loops;
  if (loops.empty()) {
    DenseMap<Value, Value> replacements;
    for (auto [origVal, replacement] :
         llvm::zip_equal(consumer->getResults(), tilingResult->replacements)) {
      replacements[origVal] = replacement;
    }
    return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                     replacements};
  }

  // To keep track of replacements for now just record the map from the original
  // untiled value to the result number of the for loop. Since the loop gets
  // potentially replaced during fusion, keeping the value directly wont work.
  DenseMap<Value, size_t> origValToResultNumber;
  for (auto [index, result] : llvm::enumerate(consumer->getResults())) {
    origValToResultNumber[result] = index;
  }

  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
  };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Find the original producer of the slice.
    auto [fusableProducer, destinationInitArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                          loops);
    if (!fusableProducer)
      continue;

    auto [fuseSlice, yieldReplacement] = options.fusionControlFn(
        candidateSliceOp, fusableProducer, destinationInitArg.has_value());
    if (!fuseSlice)
      continue;

    // The operands of the fused producer might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp,
                                              loops);
    if (!fusedResult)
      continue;

    if (yieldReplacement) {
      if (failed(yieldReplacementForFusedProducer(
              rewriter, candidateSliceOp, fusedResult.value(), loops))) {
        return rewriter.notifyMatchFailure(
            fusableProducer.getOwner(), "failed to replacement value for this "
                                        "oepration from within the tiled loop");
      }
      origValToResultNumber[fusableProducer] =
          loops.front()->getNumResults() - 1;
    }

    if (Operation *tiledAndFusedOp =
            fusedResult->tiledAndFusedProducer.getDefiningOp()) {
      fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
      tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, candidates);
    }
  }

  DenseMap<Value, Value> replacements;
  for (auto [origVal, resultNumber] : origValToResultNumber) {
    replacements[origVal] = loops.front()->getResult(resultNumber);
  }

  return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                   replacements};
}

// Return true if `op` can be tiled using `tileSizes`. Require to statically
// know the range and the tile factor. The tile must be full.
static bool canBeTiledWithCurrentSpec(Operation *op,
                                      ArrayRef<OpFoldResult> tileSizes,
                                      int64_t minTileFactor) {
  assert(isa<TilingInterface>(op) &&
         "expect an op implementing the tiling interface");
  assert(!tileSizes.empty() && "expect tile sizes to be non-empty");
  SmallVector<utils::IteratorType> loopIteratorTypes =
      cast<TilingInterface>(op).getLoopIteratorTypes();
  if (tileSizes.size() > loopIteratorTypes.size())
    return false;

  // Validate tiles:
  // - All zeros, nothing to do.
  // - Each tile must be statically known and perfectly divides the dimension.
  // - Require tiling on parallel dimensions only.
  if (llvm::all_of(tileSizes, [](OpFoldResult tile) {
        return isConstantIntValue(tile, 0);
      })) {
    return false;
  }

  llvm::errs() << "Running tile validations ----\n";
  if (!validateFullTilesOnDims(cast<TilingInterface>(op), tileSizes, /*dim=*/{},
                               minTileFactor)) {
    llvm::errs() << "FAILED\n";
    return false;
  }
  llvm::errs() << "OK\n";

  for (auto tileIdx : llvm::seq<size_t>(0, tileSizes.size())) {
    if (isConstantIntValue(tileSizes[tileIdx], 0))
      continue;
    if (!linalg::isParallelIterator(loopIteratorTypes[tileIdx]))
      return false;
  }

  // Candidate op is good to go.
  return true;
}

// Entry point for fusion with element-wise operations.
static FailureOr<scf::SCFTileAndFuseResult> fuseWithEltwise(
    RewriterBase &rewriter, TilingInterface consumer,
    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> &tileSizes,
    llvm::SmallDenseSet<Operation *> &alreadyFusedOps, int64_t maxDepth,
    int64_t minTileFactor) {
  // Step 0. Early exit if tileSizes are empty.
  if (tileSizes.empty() || !tileSizes.count(consumer)) {
    llvm::errs() << "EMPTY TILE SIZES\n";
    return failure();
  }

  // Step 1. If the consumer is already tiled and fused, bail out.
  if (alreadyFusedOps.count(consumer)) {
    llvm::errs() << "CONSUMER: " << consumer << "\nALREADY TILED AND FUSED\n";
    return failure();
  }

  // Step 2. Check if the tile configuration fits the consumer.
  if (!canBeTiledWithCurrentSpec(consumer, tileSizes.at(consumer),
                                 minTileFactor)) {
    llvm::errs() << "CONSUMER: " << consumer
                 << "\nCANNOT BE TILED WITH CURRENT CONFIG\n";
    return failure();
  }

  // Step 3. Collect the operations that can be tiled and fused.
  // llvm::SmallDenseSet<Operation *> worklist =
  //     collectFusableProducers(consumer, tileSizes, alreadyFusedOps,
  //     maxDepth);
  // llvm::errs() << "#WORKLIST: " << worklist.size() << "\n";
  // if (worklist.size() < 1)
  //   return failure();

  // Step 4. Tile the consumer and move the producers
  // in the fusion domain.
  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizes.at(consumer));
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(options);
  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand) {
        Operation *candidateOp = originalProducer.getOwner();
        if (!candidateOp || (alreadyFusedOps.count(candidateOp) &&
                             !isa<linalg::FillOp>(candidateOp))) {
          return std::make_tuple(false, false);
        }
        return std::make_tuple(true, false);
      };
  tileAndFuseOptions.setFusionControlFn(controlFn);
  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      myTileConsumerAndFuseProducersUsingSCF(rewriter, consumer,
                                             tileAndFuseOptions);
  if (failed(tileAndFuseResult)) {
    return rewriter.notifyMatchFailure(
        consumer, "failed to tile and fuse with op as root");
  }
  if (!tileAndFuseResult->loops.empty()) {
    tileAndFuseResult->loops[0].dump();
    // tileAndFuseResult->loops[0]->setAttr(
    //     linalgx::utils::kLoopParallel,
    //     rewriter.getStringAttr(linalgx::utils::kLoopRoot));
  }
  return *tileAndFuseResult;
}

//===----------------------------------------------------------------------===//
// tileUsingSCF implementation.
//===----------------------------------------------------------------------===//

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.

static int64_t getTileForDim(linalg::LinalgOp linalgOp, unsigned dim) {
  const int64_t tile = 32;
  SmallVector<int64_t, 4> loopsRange = linalgOp.getStaticLoopRanges();
  llvm::errs() << "[Loops]: ";
  llvm::interleaveComma(loopsRange, llvm::errs());
  llvm::errs() << "\n";
  if (loopsRange[dim] == ShapedType::kDynamic)
    return tile;
  if (loopsRange[dim] < tile || loopsRange[dim] % tile != 0)
    return 0;
  return tile;
}

static SmallVector<int64_t>
getInitialTileSizesForMatmulOp(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t> tiles(linalgOp.getNumLoops(), 0);

  mlir::SmallVector<std::pair<Value, unsigned>> operandDimPairs{};
  // expecting to treat it as d0
  // so expected result is in0 and out
  linalgOp.mapIterationSpaceDimToAllOperandDims(0, operandDimPairs);
  linalgOp.dump();
  llvm::errs() << "\n";
  for (auto &p : operandDimPairs) {
    llvm::errs() << "val: {";
    mlir::OpPrintingFlags printFlags;
    p.first.printAsOperand(llvm::errs(), printFlags);
    llvm::errs() << " ";
    llvm::errs() << p.first.getType();
    llvm::errs() << "} dim pos in shape: " << p.second << "\n";
  }
  if (isa<linalg::MatmulOp>(linalgOp)) {
    tiles[0] = getTileForDim(linalgOp, 0); // i loop
    tiles[1] = getTileForDim(linalgOp, 1); // j loop
    return tiles;
  }
  llvm::errs() << "No initial tile sizes.\n";
}

class TileLinalg : public mlir::gc::impl::TileLinalgNamedBase<TileLinalg> {

  void runOnOperation() override {
    // Step1. Tile and fuse pack consumer and producer.
    auto *ctx = &getContext();
    auto ops = getOperation().getOps<func::FuncOp>();
    IRRewriter rewriter(ctx);

    SmallVector<linalg::LinalgOp> linalgOperations;
    // Walk postorder to increase fusion boundaries.
    getOperation()->walk<WalkOrder::PostOrder>([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::MatmulOp>(linalgOp)) {
        llvm::errs() << "func op body  to tile op: " << linalgOp << "\n";
        linalgOperations.push_back(linalgOp);
      }
    });

    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> initialTiles;
    for (auto linalgOp : linalgOperations) {
      auto tiles = getInitialTileSizesForMatmulOp(linalgOp);
      linalgOp.dump();
      llvm::errs() << " [Tiles]: ";
      llvm::interleaveComma(tiles, llvm::errs());
      llvm::errs() << "\n";
      // llvm::errs() << "i loop " << getTileForDim(linalgOp, 0) << "\n";
      // llvm::errs() << "j loop " << getTileForDim(linalgOp, 1) << "\n";
      initialTiles[linalgOp] =
          getAsOpFoldResult(rewriter.getI64ArrayAttr(tiles));
    }

    for (Operation *linalgOp : linalgOperations) {
      // Set to keep track of fused ops.
      llvm::SmallDenseSet<Operation *> fusedOps;

      FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult =
          fuseWithEltwise(rewriter, cast<TilingInterface>(linalgOp),
                          initialTiles, fusedOps, 5, 2);
    }

    // scf::SCFTilingOptions tilingOptions;
  }
};

} // namespace
