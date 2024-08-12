//===-- TilingUtil.cpp - Implementation of linalg Tiling --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include <optional>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_LINALGTILINGPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-tiling"

namespace mlir {
namespace linalgX {

struct LinalgOpPartialReductionInterface {
  static FailureOr<SmallVector<Value>> generateInitialTensorForPartialReduction(
      Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
      ArrayRef<int> reductionDims, ArrayRef<int> newParallelDims) {
    auto linalgOp = cast<LinalgOp>(op);
    OpBuilder::InsertionGuard guard(b);

    if (newParallelDims.empty())
      newParallelDims = reductionDims;

    if (linalgOp.hasPureBufferSemantics())
      return op->emitOpError("expected operation to have tensor semantics");
    // Insert the new parallel dimension based on the index of the reduction
    // loops. This could be controlled by user for more flexibility.
    SmallVector<Value> inits;
    for (int initIdx = 0, e = linalgOp.getNumDpsInits(); initIdx < e;
         ++initIdx) {
      SmallVector<Operation *, 4> combinerOps;
      if (!matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps) ||
          combinerOps.size() != 1)
        return op->emitOpError("Failed to anaysis the reduction operation.");

      Operation *reductionOp = combinerOps[0];
      std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
      if (!identity.has_value())
        return op->emitOpError(
            "Failed to get an identity value for the reduction operation.");

      ArrayRef<int64_t> oldShape =
          linalgOp.getShape(linalgOp.getDpsInitOperand(0));

      // Extend tile size vector to the rank of the output tensor.
      SmallVector<Value> tileSizeVector =
          getValueOrCreateConstantIndexOp(b, loc, sizes);
      if (tileSizeVector.size() < oldShape.size()) {
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        tileSizeVector.append(oldShape.size() - tileSizeVector.size(), zero);
      }

      // Calculate the new shape, we insert the new dimensions based on the
      // index of the reduction dimensions.
      SmallVector<int64_t> newOutputShape;
      SmallVector<Value> dynamicDims;
      int64_t currReductionDims = 0;
      DenseSet<int> newParallelDimsSet(newParallelDims.begin(),
                                       newParallelDims.end());
      for (int64_t idx :
           llvm::seq<int64_t>(0, oldShape.size() + newParallelDims.size())) {
        if (newParallelDimsSet.contains(idx)) {
          dispatchIndexOpFoldResults(sizes[reductionDims[currReductionDims]],
                                     dynamicDims, newOutputShape);
          currReductionDims++;
          continue;
        }
        int64_t oldIdx = idx - currReductionDims;
        int64_t dim = oldShape[oldIdx];
        newOutputShape.push_back(dim);
        if (ShapedType::isDynamic(dim))
          dynamicDims.push_back(b.create<tensor::DimOp>(
              loc, linalgOp.getDpsInitOperand(0)->get(), oldIdx));
      }
      Value emptyTensor = b.create<tensor::EmptyOp>(
          loc, newOutputShape, linalgOp.getRegionOutputArgs()[0].getType(),
          dynamicDims);
      Value constantOp = b.create<arith::ConstantOp>(loc, *identity);
      auto identityTensor =
          b.create<linalg::FillOp>(loc, constantOp, emptyTensor);
      inits.push_back(identityTensor.getResult(0));
    }
    return inits;
  }

  static Operation *tileToPartialReduction(Operation *op, OpBuilder &b,
                                           Location loc, ValueRange init,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes,
                                           ArrayRef<int> reductionDims) {
    OpBuilder::InsertionGuard guard(b);
    auto linalgOp = cast<LinalgOp>(op);

    AffineMap oldOutputMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
    SmallVector<AffineExpr> outputExpr(oldOutputMap.getNumResults() +
                                       reductionDims.size());

    for (int idx : reductionDims)
      outputExpr[idx] = b.getAffineDimExpr(idx);
    int currExpr = 0;
    for (int idx : llvm::seq<int>(0, outputExpr.size())) {
      if (outputExpr[idx])
        continue;
      outputExpr[idx] = oldOutputMap.getResult(currExpr++);
    }

    // Step 1: Extract a slice of the input operands.
    SmallVector<Value> valuesToTile = linalgOp.getDpsInputs();
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

    // Step 2: Extract the accumulator operands
    SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
    SmallVector<OpFoldResult> outOffsets(offsets.size(), b.getIndexAttr(0));
    // TODO: use SubsetExtractOpInterface once it is available.
    Value out = b.create<tensor::ExtractSliceOp>(loc, init[0], outOffsets,
                                                 sizes, strides);

    // Step3. Create a generic op where the reduction dimensions are replaced
    // by a parallel dimension of the size of reduction.
    SmallVector<utils::IteratorType> newIteratorTypes =
        linalgOp.getIteratorTypesArray();
    for (int dim : reductionDims)
      newIteratorTypes[dim] = utils::IteratorType::parallel;
    SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
    newMaps.back() = AffineMap::get(newMaps.back().getNumDims(), 0, outputExpr,
                                    linalgOp.getContext());
    auto genericOp =
        b.create<GenericOp>(loc, TypeRange({out.getType()}), tiledOperands,
                            ValueRange({out}), newMaps, newIteratorTypes);
    IRMapping mapping;
    op->getRegion(0).cloneInto(&genericOp.getRegion(),
                               genericOp.getRegion().begin(), mapping);
    return genericOp.getOperation();
  }

  static Operation *mergeReductions(Operation *op, OpBuilder &b, Location loc,
                                    ValueRange partialReduce,
                                    ArrayRef<int> reductionDims) {
    auto linalgOp = cast<LinalgOp>(op);
    SmallVector<int64_t> reductionDimsInt64(reductionDims.begin(),
                                            reductionDims.end());
    SmallVector<Operation *, 4> combinerOps;
    matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps);
    Operation *reductionOp = combinerOps[0];

    auto reduction = b.create<linalg::ReduceOp>(
        loc, ValueRange({partialReduce[0]}),
        ValueRange({linalgOp.getDpsInits()[0]}), reductionDimsInt64,
        [reductionOp](OpBuilder &b, Location loc, ValueRange inputs) {
          Operation *clonedReductionOp = b.clone(*reductionOp);
          clonedReductionOp->setOperand(0, inputs[0]);
          clonedReductionOp->setOperand(1, inputs[1]);
          b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
        });
    return reduction.getOperation();
  }
};

std::tuple<SmallVector<Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(RewriterBase &b, Location loc, AffineMap map,
                    ArrayRef<OpFoldResult> allShapeSizes,
                    ArrayRef<OpFoldResult> allTileSizes) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get shape sizes in loop order.
  SmallVector<OpFoldResult> shapeSizes =
      makeComposedFoldedMultiResultAffineApply(b, loc, map, allShapeSizes);
  SmallVector<OpFoldResult> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  for (int idx = 0, e = tileSizes.size(), zerosCount = 0; idx < e; ++idx) {
    if (getConstantIntValue(tileSizes[idx - zerosCount]) ==
        static_cast<int64_t>(0)) {
      shapeSizes.erase(shapeSizes.begin() + idx - zerosCount);
      tileSizes.erase(tileSizes.begin() + idx - zerosCount);
      ++zerosCount;
      continue;
    }
    loopIndexToRangeIndex[idx] = idx - zerosCount;
  }

  // Create a new range with the applied tile sizes.
  SmallVector<Range, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx)
    res.push_back(Range{b.getIndexAttr(0), shapeSizes[idx], tileSizes[idx]});
  return std::make_tuple(res, loopIndexToRangeIndex);
}

void transformIndexOps(RewriterBase &b, LinalgOp op,
                       SmallVectorImpl<Value> &ivs,
                       const LoopIndexToRangeIndexMap &loopIndexToRangeIndex) {
  SmallVector<Value> allIvs(op.getNumLoops(), nullptr);
  for (auto en : enumerate(allIvs)) {
    auto rangeIndex = loopIndexToRangeIndex.find(en.index());
    if (rangeIndex == loopIndexToRangeIndex.end())
      continue;
    en.value() = ivs[rangeIndex->second];
  }
  offsetIndices(b, op, getAsOpFoldResult(allIvs));
}

/// Returns true if the maximum tile offset `tileSize * numThreads-1` is less
/// than `iterationSize`.
static bool canOmitTileOffsetInBoundsCheck(OpFoldResult tileSize,
                                           OpFoldResult numThreads,
                                           OpFoldResult iterationSize) {
  std::optional<int64_t> tileSizeConst = getConstantIntValue(tileSize);
  std::optional<int64_t> numThreadsConst = getConstantIntValue(numThreads);
  std::optional<int64_t> iterSizeConst = getConstantIntValue(iterationSize);
  if (!tileSizeConst || !numThreadsConst || !iterSizeConst)
    return false;
  return *tileSizeConst * (*numThreadsConst - 1) < *iterSizeConst;
}

/// Build an `affine_max` of all the `vals`.
static OpFoldResult buildMax(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return affine::makeComposedFoldedAffineMax(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Build an `affine_min` of all the `vals`.
static OpFoldResult buildMin(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return affine::makeComposedFoldedAffineMin(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Fill out the `tiledOffsets` and `tiledSizes` to be used to tile to a given
/// number of threads.
static void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, scf::ForallOp forallOp,
    ArrayRef<OpFoldResult> numThreads, SmallVector<Range> loopRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(forallOp.getBody(0));

  SmallVector<Value> threadIds = forallOp.getInductionVars();
  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  int64_t nLoops = loopRanges.size();
  tiledOffsets.reserve(nLoops);
  tiledSizes.reserve(nLoops);
  for (unsigned loopIdx = 0, threadIdIdx = 0; loopIdx < nLoops; ++loopIdx) {
    bool overflow = loopIdx >= numThreads.size();
    bool isZero = !overflow && isConstantIntValue(numThreads[loopIdx], 0);
    // Degenerate case: take the whole domain.
    if (overflow || isZero) {
      tiledOffsets.push_back(loopRanges[loopIdx].offset);
      tiledSizes.push_back(loopRanges[loopIdx].size);
      continue;
    }

    // Tiled case: compute the offset and size.
    AffineExpr i, j, m, n, o;
    bindDims(b.getContext(), i, j);
    bindSymbols(b.getContext(), m, n, o);
    OpFoldResult size = loopRanges[loopIdx].size;
    OpFoldResult offset = loopRanges[loopIdx].offset;
    OpFoldResult threadId = threadIds[threadIdIdx];
    // Symbolic fixed max size per thread.
    // TODO: floor + 0/1 depending on case for better load-balancing.
    OpFoldResult tileSizePerThread =
        nominalTileSizes.has_value()
            ? (*nominalTileSizes)[loopIdx]
            : makeComposedFoldedAffineApply(
                  b, loc, m.ceilDiv(n),
                  ArrayRef<OpFoldResult>{size, nonZeroNumThreads[threadIdIdx]});
    // Dynamic offset shifted by threadId * maxSizePerThread.
    OpFoldResult offsetPerThread = makeComposedFoldedAffineApply(
        b, loc, i + j * m, {offset, threadId, tileSizePerThread});
    // Dynamic upper-bound depending on the threadId.
    OpFoldResult residualTileSize = makeComposedFoldedAffineApply(
        b, loc, i + j * m - n,
        {offset, nonZeroNumThreads[threadIdIdx], tileSizePerThread, size});
    if (!isConstantIntValue(residualTileSize, 0)) {
      OpFoldResult sizeMinusOffsetPerThread = makeComposedFoldedAffineApply(
          b, loc, -i + m, {offsetPerThread, size});
      tileSizePerThread =
          buildMin(b, loc, {sizeMinusOffsetPerThread, tileSizePerThread});
    }

    tiledOffsets.push_back(offsetPerThread);
    // TODO: if tileSizePerThread <= 0 early exit.
    if (!omitTileOffsetBoundsCheck &&
        !canOmitTileOffsetInBoundsCheck(tileSizePerThread,
                                        nonZeroNumThreads[threadIdIdx], size))
      tileSizePerThread =
          buildMax(b, loc, {b.getIndexAttr(0), tileSizePerThread});

    tiledSizes.push_back(tileSizePerThread);
    ++threadIdIdx;
  }
}

template <typename LoopTy>
static FailureOr<TiledLinalgOp>
tileLinalgOpImpl(RewriterBase &b, LinalgOp op, ArrayRef<OpFoldResult> tileSizes,
                 const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);

  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      })) {
    TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  // 1. Build the tiled loop ranges.
  SmallVector<OpFoldResult> allShapeSizes =
      op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();

  auto [loopRanges, loopIndexToRangeIndex] = makeTiledLoopRanges(
      b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);

  SmallVector<utils::IteratorType, 4> iteratorTypes;
  for (const auto &attr : enumerate(op.getIteratorTypesArray())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(attr.value());
  }
  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(tileSizes.size(), b.getContext());
  if (!options.interchangeVector.empty()) {
    // Based on the pruned iterations (due to zero tile size), recompute the
    // interchange vector.
    SmallVector<unsigned, 4> interchangeVector;
    interchangeVector.reserve(options.interchangeVector.size());
    for (auto pos : options.interchangeVector) {
      auto it = loopIndexToRangeIndex.find(pos);
      if (it == loopIndexToRangeIndex.end())
        continue;
      interchangeVector.push_back(it->second);
    }
    // Interchange vector is guaranteed to be a permutation,
    // `inversePermutation` must succeed.
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(interchangeVector, b.getContext()));
    assert(invPermutationMap);
    SmallVector<int64_t> permutation(interchangeVector.begin(),
                                     interchangeVector.end());
    applyPermutationToVector(loopRanges, permutation);
    applyPermutationToVector(iteratorTypes, permutation);
  }

  // Handle distribution. Create a vector of the same size of loops that are to
  // be tiled.
  SmallVector<linalg::ProcInfo> procInfo;
  if (options.distribution) {
    procInfo.resize(
        iteratorTypes.size(),
        linalg::ProcInfo{nullptr, nullptr, linalg::DistributionMethod::None});
    // Collect loop ranges of tiled loops, loops that are parallel.
    SmallVector<Range> parallelLoopRanges;
    for (const auto &iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value()))
        break;
      parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);
    }
    auto returnedProcInfo =
        options.distribution->procInfo(b, op.getLoc(), parallelLoopRanges);
    unsigned procIdIdx = 0;
    // Update the distribution information for the loops.
    for (const auto &iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value()))
        break;
      procInfo[iteratorType.index()] = returnedProcInfo[procIdIdx++];
    }
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder &builder, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;
    if (!options.interchangeVector.empty()) {
      for (AffineExpr result : invPermutationMap.getResults())
        interchangedIvs.push_back(
            ivs[cast<AffineDimExpr>(result).getPosition()]);
    } else {
      interchangedIvs.assign(ivs.begin(), ivs.end());
    }

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(op->getNumOperands()) &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    SmallVector<OpFoldResult> sizeBounds =
        makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                                 allShapeSizes);
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, loc, op, valuesToTile, getAsOpFoldResult(interchangedIvs), tileSizes,
        sizeBounds,
        /*omitPartialTileCheck=*/false);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(op, tiledOperands);
    res = clone(b, op, resultTensorTypes, tiledOperands);
    tensorResults =
        insertSlicesBack(builder, loc, op, tiledOperands, res->getResults());
    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  GenerateLoopNest<LoopTy>::doit(b, op.getLoc(), loopRanges, op, iteratorTypes,
                                 tiledLoopBodyBuilder, procInfo);

  // 3. Transform IndexOp results w.r.t. the tiling.
  linalg::transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (isa<BlockArgument>(iv)) {
      loops.push_back(cast<BlockArgument>(iv).getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      // TODO: Instead of doing this, try to recover the ops used instead of the
      // loop.
      loops.push_back(nullptr);
    }
  }

  // 5. Get the tensor results from the outermost loop if available. Otherwise
  // use the previously captured `tensorResults`.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  return TiledLinalgOp{
      res, loops, outermostLoop ? outermostLoop->getResults() : tensorResults};
}

FailureOr<linalg::ForallReductionTilingResult> tileReductionUsingForall(
    RewriterBase &b, PartialReductionOpInterface op,
    ArrayRef<OpFoldResult> threadNums, ArrayRef<OpFoldResult> tileSizes,
    ArrayRef<OpFoldResult> newParallelDims, std::optional<ArrayAttr> mapping) {
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(b);

  // Ops implementing PartialReductionOpInterface are expected to implement
  // TilingInterface.
  // TODO: proper core mechanism to tie interfaces together.
  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());

  // Ops implementing PartialReductionOpInterface are not necessarily expected
  // to implement TilingInterface.. This cast is unsafe atm.
  // TODO: proper core mechanism to tie interfaces together.
  // TODO: this function requires a pair of interfaces ..
  auto destinationStyleOp =
      dyn_cast<DestinationStyleOpInterface>(op.getOperation());
  if (!destinationStyleOp)
    return b.notifyMatchFailure(op, "not a destination style op");

  // Actually this only work for Linalg ops atm.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
  if (!linalgOp)
    return b.notifyMatchFailure(op, "not a linalg op");

  SmallVector<Range> iterationDomain = tilingInterfaceOp.getIterationDomain(b);
  if (op->getNumResults() != 1)
    return b.notifyMatchFailure(
        op, "don't support ops with multiple results for now");

  SmallVector<utils::IteratorType> iterators =
      tilingInterfaceOp.getLoopIteratorTypes();
  SmallVector<int> redDims;
  for (auto [idx, iteratorType] :
       llvm::enumerate(tilingInterfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::reduction)
      redDims.push_back(idx);
  }

  SmallVector<OpFoldResult> numThreads(threadNums.begin(), threadNums.end());
  if (numThreads.empty()) {
    SmallVector<Range> loopRanges = tilingInterfaceOp.getIterationDomain(b);
    unsigned nLoops = loopRanges.size();
    numThreads.reserve(nLoops);
    AffineExpr s0, s1;
    bindSymbols(b.getContext(), s0, s1);
    AffineExpr divExpr = s0.ceilDiv(s1);
    for (const auto &it : llvm::zip(tileSizes, loopRanges)) {
      OpFoldResult numTiles = std::get<0>(it);
      if (!isConstantIntValue(numTiles, 0))
        numTiles = makeComposedFoldedAffineApply(
            b, op.getLoc(), divExpr, {std::get<1>(it).size, std::get<0>(it)});
      numThreads.push_back(numTiles);
    }
  }

  if (!tileSizes.empty() && tileSizes.size() != numThreads.size())
    return b.notifyMatchFailure(op, "if tile sizes are present it must have as "
                                    "many elements as number of threads");

  if ((unsigned)redDims.front() >= numThreads.size())
    return b.notifyMatchFailure(
        op, "reduction dimension must be mapped to threads");
  SmallVector<int> constantNewParallelDims;
  for (auto dim : newParallelDims) {
    if (getConstantIntValue(dim) == std::nullopt)
      return b.notifyMatchFailure(
          op, "Expected new parallel dims to be constant integers.");
    constantNewParallelDims.push_back(*getConstantIntValue(dim));
  }
  if (newParallelDims.empty())
    constantNewParallelDims = redDims;
  if (constantNewParallelDims.size() != redDims.size())
    return b.notifyMatchFailure(
        op, "reduction dimension must be mapped to new parallel dims");
  // 1. Create the inital tensor value.
  FailureOr<SmallVector<Value>> maybeInitTensors =
      LinalgOpPartialReductionInterface::
          generateInitialTensorForPartialReduction(
              op, b, loc, numThreads, redDims, constantNewParallelDims);
  if (failed(maybeInitTensors))
    return b.notifyMatchFailure(
        op, "Failed to create inital tensors for partial reduction");
  SmallVector<Value> &initTensors = maybeInitTensors.value();

  // Gather destination tensors.
  SmallVector<Value> dest;
  if (failed(tensor::getOrCreateDestinations(b, loc, op, dest)))
    return b.notifyMatchFailure(op, "failed to get destination tensors");

  Operation *tiledOp = nullptr;
  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  SmallVector<Value> materializedNonZeroNumThreads =
      getValueOrCreateConstantIndexOp(b, loc, nonZeroNumThreads);
  // 2. Create the ForallOp with an empty region.
  scf::ForallOp forallOp = b.create<scf::ForallOp>(
      loc, getAsOpFoldResult(materializedNonZeroNumThreads), initTensors,
      mapping);
  // 3. Calculate the tile offsets and sizes for the subsequent loop that will
  // be nested under `forallOp`.
  SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
  std::optional<ArrayRef<OpFoldResult>> nominalTileSizes = std::nullopt;
  if (!tileSizes.empty() && threadNums.empty()) {
    nominalTileSizes = tileSizes;
  }
  calculateTileOffsetsAndSizes(b, loc, forallOp, numThreads, iterationDomain,
                               /*omitTileOffsetBoundsCheck =*/false,
                               /*nominalTileSizes=*/nominalTileSizes,
                               tiledOffsets, tiledSizes);
  // 4. Clone the tileable op and update its destination operands to use the
  // output bbArgs of the ForallOp.
  SmallVector<Value> tilingResults;
  ArrayRef<BlockArgument> destBbArgs = forallOp.getRegionIterArgs();
  {
    // 4.a. RAII guard, inserting within forallOp, before terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(forallOp.getTerminator());

    SmallVector<Value> tiledDpsInitOperands;
    for (Value initOperand : destinationStyleOp.getDpsInits()) {
      auto *it = llvm::find(dest, initOperand);
      assert(it != dest.end() && "dest operand not found in dest");
      unsigned destNum = std::distance(dest.begin(), it);
      auto dest = destBbArgs[destNum];
      auto destShape = cast<RankedTensorType>(dest.getType()).getShape();
      SmallVector<OpFoldResult> strides(destShape.size(), b.getIndexAttr(1));
      SmallVector<OpFoldResult> outOffsets(destShape.size(), b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes(destShape.size(), b.getIndexAttr(0));
      for (const auto &iteratorType :
           llvm::enumerate(cast<RankedTensorType>(dest.getType()).getShape())) {
        sizes[iteratorType.index()] =
            getAsIndexOpFoldResult(b.getContext(), iteratorType.value());
        if (llvm::find(constantNewParallelDims, iteratorType.index()) !=
            constantNewParallelDims.end()) {
          sizes[iteratorType.index()] = b.getIndexAttr(1);
        }
      }

      auto nonZeroDimIdx = 0;
      auto currentReductionIdx = 0;
      for (const auto &iteratorType : llvm::enumerate(numThreads)) {
        if (!isConstantIntValue(iteratorType.value(), 0)) {
          if (llvm::find(redDims, iteratorType.index()) != redDims.end()) {
            outOffsets[constantNewParallelDims[currentReductionIdx++]] =
                forallOp.getInductionVars()[nonZeroDimIdx];
          }
          nonZeroDimIdx++;
        }
      }
      // TODO: use SubsetExtractOpInterface once it is available.
      tiledDpsInitOperands.push_back(b.create<tensor::ExtractSliceOp>(
          loc, cast<RankedTensorType>(initOperand.getType()), dest, outOffsets,
          sizes, strides));
    }

    // 4.b. Clone the op and update init operands.
    // We cannot use a IRMapping here because it can replace
    // different OpOperands with the same value.
    Operation *clonedOp = b.clone(*op.getOperation());
    b.modifyOpInPlace(clonedOp, [&]() {
      for (auto [initOperandPtr, tiledInitValue] : llvm::zip_equal(
               cast<DestinationStyleOpInterface>(clonedOp).getDpsInitsMutable(),
               tiledDpsInitOperands)) {
        initOperandPtr.set(tiledInitValue);
      }
    });
    // 5. Tile the cloned op and delete the clone.
    if (tileSizes.empty() || threadNums.empty()) {
      FailureOr<TilingResult> tilingResult =
          cast<TilingInterface>(clonedOp).getTiledImplementation(
              b, tiledOffsets, tiledSizes);
      if (failed(tilingResult))
        return clonedOp->emitError("Failed to tile op: ");
      if (tilingResult->tiledOps.size() != 1) {
        return clonedOp->emitError("expected a single produced tiled op, got ")
               << tilingResult->tiledOps.size();
      }
      tiledOp = tilingResult->tiledOps.front();
      tilingResults = tilingResult->tiledValues;
    } else {
      LinalgTilingOptions options;
      FailureOr<TiledLinalgOp> maybeTiled = tileLinalgOpImpl<scf::ForOp>(
          b, cast<LinalgOp>(clonedOp), tileSizes, options);
      if (failed(maybeTiled))
        return b.notifyMatchFailure(op, "failed tileLinalgOpImpl");

      SmallVector<Value> ids = forallOp.getInductionVars();
      mapLoopToProcessorIds(cast<scf::ForOp>(maybeTiled->loops.back()), ids,
                            materializedNonZeroNumThreads);
      if (maybeTiled->loops.size() != 1) {
        return clonedOp->emitError("expected a single produced loop");
      }
      tiledOp = maybeTiled->op;
      tilingResults = maybeTiled->loops.front()->getResults();
    }

    b.eraseOp(clonedOp);
  }

  // 6. Insert the partial reductions back into a new tensor.
  for (auto [index, result, bbArg] : llvm::zip(
           llvm::seq<unsigned>(0, dest.size()), tilingResults, destBbArgs)) {
    // 6.a. Partial subset information is inserted just before the terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(forallOp.getTerminator());

    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(tilingInterfaceOp.getResultTilePosition(
            b, index, tiledOffsets, tiledSizes, resultOffsets, resultSizes)))
      return op->emitOpError("output offsets couldn't be calculated");
    SmallVector<OpFoldResult> resultOffsetsRank, resultSizesRank;
    uint64_t offIdx = 0;
    int64_t nonZeroDimIdx = 0;
    SmallVector<Value> reductionInductionVars;
    for (auto i = 0UL; i < numThreads.size(); ++i) {
      if (llvm::find(constantNewParallelDims, i) !=
          constantNewParallelDims.end()) {
        resultOffsetsRank.push_back(b.getIndexAttr(1));
        resultSizesRank.push_back(b.getIndexAttr(1));
      } else if (offIdx < resultOffsets.size()) {
        resultOffsetsRank.push_back(resultOffsets[offIdx]);
        resultSizesRank.push_back(resultSizes[offIdx++]);
      }
      if (llvm::find(redDims, i) != redDims.end()) {
        reductionInductionVars.push_back(
            forallOp.getInductionVars()[nonZeroDimIdx]);
      }
      if (!isConstantIntValue(numThreads[i], 0)) {
        nonZeroDimIdx++;
      }
    }
    for (auto [parallelDims, redVar] :
         llvm::zip(constantNewParallelDims, reductionInductionVars)) {
      resultOffsetsRank[parallelDims] = redVar;
      resultSizesRank[parallelDims] = b.getIndexAttr(1);
    }
    SmallVector<OpFoldResult> strides(resultSizesRank.size(),
                                      b.getIndexAttr(1));

    // 6.b. Parallel insertions are inserted at the end of the combining
    // terminator.
    b.setInsertionPointToEnd(forallOp.getTerminator().getBody());
    b.create<tensor::ParallelInsertSliceOp>(
        loc, result, bbArg, resultOffsetsRank, resultSizesRank, strides);
  }
  // 7. Merge the partial reductions.
  Operation *mergeOp = nullptr;
  b.setInsertionPointAfter(forallOp);
  mergeOp = linalgX::LinalgOpPartialReductionInterface::mergeReductions(
      op, b, loc, forallOp->getResults(), constantNewParallelDims);
  b.replaceOp(op, mergeOp->getResults());
  // 8. Return.
  ForallReductionTilingResult results;
  results.initialValues = initTensors;
  results.loops = forallOp;
  results.parallelTiledOps = SmallVector<Operation *>{tiledOp};
  results.mergeOps = SmallVector<Operation *>{mergeOp};
  return results;
}

} // namespace linalgX
} // namespace mlir