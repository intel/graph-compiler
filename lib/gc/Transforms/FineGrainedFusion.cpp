//===-- FineGrainedFusion.cpp - Fine-Grained Fusion -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include <llvm/Support/Debug.h>
#include <memory>

#include "TilingUsingInterfaceX.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_FINEGRAINEDFUSION
#include "gc/Transforms/Passes.h.inc"

static FailureOr<tensor::ExtractSliceOp>
getClosestExtractSliceOfOperand(OpOperand &operand) {
  if (auto iterArg = dyn_cast<BlockArgument>(operand.get())) {
    if (auto loop =
            dyn_cast<LoopLikeOpInterface>(iterArg.getOwner()->getParentOp())) {
      return getClosestExtractSliceOfOperand(*loop.getTiedLoopInit(iterArg));
    }
  }

  Operation *defineOp = operand.get().getDefiningOp();
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defineOp)) {
    return sliceOp;
  } else if (isa<linalg::FillOp, tensor::ExpandShapeOp,
                 tensor::CollapseShapeOp>(defineOp)) {
    // For downstream cases
    return getClosestExtractSliceOfOperand(defineOp->getOpOperand(0));
  } else {
    return failure();
  }
}

static FailureOr<OffsetSizeAndStrideOpInterface>
getClosestInsertSliceOfResult(OpResult result) {
  OffsetSizeAndStrideOpInterface sliceOp;
  for (auto &useOfResult : result.getUses()) {
    if (isa<tensor::InsertSliceOp>(useOfResult.getOwner()) ||
        isa<tensor::ParallelInsertSliceOp>(useOfResult.getOwner())) {
      if (llvm::detail::isPresent(sliceOp))
        return failure();
      sliceOp =
          dyn_cast<OffsetSizeAndStrideOpInterface>(useOfResult.getOwner());
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(useOfResult.getOwner())) {
      if (auto loop = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp())) {
        return getClosestInsertSliceOfResult(
            loop->getResult(useOfResult.getOperandNumber()));
      }
    }
  }

  if (!llvm::detail::isPresent(sliceOp))
    return failure();
  else {
    return sliceOp;
  }
}

struct CandidateDefOrUse {
  enum Type { def = 0, use };
  Operation *ownerOp;
  Type type;
  union {
    OpOperand *operand;
    OpResult result;
  };

  CandidateDefOrUse(OpResult resultOfDefOp)
      : ownerOp(resultOfDefOp.getDefiningOp()), type(Type::def),
        result(resultOfDefOp) {}
  CandidateDefOrUse(OpOperand *operandOfUseOp)
      : ownerOp(operandOfUseOp->getOwner()), type(Type::use),
        operand(operandOfUseOp) {}

  bool isDef() const { return type == Type::def; }
  bool isUse() const { return type == Type::use; }
};

using CandidateSliceFilter = std::function<LogicalResult(
    RewriterBase &, OffsetSizeAndStrideOpInterface, CandidateDefOrUse)>;

using CandidateSliceComparer =
    std::function<int(RewriterBase &, OffsetSizeAndStrideOpInterface,
                      OffsetSizeAndStrideOpInterface, CandidateDefOrUse)>;

static LogicalResult
noTilingOnReductionFilter(RewriterBase &rewriter,
                          OffsetSizeAndStrideOpInterface candidate,
                          CandidateDefOrUse defOrUse) {
  linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(defOrUse.ownerOp);
  if (!linalgOp)
    return success();

  AffineMap affMap =
      defOrUse.isDef() ? linalgOp.getIndexingMapMatchingResult(defOrUse.result)
                       : linalgOp.getMatchingIndexingMap(defOrUse.operand);

  TilingInterface tilableOp = dyn_cast<TilingInterface>(defOrUse.ownerOp);
  SmallVector<Range> iterDomain = tilableOp.getIterationDomain(rewriter);
  SmallVector<utils::IteratorType> iterTypes = tilableOp.getLoopIteratorTypes();
  SmallVector<OpFoldResult> tileSizes = candidate.getMixedSizes();
  // check reduction iteration is full on TileSizes
  for (const auto &resultExpr : llvm::enumerate(affMap.getResults())) {
    unsigned iterPosition =
        cast<AffineDimExpr>(resultExpr.value()).getPosition();
    if (iterTypes[iterPosition] == utils::IteratorType::reduction) {
      std::optional<int64_t> cstIterDomain =
          getConstantIntValue(iterDomain[iterPosition].size);
      FailureOr<int64_t> cstTileSizes =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, tileSizes[resultExpr.index()], nullptr,
              true);
      if (!cstIterDomain || failed(cstTileSizes) ||
          cstIterDomain != cstTileSizes) {
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult
exactTilingOnPackUnPackFilter(RewriterBase &rewriter,
                              OffsetSizeAndStrideOpInterface candidate,
                              CandidateDefOrUse defOrUse) {
  if (!isa<tensor::PackOp, tensor::UnPackOp>(defOrUse.ownerOp))
    return success();

  SmallVector<OpFoldResult> tileSizes = candidate.getMixedSizes();
  // collect target TileSizes and InnerTileSize to compare
  SmallVector<OpFoldResult> targetTileSizes, targetInnerTileSizes;
  if (auto packOp = dyn_cast<tensor::PackOp>(defOrUse.ownerOp)) {
    // tileSize comes from OpResult
    if (defOrUse.isDef()) {
      targetInnerTileSizes = packOp.getInnerTiles();
      targetTileSizes = llvm::to_vector(
          ArrayRef(tileSizes).take_back(targetInnerTileSizes.size()));
    } else {
      // tileSize comes from OpOperand
      targetTileSizes = llvm::to_vector(tileSizes);
      DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
          packOp.getDimAndTileMapping();
      targetInnerTileSizes.resize(dimAndTileMapping.size());
      for (const auto &dimAndTile : dimAndTileMapping) {
        targetInnerTileSizes[dimAndTile.first] = dimAndTile.second;
      }
    }
  } else if (auto unPackOp = dyn_cast<tensor::UnPackOp>(defOrUse.ownerOp)) {
    // tileSize comes from OpResult
    if (defOrUse.isDef()) {
      targetTileSizes = llvm::to_vector(tileSizes);
      DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
          unPackOp.getDimAndTileMapping();
      targetInnerTileSizes.resize(dimAndTileMapping.size());
      for (const auto &dimAndTile : dimAndTileMapping) {
        targetInnerTileSizes[dimAndTile.first] = dimAndTile.second;
      }
    } else {
      // tileSize comes from OpOperand
      targetInnerTileSizes = unPackOp.getInnerTiles();
      targetTileSizes = llvm::to_vector(
          ArrayRef(tileSizes).take_back(targetInnerTileSizes.size()));
    }
  }

  // check tileSizes is full on or multiple of `inner_tile_size`
  for (auto [tile, innerTile] :
       llvm::zip_equal(targetTileSizes, targetInnerTileSizes)) {
    if (isEqualConstantIntOrValue(tile, innerTile))
      continue;
    FailureOr<int64_t> cstSize = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, tile,
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    std::optional<int64_t> cstInnerSize = getConstantIntValue(innerTile);
    if (!failed(cstSize) && cstInnerSize) {
      if (*cstSize % *cstInnerSize == 0)
        continue;
    }
    return failure();
  }
  return success();
}

static LogicalResult
alreadyTiledOpFilter(RewriterBase &rewriter,
                     OffsetSizeAndStrideOpInterface candidate,
                     CandidateDefOrUse defOrUse) {
  // In general tiledOp would not have uses any more.
  return failure(defOrUse.ownerOp->use_empty());
}

static LogicalResult
SingleCandidateInBlockFilter(RewriterBase &rewriter,
                             OffsetSizeAndStrideOpInterface candidate,
                             CandidateDefOrUse defOrUse) {
  Block *parent = candidate->getBlock();

  // a. traverse all ops contained in parent Block.
  for (auto &opInBlock : parent->getOperations()) {
    // b. skip candidate slice
    if (&opInBlock == candidate.getOperation())
      continue;
    // c. check if all the other sliceOp not defined or used by the same owner
    // with candidate slice.
    if (auto otherCandidate =
            dyn_cast<OffsetSizeAndStrideOpInterface>(&opInBlock)) {
      if (defOrUse.isDef()) {
        SmallVector<tensor::ExtractSliceOp> backwardSlice;
        FailureOr<OpResult> realProducer =
            scfX::getRealProducerOfExtractSliceOp(otherCandidate,
                                                  backwardSlice);
        if (succeeded(realProducer) &&
            realProducer->getDefiningOp() == defOrUse.ownerOp) {
          return failure();
        }
      } else {
        SmallVector<OffsetSizeAndStrideOpInterface> forwardSlice;
        FailureOr<SmallVector<OpOperand *>> realConsumers =
            scfX::getRealConsumersFromInsertSliceOp(otherCandidate,
                                                    forwardSlice);
        if (succeeded(realConsumers) &&
            llvm::any_of(*realConsumers, [&defOrUse](OpOperand *use) {
              return use->getOwner() == defOrUse.ownerOp;
            })) {
          return failure();
        }
      }
    }
  }
  return success();
}

template <typename T1, typename T2> struct CandidateSliceProcessPipeLine {
  SmallVector<T1> candidateProcessFn;
  CandidateSliceProcessPipeLine() {
    append(static_cast<T2 *>(this)->getDefaultPipeLine());
  }
  CandidateSliceProcessPipeLine(const T1 &newFn)
      : CandidateSliceProcessPipeLine() {
    append(newFn);
  }
  CandidateSliceProcessPipeLine(ArrayRef<T1> newFns)
      : CandidateSliceProcessPipeLine() {
    append(newFns);
  }

  void append(const T1 &newFn) { candidateProcessFn.push_back(newFn); }
  void append(ArrayRef<T1> newFns) {
    llvm::append_range(candidateProcessFn, newFns);
  }

  SmallVector<T1> getDefaultPipeLine() { return {}; }
};

struct CandidateSliceFilterPipeLine
    : public CandidateSliceProcessPipeLine<CandidateSliceFilter,
                                           CandidateSliceFilterPipeLine> {
  CandidateSliceFilterPipeLine() : CandidateSliceProcessPipeLine() {}
  CandidateSliceFilterPipeLine(const CandidateSliceFilter &filter)
      : CandidateSliceProcessPipeLine(filter) {}
  CandidateSliceFilterPipeLine(const SmallVector<CandidateSliceFilter> &filters)
      : CandidateSliceProcessPipeLine(filters) {}

  SmallVector<CandidateSliceFilter> getDefaultPipeLine() {
    return SmallVector<CandidateSliceFilter>{
        alreadyTiledOpFilter, noTilingOnReductionFilter,
        exactTilingOnPackUnPackFilter, SingleCandidateInBlockFilter};
  }

  LogicalResult filter(RewriterBase &rewriter,
                       OffsetSizeAndStrideOpInterface candidate,
                       CandidateDefOrUse defOrUse) const {
    return success(llvm::all_of(
        candidateProcessFn,
        [&rewriter, &candidate, &defOrUse](const CandidateSliceFilter &filter) {
          return succeeded(filter(rewriter, candidate, defOrUse));
        }));
  }
};

static FailureOr<int64_t>
computeTileSizeProductOfCandidate(OffsetSizeAndStrideOpInterface candidate) {
  SmallVector<OpFoldResult> tileSizes = candidate.getMixedSizes();
  int64_t totalSize = 1;
  for (auto &tile : tileSizes) {
    FailureOr<int64_t> cstSize = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, tile,
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(cstSize)) {
      return failure();
    }
    totalSize *= *cstSize;
  };
  return totalSize;
}

static int TilingSizeComparer(RewriterBase &rewriter,
                              OffsetSizeAndStrideOpInterface candidateA,
                              OffsetSizeAndStrideOpInterface candidateB,
                              CandidateDefOrUse defOrUse) {
  FailureOr<int64_t> sizeProductA =
                         computeTileSizeProductOfCandidate(candidateA),
                     sizeProductB =
                         computeTileSizeProductOfCandidate(candidateB);
  if (failed(sizeProductA) || failed(sizeProductB)) {
    return 0;
  }
  // deal with equality
  if (*sizeProductA == *sizeProductB) {
    return 0;
  } else {
    return *sizeProductA < *sizeProductB ? -1 : 1;
  }
}

struct CandidateSliceComparerPipeLine
    : public CandidateSliceProcessPipeLine<CandidateSliceComparer,
                                           CandidateSliceComparerPipeLine> {
  CandidateSliceComparerPipeLine() : CandidateSliceProcessPipeLine() {}

  SmallVector<CandidateSliceComparer> getDefaultPipeLine() {
    return SmallVector<CandidateSliceComparer>{TilingSizeComparer};
  }

  bool compare(RewriterBase &rewriter,
               OffsetSizeAndStrideOpInterface candidateA,
               OffsetSizeAndStrideOpInterface candidateB,
               CandidateDefOrUse defOrUse) const {
    // deal with weak order
    int cmpResult = -1;
    for (auto &fn : candidateProcessFn) {
      cmpResult = fn(rewriter, candidateA, candidateB, defOrUse);
      if (cmpResult != 0)
        break;
    }
    return cmpResult == -1;
  }
};

struct CandidateSliceOptions {
  // Use for validity
  CandidateSliceFilterPipeLine filterPipeLine;
  // Use for performance
  CandidateSliceComparerPipeLine comparerPipeLine;

  CandidateSliceOptions() = default;

  void addFilter(const CandidateSliceFilter &filter) {
    filterPipeLine.append(filter);
  }
  void addFilter(ArrayRef<CandidateSliceFilter> filters) {
    filterPipeLine.append(filters);
  }
  void addComparer(const CandidateSliceComparer &comparer) {
    comparerPipeLine.append(comparer);
  }
  void addFilter(ArrayRef<CandidateSliceComparer> comparers) {
    comparerPipeLine.append(comparers);
  }
};

std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfOpOperand(RewriterBase &rewriter, OpOperand &operand,
                               const CandidateSliceOptions &options) {
  // a. Find the closest sliceOp
  FailureOr<tensor::ExtractSliceOp> closestSliceOp =
      getClosestExtractSliceOfOperand(operand);
  if (failed(closestSliceOp)) {
    return std::nullopt;
  }
  // b. Find the real producer and collect the sliceOp chain during backward
  // stage, sorted from inner to outer.
  SmallVector<tensor::ExtractSliceOp> backwardSlice;
  FailureOr<OpResult> realProducer =
      scfX::getRealProducerOfExtractSliceOp(*closestSliceOp, backwardSlice);
  if (failed(realProducer)) {
    return std::nullopt;
  }
  // c. Check the producer of root source if is tilable.
  Operation *producer = realProducer->getDefiningOp<TilingInterface>();
  if (!producer)
    return std::nullopt;

  CandidateDefOrUse defOrUse{*realProducer};
  // d. Filter out invalid candidates
  SmallVector<tensor::ExtractSliceOp> validCandidates =
      llvm::to_vector(llvm::make_filter_range(
          backwardSlice,
          [&rewriter, &options, &defOrUse](tensor::ExtractSliceOp &candidate) {
            return succeeded(options.filterPipeLine.filter(
                rewriter,
                cast<OffsetSizeAndStrideOpInterface>(candidate.getOperation()),
                defOrUse));
          }));
  if (validCandidates.empty())
    return std::nullopt;
  // e. Select best candidates by Cost Model
  tensor::ExtractSliceOp bestCandidate = *llvm::min_element(
      validCandidates,
      [&rewriter, &options, &defOrUse](tensor::ExtractSliceOp &candidateA,
                                       tensor::ExtractSliceOp &candidateB) {
        return options.comparerPipeLine.compare(
            rewriter,
            cast<OffsetSizeAndStrideOpInterface>(candidateA.getOperation()),
            cast<OffsetSizeAndStrideOpInterface>(candidateB.getOperation()),
            defOrUse);
      });
  // f. call tiling interface
  return scfX::tileAndFuseProducerOfSlice(rewriter, bestCandidate);
}

std::optional<SmallVector<scf::SCFFuseConsumerOfSliceResult>>
tileAndFuseConsumerOfOpResult(RewriterBase &rewriter, OpResult result,
                              const CandidateSliceOptions &options) {
  // a. Find the closest sliceOp
  FailureOr<tensor::ExtractSliceOp> closestSliceOp =
      getClosestInsertSliceOfResult(result);
  if (failed(closestSliceOp)) {
    return std::nullopt;
  }
  // b. Find the real consumers and collect the sliceOp chain during forward
  // stage, sorted from inner to outer.
  SmallVector<OffsetSizeAndStrideOpInterface> forwardSlice;
  FailureOr<SmallVector<OpOperand *>> realConsumers =
      scfX::getRealConsumersFromInsertSliceOp(*closestSliceOp, forwardSlice);
  if (failed(realConsumers)) {
    return std::nullopt;
  }

  SmallVector<scf::SCFFuseConsumerOfSliceResult> fusedResultList;
  for (auto useOperand : *realConsumers) {
    // c. Check the consumer of top level result if is tilable.
    Operation *consumer = dyn_cast<TilingInterface>(useOperand->getOwner());
    if (!consumer)
      continue;

    CandidateDefOrUse defOrUse{useOperand};
    // d. Filter out invalid candidates
    SmallVector<OffsetSizeAndStrideOpInterface> validCandidates =
        llvm::to_vector(llvm::make_filter_range(
            forwardSlice, [&rewriter, &options, &defOrUse](
                              const OffsetSizeAndStrideOpInterface &candidate) {
              return succeeded(
                  options.filterPipeLine.filter(rewriter, candidate, defOrUse));
            }));
    if (validCandidates.empty())
      continue;

    // e. Select best candidates by Cost Model
    OffsetSizeAndStrideOpInterface bestCandidate = *llvm::min_element(
        validCandidates, [&rewriter, &options, &defOrUse](
                             const OffsetSizeAndStrideOpInterface &candidateA,
                             const OffsetSizeAndStrideOpInterface &candidateB) {
          return options.comparerPipeLine.compare(rewriter, candidateA,
                                                  candidateB, defOrUse);
        });
    // f. call tiling interface
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        scfX::tileAndFuseConsumerOfSlice(rewriter, bestCandidate);

    if (succeeded(fusedResult)) {
      fusedResultList.push_back(*fusedResult);
      auto whileProducerOutOfLoopBlock =
          [&fusedResult](LoopLikeOpInterface loop) -> LogicalResult {
        Block &body = loop->getRegion(0).front();
        return failure(fusedResult.value().tiledOps[0]->getBlock() == &body);
      };
      SmallVector<LoopLikeOpInterface> outerLoops =
          scfX::getOuterNestLoopsWhile(
              bestCandidate->getParentOfType<LoopLikeOpInterface>(),
              whileProducerOutOfLoopBlock);
      // g. Manually run cse on region which contains top-level loop of
      // candidate slice in avoid of conflict with subsequent
      // `tileAndFuseConsumerOfSlice` get nest loops between next candidate
      // sliceOp and tiled producer.
      auto region = outerLoops.front()->getParentRegion();
      (void)mlir::eraseUnreachableBlocks(rewriter, {*region});
      (void)mlir::runRegionDCE(rewriter, {*region});
    }
  }
  if (fusedResultList.empty()) {
    return std::nullopt;
  } else {
    return fusedResultList;
  }
}

/// Target at following general topology:
///
/// producer1   producer2
///    \         /
///        Op
///    /         \
/// consumer1  consumer2
///
/// where:
///
/// Support iterative producer and consumer fusion in BFS fashion.
LogicalResult iterativelyFuseProducerAndConsumerOfTiledOp(
    RewriterBase &rewriter, Operation *tiledOp,
    const CandidateSliceOptions &options) {
  int numTiledOps = 0;
  std::deque<Operation *> tiledOpList = {tiledOp};
  while (!tiledOpList.empty()) {
    tiledOp = tiledOpList.front();
    tiledOpList.pop_front();
    numTiledOps++;
    // fuse producer
    for (OpOperand &operand : tiledOp->getOpOperands()) {
      if (std::optional<scf::SCFFuseProducerOfSliceResult> fuseProducerResult =
              tileAndFuseProducerOfOpOperand(rewriter, operand, options)) {
        tiledOpList.push_back(fuseProducerResult.value().tiledOps[0]);
      }
    }
    // fuse consumer(s)
    for (OpResult result : tiledOp->getResults()) {
      if (std::optional<SmallVector<scf::SCFFuseConsumerOfSliceResult>>
              fuseConsumerResults =
                  tileAndFuseConsumerOfOpResult(rewriter, result, options)) {
        for (auto &fuseConsumerResult : *fuseConsumerResults) {
          tiledOpList.push_back(fuseConsumerResult.tiledOps[0]);
        }
      }
    }
  }
  return success(numTiledOps > 1);
}

/// What is single tiled op in loop?
/// E.g.
/// %1 = scf.for(){
///   %2 = scf.for(){
///       %3 = extract_slice
///       %4 = tiled_op(%3)
///       %5 = insert %4
///       yield %5
///   }
/// }
static LogicalResult isSingleTiledOpInLoop(Operation *targetOp) {
  // 0. check tilable
  if (!isa<TilingInterface>(targetOp)) {
    return failure();
  }
  // 1. check parentOp
  auto forOp = targetOp->getParentOfType<LoopLikeOpInterface>();
  if (!forOp) {
    return failure();
  }
  // 2. check single one tiling interface in loop body
  auto walkResult = forOp->walk([&targetOp](TilingInterface op) {
    // some special op maybe already deal with in template
    if (isa<linalg::FillOp>(op))
      return WalkResult::skip();
    return op != targetOp ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }
  // 3. check whether has either extract or insert slice op
  walkResult = forOp->walk(
      [](tensor::ExtractSliceOp) { return WalkResult::interrupt(); });
  if (walkResult.wasInterrupted()) {
    return success();
  }
  walkResult = forOp->walk(
      [](tensor::InsertSliceOp) { return WalkResult::interrupt(); });
  return success(walkResult.wasInterrupted());
}

template <typename OpTy>
static bool defaultTilingOfType(RewriterBase &rewriter, Operation *op) {
  // a. Check <OpTy>
  if (!isa<TilingInterface>(op) || !isa<OpTy>(op))
    return false;
  auto tilingInterfaceOp = cast<TilingInterface>(op);

  scf::SCFTilingOptions options;
  // b. Get default tiling size
  SmallVector<utils::IteratorType> iteratorTypes =
      tilingInterfaceOp.getLoopIteratorTypes();

  SmallVector<OpFoldResult> defaultTileSize(iteratorTypes.size(),
                                            rewriter.getIndexAttr(0));

  for (auto &&[en, iterType] : llvm::enumerate(iteratorTypes)) {
    // All outer non reduction loop should contribute parallelism. In another
    // word, all reduction dimensions should not be tiled.
    if (iterType == utils::IteratorType::parallel &&
        (en != iteratorTypes.size() - 1 ||
         llvm::count(iteratorTypes, utils::IteratorType::reduction))) {
      defaultTileSize[en] = rewriter.getIndexAttr(1);
    }
  }

  options.setTileSizes(defaultTileSize);
  // c. Set loop type
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  // d. Use builtin tiling interface
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
  if (succeeded(tilingResult)) {
    rewriter.replaceOp(op, tilingResult->replacements);
    return true;
  } else {
    return false;
  }
}

struct IterativeFusionOptions {
  bool useCostModel = false;
};

struct SystemDesc {
  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads() {
    std::optional<Attribute> numThreads = layout.getDevicePropertyValue(
        Builder(ctx).getStringAttr("CPU" /* device ID*/),
        Builder(ctx).getStringAttr("num_threads"));
    if (numThreads && isa<IntegerAttr>(*numThreads)) {
      return dyn_cast<IntegerAttr>(*numThreads).getInt();
    }
    return 1;
  }
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel) {
    if (cacheLevel == 1) {
      std::optional<Attribute> cacheSize = layout.getDevicePropertyValue(
          Builder(ctx).getStringAttr("CPU" /* device ID*/),
          Builder(ctx).getStringAttr("L1_cache_size_in_bytes"));
      if (cacheSize && isa<IntegerAttr>(*cacheSize)) {
        return dyn_cast<IntegerAttr>(*cacheSize).getInt();
      }
    } else if (cacheLevel == 2) {
      std::optional<Attribute> cacheSize = layout.getDevicePropertyValue(
          Builder(ctx).getStringAttr("CPU" /* device ID*/),
          Builder(ctx).getStringAttr("L2_cache_size_in_bytes"));
      if (cacheSize && isa<IntegerAttr>(*cacheSize)) {
        return dyn_cast<IntegerAttr>(*cacheSize).getInt();
      }
    } else if (cacheLevel == 3) {
      std::optional<Attribute> cacheSize = layout.getDevicePropertyValue(
          Builder(ctx).getStringAttr("CPU" /* device ID*/),
          Builder(ctx).getStringAttr("L3_cache_size_in_bytes"));
      if (cacheSize && isa<IntegerAttr>(*cacheSize)) {
        return dyn_cast<IntegerAttr>(*cacheSize).getInt();
      }
    }
    return 0;
  }

  // get the maximum vector length in bits
  size_t getMaxVectorLength() {
    std::optional<Attribute> maxVectorLength = layout.getDevicePropertyValue(
        Builder(ctx).getStringAttr("CPU" /* device ID*/),
        Builder(ctx).getStringAttr("max_vector_width"));
    if (maxVectorLength && isa<IntegerAttr>(*maxVectorLength)) {
      return dyn_cast<IntegerAttr>(*maxVectorLength).getInt();
    }
    return 512;
  }

  SystemDesc(ModuleOp m) : layout(m), ctx(m->getContext()) {}

private:
  DataLayout layout;
  MLIRContext *ctx;
};

void iterativeTilingAndFusion(RewriterBase &rewriter, func::FuncOp &f,
                              const IterativeFusionOptions &fuseOptions) {
  // Collect untiled and tiled ops respectively
  llvm::SetVector<Operation *> singleTiledOpInLoop, unTiledOps;

  auto collectUnTiledOps = [&f, &unTiledOps]() -> bool {
    // Reset
    unTiledOps.clear();
    // Pre-order walk through funcOp
    f->walk<WalkOrder::PreOrder>([&unTiledOps](Operation *op) {
      if (isa<LoopLikeOpInterface>(op)) {
        return WalkResult::skip();
      }
      if (isa<TilingInterface>(op) && !op->use_empty()) {
        auto parentLoop = op->getParentOfType<LoopLikeOpInterface>();
        if (!parentLoop.getOperation()) {
          unTiledOps.insert(op);
        }
      }
      return WalkResult::advance();
    });
    return !unTiledOps.empty();
  };

  auto collectSingleTiledOpInLoop = [&f, &singleTiledOpInLoop]() -> bool {
    // Reset
    singleTiledOpInLoop.clear();
    // Walk through funcOp
    f->walk([&singleTiledOpInLoop](Operation *op) {
      // Target at certain kind of tiled op, such as matmul/conv implemented
      // by multiple level of nest loops and candidate slices for better
      // utilization of parallelism and memory hierarchy.
      if (succeeded(isSingleTiledOpInLoop(op))) {
        singleTiledOpInLoop.insert(op);
      }
    });
    return !singleTiledOpInLoop.empty();
  };

  SystemDesc sysDesc(f->getParentOfType<ModuleOp>());
  // Flexible options to control which candidate slice would be selected from
  // the view of both validity and performance.
  CandidateSliceOptions sliceOptions;
  // Since most filters regarding to validity have already been built-in
  // enabled. Users could focus on performance related filters, a.k.a. cost
  // model.
  if (fuseOptions.useCostModel) {
    // Customized filter by cost model.
    CandidateSliceFilter costModelFilter =
        [&sysDesc](RewriterBase &rewriter,
                   OffsetSizeAndStrideOpInterface candidate,
                   CandidateDefOrUse defOrUse) -> LogicalResult {
      // Get cache size
      size_t l2CacheSize = sysDesc.getCacheSize(2);
      FailureOr<int64_t> tileSizeProduct =
          computeTileSizeProductOfCandidate(candidate);
      return success(succeeded(tileSizeProduct) &&
                     (*tileSizeProduct <= (int64_t)l2CacheSize));
    };
    sliceOptions.addFilter(costModelFilter);
  }

  // Iterative tiling and fusion until exhaustion.
  while (collectUnTiledOps()) {
    // If existing tiled op before tiling.
    if (collectSingleTiledOpInLoop()) {
      // Sort by topology
      mlir::topologicalSort(singleTiledOpInLoop);
      // Record if any fusion happens
      bool changed = false;
      // Iteratively fuse in forward and backward fashion.
      llvm::for_each(singleTiledOpInLoop, [&rewriter, &sliceOptions,
                                           &changed](Operation *tiledOp) {
        changed |= succeeded(iterativelyFuseProducerAndConsumerOfTiledOp(
            rewriter, tiledOp, sliceOptions));
      });
      if (!changed) {
        // If no new fusion happens, terminate iteration.
        break;
      } else {
        (void)mlir::eraseUnreachableBlocks(rewriter, {f.getRegion()});
        (void)mlir::runRegionDCE(rewriter, {f.getRegion()});
      }
    } else {
      // Auto tiling with default tile size if no tiled op found. Follow tiling
      // priority based on OpTy: `Contraction`->`Reduction`->`Elementwise`.
      SmallVector<std::function<bool(RewriterBase &, Operation *)>>
          priorityTilingPipeLine = {
              defaultTilingOfType<mlir::linalg::ContractionOpInterface>,
              defaultTilingOfType<mlir::linalg::ReduceOp>,
              defaultTilingOfType<mlir::linalg::LinalgOp>};
      if (llvm::all_of(
              priorityTilingPipeLine,
              [&rewriter, &unTiledOps](
                  function_ref<bool(RewriterBase &, Operation *)> tilingFn) {
                return !llvm::any_of(unTiledOps,
                                     std::bind(tilingFn, std::ref(rewriter),
                                               std::placeholders::_1));
              })) {
        // If no op can be tiled
        break;
      }
    }
  }
}

struct FineGrainedFusion
    : public impl::FineGrainedFusionBase<FineGrainedFusion> {
  using FineGrainedFusionBase::FineGrainedFusionBase;

public:
  void runOnOperation() final {
    auto &ctx = getContext();
    {
      // Get funcOp
      func::FuncOp func = getOperation();
      // Get rewriter
      IRRewriter rewriter(&ctx);
      // Run iterative fusion
      iterativeTilingAndFusion(rewriter, func,
                               IterativeFusionOptions{useCostModel});
    }

    {
      RewritePatternSet patternSet(&ctx);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patternSet))))
        signalPassFailure();
    }
  }
};

} // namespace gc
} // namespace mlir