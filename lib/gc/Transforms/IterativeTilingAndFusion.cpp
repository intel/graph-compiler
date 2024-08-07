//===-- IterativeTilingAndFusion.cpp - Iterative Tiling+Fusion --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
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
#include <unordered_map>

#include "TilingUsingInterfaceX.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_ITERATIVETILINGANDFUSION
#include "gc/Transforms/Passes.h.inc"

static FailureOr<tensor::ExtractSliceOp>
getClosestExtractSliceOfOperand(OpOperand &operand) {
  if (auto iterArg = dyn_cast<BlockArgument>(operand.get())) {
    if (auto loop =
            dyn_cast<LoopLikeOpInterface>(iterArg.getOwner()->getParentOp()))
      return getClosestExtractSliceOfOperand(*loop.getTiedLoopInit(iterArg));
  }

  Operation *defineOp = operand.get().getDefiningOp();
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defineOp))
    return sliceOp;
  // For downstream cases
  if (isa<linalg::FillOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
          defineOp))
    return getClosestExtractSliceOfOperand(defineOp->getOpOperand(0));

  return failure();
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
      if (isa<LoopLikeOpInterface, RegionBranchOpInterface>(
              yieldOp->getParentOp())) {
        return getClosestInsertSliceOfResult(
            yieldOp->getParentOp()->getResult(useOfResult.getOperandNumber()));
      }
    } else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
                   useOfResult.getOwner())) {
      return getClosestInsertSliceOfResult(
          useOfResult.getOwner()->getResult(useOfResult.getOperandNumber()));
    }
  }

  if (!llvm::detail::isPresent(sliceOp))
    return failure();

  return sliceOp;
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

using CandidateSliceComparer = std::function<int(
    OffsetSizeAndStrideOpInterface, OffsetSizeAndStrideOpInterface)>;

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
          cstIterDomain != cstTileSizes)
        return failure();
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
      targetInnerTileSizes = packOp.getMixedTiles();
      targetTileSizes = llvm::to_vector(
          ArrayRef(tileSizes).take_back(targetInnerTileSizes.size()));
    } else {
      // tileSize comes from OpOperand
      targetTileSizes = tileSizes;
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
      targetTileSizes = tileSizes;
      DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
          unPackOp.getDimAndTileMapping();
      targetInnerTileSizes.resize(dimAndTileMapping.size());
      for (const auto &dimAndTile : dimAndTileMapping) {
        targetInnerTileSizes[dimAndTile.first] = dimAndTile.second;
      }
    } else {
      // tileSize comes from OpOperand
      targetInnerTileSizes = unPackOp.getMixedTiles();
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

static LogicalResult unTiledOpFilter(RewriterBase &rewriter,
                                     OffsetSizeAndStrideOpInterface candidate,
                                     CandidateDefOrUse defOrUse) {
  // In general tiledOp would not have uses any more.
  return failure(defOrUse.ownerOp->use_empty());
}

static LogicalResult
NonContractionOpFilter(RewriterBase &rewriter,
                       OffsetSizeAndStrideOpInterface candidate,
                       CandidateDefOrUse defOrUse) {
  // Currently this pass focuses on fine-grained fusion, which does not expect
  // two consecutive contraction ops.
  return failure(isa<mlir::linalg::ContractionOpInterface>(defOrUse.ownerOp));
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
            realProducer->getDefiningOp() == defOrUse.ownerOp)
          return failure();
      } else {
        SmallVector<OffsetSizeAndStrideOpInterface> forwardSlice;
        FailureOr<SmallVector<OpOperand *>> realConsumers =
            scfX::getRealConsumersFromInsertSliceOp(otherCandidate,
                                                    forwardSlice);
        if (succeeded(realConsumers) &&
            llvm::any_of(*realConsumers, [&defOrUse](OpOperand *use) {
              return use->getOwner() == defOrUse.ownerOp;
            }))
          return failure();
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
        unTiledOpFilter, NonContractionOpFilter, noTilingOnReductionFilter,
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

static int TilingSizeComparer(OffsetSizeAndStrideOpInterface candidateA,
                              OffsetSizeAndStrideOpInterface candidateB) {
  FailureOr<int64_t> sizeProductA =
                         computeTileSizeProductOfCandidate(candidateA),
                     sizeProductB =
                         computeTileSizeProductOfCandidate(candidateB);
  if (failed(sizeProductA) || failed(sizeProductB))
    return 0;
  // deal with equality
  if (*sizeProductA == *sizeProductB)
    return 0;

  return *sizeProductA < *sizeProductB ? -1 : 1;
}

struct CandidateSliceComparerPipeLine
    : public CandidateSliceProcessPipeLine<CandidateSliceComparer,
                                           CandidateSliceComparerPipeLine> {
  CandidateSliceComparerPipeLine() : CandidateSliceProcessPipeLine() {}

  SmallVector<CandidateSliceComparer> getDefaultPipeLine() {
    return SmallVector<CandidateSliceComparer>{TilingSizeComparer};
  }

  bool compare(OffsetSizeAndStrideOpInterface candidateA,
               OffsetSizeAndStrideOpInterface candidateB) const {
    // deal with weak order
    int cmpResult = -1;
    llvm::any_of(candidateProcessFn, [&cmpResult, &candidateA, &candidateB](
                                         const CandidateSliceComparer &fn) {
      cmpResult = fn(candidateA, candidateB);
      return cmpResult != 0;
    });
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

static FailureOr<OffsetSizeAndStrideOpInterface> filterAndSelectCandidate(
    RewriterBase &rewriter,
    ArrayRef<OffsetSizeAndStrideOpInterface> candidateSliceList,
    const CandidateDefOrUse &defOrUse, const CandidateSliceOptions &options) {
  SmallVector<OffsetSizeAndStrideOpInterface> validCandidates =
      llvm::to_vector(llvm::make_filter_range(
          candidateSliceList,
          [&rewriter, &options,
           &defOrUse](const OffsetSizeAndStrideOpInterface &candidate) {
            return succeeded(
                options.filterPipeLine.filter(rewriter, candidate, defOrUse));
          }));
  if (validCandidates.empty())
    return failure();

  OffsetSizeAndStrideOpInterface bestCandidate = *llvm::min_element(
      validCandidates, [&options](OffsetSizeAndStrideOpInterface &candidateA,
                                  OffsetSizeAndStrideOpInterface &candidateB) {
        return options.comparerPipeLine.compare(candidateA, candidateB);
      });
  return bestCandidate;
}

std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfOpOperand(RewriterBase &rewriter, OpOperand &operand,
                               const CandidateSliceOptions &options) {
  // a. Find the closest sliceOp
  FailureOr<tensor::ExtractSliceOp> closestSliceOp =
      getClosestExtractSliceOfOperand(operand);
  if (failed(closestSliceOp))
    return std::nullopt;

  // b. Find the real producer and collect the sliceOp chain during backward
  // stage, sorted from inner to outer.
  SmallVector<tensor::ExtractSliceOp> backwardSlice;
  FailureOr<OpResult> realProducer =
      scfX::getRealProducerOfExtractSliceOp(*closestSliceOp, backwardSlice);
  if (failed(realProducer))
    return std::nullopt;

  // c. Check the producer of root source if is tilable.
  Operation *producer = realProducer->getDefiningOp<TilingInterface>();
  if (!producer)
    return std::nullopt;

  CandidateDefOrUse defOrUse{*realProducer};
  // d. Filter out invalid candidates and select best candidates
  SmallVector<OffsetSizeAndStrideOpInterface> ossBackwardSlice =
      llvm::map_to_vector(backwardSlice,
                          [](tensor::ExtractSliceOp &extractSlice) {
                            return cast<OffsetSizeAndStrideOpInterface>(
                                extractSlice.getOperation());
                          });
  FailureOr<OffsetSizeAndStrideOpInterface> bestCandidate =
      filterAndSelectCandidate(rewriter, ossBackwardSlice, defOrUse, options);
  if (failed(bestCandidate))
    return std::nullopt;

  // e. call tiling interface
  return scfX::tileAndFuseProducerOfSlice(rewriter, *bestCandidate);
}

std::optional<SmallVector<scf::SCFFuseConsumerOfSliceResult>>
tileAndFuseConsumerOfOpResult(RewriterBase &rewriter, OpResult result,
                              const CandidateSliceOptions &options) {
  // a. Find the closest sliceOp
  FailureOr<tensor::ExtractSliceOp> closestSliceOp =
      getClosestInsertSliceOfResult(result);
  if (failed(closestSliceOp))
    return std::nullopt;

  // b. Find the real consumers and collect the sliceOp chain during forward
  // stage, sorted from inner to outer.
  SmallVector<OffsetSizeAndStrideOpInterface> forwardSlice;
  FailureOr<SmallVector<OpOperand *>> realConsumers =
      scfX::getRealConsumersFromInsertSliceOp(*closestSliceOp, forwardSlice);
  if (failed(realConsumers))
    return std::nullopt;

  SmallVector<scf::SCFFuseConsumerOfSliceResult> fusedResultList;
  for (auto useOperand : *realConsumers) {
    // c. Check the consumer of top level result if is tilable.
    Operation *consumer = dyn_cast<TilingInterface>(useOperand->getOwner());
    if (!consumer)
      continue;

    CandidateDefOrUse defOrUse{useOperand};
    // d. Filter out invalid candidates and select best candidates
    FailureOr<OffsetSizeAndStrideOpInterface> bestCandidate =
        filterAndSelectCandidate(rewriter, forwardSlice, defOrUse, options);
    if (failed(bestCandidate))
      continue;

    // e. call tiling interface
    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        scfX::tileAndFuseConsumerOfSlice(rewriter, *bestCandidate);

    if (succeeded(fusedResult)) {
      fusedResultList.push_back(*fusedResult);
      auto whileProducerOutOfLoopBlock =
          [&fusedResult](LoopLikeOpInterface loop) -> LogicalResult {
        Block &body = loop->getRegion(0).front();
        return failure(fusedResult.value().tiledOps[0]->getBlock() == &body);
      };
      SmallVector<LoopLikeOpInterface> outerLoops =
          scfX::getOuterNestLoopsWhile(
              (*bestCandidate)->getParentOfType<LoopLikeOpInterface>(),
              whileProducerOutOfLoopBlock);
      // g. Manually run cse on region which contains top-level loop of
      // candidate slice in avoid of conflict with subsequent
      // `tileAndFuseConsumerOfSlice` get nest loops between next candidate
      // sliceOp and tiled producer.
      (void)mlir::simplifyRegions(rewriter,
                                  {*outerLoops.front()->getParentRegion()});
    }
  }
  if (fusedResultList.empty())
    return std::nullopt;

  return fusedResultList;
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
  unsigned numTiledOps = 0;
  std::deque<Operation *> tiledOpList = {tiledOp};
  while (!tiledOpList.empty()) {
    tiledOp = tiledOpList.front();
    tiledOpList.pop_front();
    numTiledOps++;
    // fuse producer
    for (OpOperand &operand : tiledOp->getOpOperands()) {
      if (std::optional<scf::SCFFuseProducerOfSliceResult> fuseProducerResult =
              tileAndFuseProducerOfOpOperand(rewriter, operand, options))
        tiledOpList.push_back(fuseProducerResult.value().tiledOps[0]);
    }
    // fuse consumer(s)
    for (OpResult result : tiledOp->getResults()) {
      if (std::optional<SmallVector<scf::SCFFuseConsumerOfSliceResult>>
              fuseConsumerResults =
                  tileAndFuseConsumerOfOpResult(rewriter, result, options)) {
        for (auto &fuseConsumerResult : *fuseConsumerResults)
          tiledOpList.push_back(fuseConsumerResult.tiledOps[0]);
      }
    }
  }
  return success(numTiledOps > 1);
}

/// This is a workaround to deal with LinalgXOp
static bool isTilableLinalgXOp(Operation *op) {
  return isa<linalgx::BatchReduceMatmulVnniOp, linalgx::MultiBatchMatmulOp,
             linalgx::Mm2DVnniOp, linalgx::Mm4DVnniOp>(op);
}

/// Check if tiled op inside a loop?
/// E.g.
/// %1 = scf.for(){
///   %2 = scf.for(){
///       %3 = extract_slice
///       %4 = tiled_op(%3)
///       %5 = insert %4
///       yield %5
///   }
/// }
static LogicalResult isTiledOpInLoop(Operation *targetOp) {
  // 1. check tilable
  if (!isa<TilingInterface>(targetOp) && !isTilableLinalgXOp(targetOp))
    return failure();
  // 2. check parentOp
  auto forOp = targetOp->getParentOfType<LoopLikeOpInterface>();
  if (!forOp)
    return failure();

  // 3. check whether has either extract or insert slice op
  auto walkResult = forOp->walk(
      [](tensor::ExtractSliceOp) { return WalkResult::interrupt(); });
  if (walkResult.wasInterrupted())
    return success();
  walkResult = forOp->walk(
      [](tensor::InsertSliceOp) { return WalkResult::interrupt(); });
  return success(walkResult.wasInterrupted());
}

using OpTileSizeMap = std::unordered_map<std::string, SmallVector<int64_t>>;

template <typename OpTy>
static FailureOr<scf::SCFTilingResult>
defaultTilingOfType(RewriterBase &rewriter, Operation *op,
                    const OpTileSizeMap &tsMap) {
  // a. Check <OpTy>
  if (!isa<TilingInterface>(op) || !isa<OpTy>(op))
    return failure();
  auto tilingInterfaceOp = cast<TilingInterface>(op);

  scf::SCFTilingOptions options;
  // b. Get default tiling size
  SmallVector<utils::IteratorType> iteratorTypes =
      tilingInterfaceOp.getLoopIteratorTypes();

  SmallVector<OpFoldResult> defaultTileSize;

  std::string opName = op->getName().getStringRef().str();
  // Erase dialect name, such as Linalg or Tensor.
  opName.erase(0, opName.find(".") + 1);

  if (tsMap.count(opName)) {
    SmallVector<int64_t> userDefaultTileSize = tsMap.find(opName)->second;
    defaultTileSize =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(userDefaultTileSize));
  } else {
    defaultTileSize.resize(iteratorTypes.size(), rewriter.getIndexAttr(0));
    for (auto &&[en, iterType] : llvm::enumerate(iteratorTypes)) {
      // All outer non reduction loop should contribute parallelism. In another
      // word, all reduction dimensions should not be tiled.
      if (iterType == utils::IteratorType::parallel &&
          (en != iteratorTypes.size() - 1 ||
           llvm::count(iteratorTypes, utils::IteratorType::reduction)))
        defaultTileSize[en] = rewriter.getIndexAttr(1);
    }
  }
  // If the tile sizes are all zero, no tiling would happen.
  if (llvm::all_of(defaultTileSize, isZeroIndex))
    return failure();

  options.setTileSizes(defaultTileSize);
  // c. Set loop type
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  // d. Use builtin tiling interface
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

  if (failed(tilingResult))
    return failure();

  return tilingResult;
}

using DefaultTilingFn = std::function<FailureOr<scf::SCFTilingResult>(
    RewriterBase &, Operation *, const OpTileSizeMap &)>;

void iterativeTilingAndFusionUntilExhaustion(
    RewriterBase &rewriter, func::FuncOp &f,
    const CandidateSliceOptions &sliceOptions, const OpTileSizeMap &tsMap) {
  // Collect untiled and tiled ops respectively
  llvm::SetVector<Operation *> tiledOps, unTiledOps;

  auto collectUnTiledOps = [&f, &unTiledOps]() -> bool {
    // Reset
    unTiledOps.clear();
    // Pre-order walk through funcOp
    f->walk<WalkOrder::PreOrder>([&unTiledOps](Operation *op) {
      if (isa<LoopLikeOpInterface>(op))
        return WalkResult::skip();
      if (isa<TilingInterface>(op)) {
        auto parentLoop = op->getParentOfType<LoopLikeOpInterface>();
        auto parentGeneric = op->getParentOfType<linalg::GenericOp>();
        if (!llvm::detail::isPresent(parentLoop) &&
            !llvm::detail::isPresent(parentGeneric))
          unTiledOps.insert(op);
      }
      return WalkResult::advance();
    });
    return !unTiledOps.empty();
  };

  // Walk through funcOp
  f->walk([&tiledOps](Operation *op) {
    if (succeeded(isTiledOpInLoop(op))) {
      tiledOps.insert(op);
    }
  });

  // Iterative tiling and fusion until exhaustion.
  while (collectUnTiledOps()) {
    // If existing tiled op before tiling.
    if (!tiledOps.empty()) {
      // Sort by topology
      mlir::topologicalSort(tiledOps);
      // Record if any fusion happens
      bool changed = false;
      // Iteratively fuse in forward and backward fashion.
      llvm::for_each(
          tiledOps, [&rewriter, &sliceOptions, &changed](Operation *tiledOp) {
            changed |= succeeded(iterativelyFuseProducerAndConsumerOfTiledOp(
                rewriter, tiledOp, sliceOptions));
          });
      tiledOps.clear();
      if (changed)
        (void)mlir::simplifyRegions(rewriter, {f.getRegion()});
    } else {
      // Auto tiling with default tile size if no tiled op found. Follow tiling
      // priority based on OpTy: `Contraction`->`Reduction`->`Elementwise`.
      SmallVector<DefaultTilingFn> priorityTilingPipeLine = {
          defaultTilingOfType<mlir::linalg::ContractionOpInterface>,
          defaultTilingOfType<mlir::linalg::ReduceOp>,
          defaultTilingOfType<TilingInterface>};

      for (auto &tilingFn : priorityTilingPipeLine) {
        for (auto &op : unTiledOps) {
          FailureOr<scf::SCFTilingResult> tilingResult =
              tilingFn(rewriter, op, tsMap);
          if (succeeded(tilingResult)) {
            tiledOps.insert(tilingResult->tiledOps[0]);
            rewriter.replaceOp(op, tilingResult->replacements);
            break;
          }
        }
        if (!tiledOps.empty())
          break;
      }
      // If no op can be tiled
      if (tiledOps.empty())
        return;
    }
  }
}

static OpTileSizeMap defaultTileSizeParser(ArrayRef<std::string> strArgs) {
  OpTileSizeMap tsMap;
  char warning[] =
      "Please follow correct argument format: opType:{ts1,ts2,...}";
  for (auto str : strArgs) {
    str.erase(llvm::remove_if(str, llvm::isSpace), str.end());
    size_t pos = str.find(":");
    if (pos == std::string::npos)
      llvm_unreachable(warning);

    std::string opType = str.substr(0, pos);
    std::string strTileSize = str.erase(0, pos + 1);
    if (strTileSize.size() <= 2 || strTileSize.front() != '{' ||
        strTileSize.back() != '}')
      llvm_unreachable(warning);

    strTileSize = strTileSize.substr(1, strTileSize.size() - 2);
    SmallVector<int64_t> intTileSize;
    while ((pos = strTileSize.find(",")) != std::string::npos) {
      intTileSize.push_back(std::stoi(strTileSize.substr(0, pos)));
      strTileSize.erase(0, pos + 1);
    }
    intTileSize.push_back(std::stoi(strTileSize));
    tsMap[opType] = intTileSize;
  }
  return tsMap;
}

struct IterativeTilingAndFusion
    : public impl::IterativeTilingAndFusionBase<IterativeTilingAndFusion> {
  using IterativeTilingAndFusionBase::IterativeTilingAndFusionBase;

public:
  void runOnOperation() final {
    auto &ctx = getContext();
    // Get funcOp
    func::FuncOp func = getOperation();
    // Get system descriptor
    CPUTargetDescriptionAnalysis sysDesc =
        getAnalysis<CPUTargetDescriptionAnalysis>();
    // Flexible options to control which candidate slice would be selected from
    // the view of both validity and performance.
    CandidateSliceOptions sliceOptions;
    // Since most filters regarding to validity have already been built-in
    // enabled. Users could focus on performance related filters, a.k.a. cost
    // model. E.g.
    if (useCostModel) {
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
    OpTileSizeMap tsMap = defaultTileSizeParser(defaultTileSize);
    // Get rewriter
    IRRewriter rewriter(&ctx);
    // Run iterative fusion
    iterativeTilingAndFusionUntilExhaustion(rewriter, func, sliceOptions,
                                            tsMap);
  }
};

} // namespace gc
} // namespace mlir