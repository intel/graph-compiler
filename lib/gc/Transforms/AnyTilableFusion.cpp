//===-- AnyTilableFusion.cpp - Fusion For Any Tilable Op --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#include "gc/Transforms/Passes.h"

#include <llvm/Support/Debug.h>

#include <memory>

#include "TilingUsingInterfaceX.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_ANYTILABLEFUSION
#include "gc/Transforms/Passes.h.inc"

struct SystemDesc {
  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads();
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel);
};

template <typename T> class FusionAnchorBase {
  static_assert(
      llvm::is_one_of<T, tensor::ExtractSliceOp, tensor::InsertSliceOp,
                      OffsetSizeAndStrideOpInterface>::value,
      "Fusion Anchor only expect either ExtractSliceOp or "
      "InsertSliceOp type as "
      "its candidates");
  Operation *fusableOp;
  SmallVector<T> candidateSliceOpList;

public:
  FusionAnchorBase(Operation *fusableOp);
  // append candidate slice op into Fusion Anchor with verifying its TileSizes
  void appendCandidateWithVerifyTileSizes(
      RewriterBase &rewriter,
      ArrayRef<OffsetSizeAndStrideOpInterface> candidateList,
      AffineMap fusableValueMap);
  void appendCandidateWithVerifyTileSizes(
      RewriterBase &rewriter,
      ArrayRef<OffsetSizeAndStrideOpInterface> candidateList);
  // select best Fusion Anchor from two perspective based on Cost Model
  FailureOr<T> selectCandidateByCostModel(RewriterBase &rewriter, Location loc,
                                          SystemDesc desc);
  // get operation to be fused
  Operation *getFusableOp() { return fusableOp; }
};

template <typename T>
FusionAnchorBase<T>::FusionAnchorBase(Operation *argFusableOp)
    : fusableOp(argFusableOp) {}

static LogicalResult
verifyTilableOpTileSizesOnAffineMap(RewriterBase &rewriter, Operation *op,
                                    AffineMap map,
                                    ArrayRef<OpFoldResult> tileSizes) {
  if (!isa<linalg::LinalgOp>(op))
    return failure();
  TilingInterface tilableOp = dyn_cast<TilingInterface>(op);
  SmallVector<Range> iterDomain = tilableOp.getIterationDomain(rewriter);
  SmallVector<utils::IteratorType> iterTypes = tilableOp.getLoopIteratorTypes();
  // check reduction iteration is full on TileSizes
  for (const auto &resultExpr : llvm::enumerate(map.getResults())) {
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

template <typename T>
void FusionAnchorBase<T>::appendCandidateWithVerifyTileSizes(
    RewriterBase &rewriter,
    ArrayRef<OffsetSizeAndStrideOpInterface> candidateList,
    AffineMap fusableValueMap) {
  for (auto candidate : candidateList) {
    if (succeeded(verifyTilableOpTileSizesOnAffineMap(
            rewriter, fusableOp, fusableValueMap, candidate.getMixedSizes()))) {
      candidateSliceOpList.push_back(cast<T>(candidate));
    }
  }
}

template <typename T>
static LogicalResult
verifyTilableOpTileSizesOnDimAndTileMap(RewriterBase &rewriter, Operation *op,
                                        ArrayRef<OpFoldResult> tileSizes) {
  if (!isa<tensor::PackOp, tensor::UnPackOp>(op))
    return failure();
  // collect target TileSizes and InnerTileSize to compare
  SmallVector<OpFoldResult> targetTileSizes, targetInnerTileSizes;
  if (auto packOp = dyn_cast<tensor::PackOp>(op)) {
    // tileSize comes from OpResult
    if (std::is_same<T, tensor::ExtractSliceOp>::value) {
      targetInnerTileSizes = packOp.getInnerTiles();
      targetTileSizes =
          llvm::to_vector(tileSizes.take_back(targetInnerTileSizes.size()));
    } else // tileSize comes from OpOperand
      if (std::is_same<T, OffsetSizeAndStrideOpInterface>::value) {
        targetTileSizes = llvm::to_vector(tileSizes);
        DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
            packOp.getDimAndTileMapping();
        targetInnerTileSizes.resize(dimAndTileMapping.size());
        for (const auto &dimAndTile : dimAndTileMapping) {
          targetInnerTileSizes[dimAndTile.first] = dimAndTile.second;
        }
      }
  } else if (auto unPackOp = dyn_cast<tensor::UnPackOp>(op)) {
    // tileSize comes from OpResult
    if (std::is_same<T, tensor::ExtractSliceOp>::value) {
      targetTileSizes = llvm::to_vector(tileSizes);
      DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
          unPackOp.getDimAndTileMapping();
      targetInnerTileSizes.resize(dimAndTileMapping.size());
      for (const auto &dimAndTile : dimAndTileMapping) {
        targetInnerTileSizes[dimAndTile.first] = dimAndTile.second;
      }
    } else // tileSize comes from OpOperand
      if (std::is_same<T, OffsetSizeAndStrideOpInterface>::value) {
        targetInnerTileSizes = unPackOp.getInnerTiles();
        targetTileSizes =
            llvm::to_vector(tileSizes.take_back(targetInnerTileSizes.size()));
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

template <typename T>
void FusionAnchorBase<T>::appendCandidateWithVerifyTileSizes(
    RewriterBase &rewriter,
    ArrayRef<OffsetSizeAndStrideOpInterface> candidateList) {
  for (auto candidate : candidateList) {
    if (succeeded(verifyTilableOpTileSizesOnDimAndTileMap<T>(
            rewriter, fusableOp, candidate.getMixedSizes()))) {
      candidateSliceOpList.push_back(cast<T>(candidate));
    }
  }
}

template <typename T>
FailureOr<T>
FusionAnchorBase<T>::selectCandidateByCostModel(RewriterBase &rewriter,
                                                Location loc, SystemDesc desc) {
  if (candidateSliceOpList.empty())
    return failure();
  /// TODO: use cost model
  return cast<T>(candidateSliceOpList.front());
}

// Target at tensor.extract_slice
class ProducerFusionAnchor : public FusionAnchorBase<tensor::ExtractSliceOp> {
public:
  ProducerFusionAnchor(
      RewriterBase &rewriter, Operation *producerOp, OpResult producerValue,
      ArrayRef<tensor::ExtractSliceOp> candidateExtractSliceOpList)
      : FusionAnchorBase<tensor::ExtractSliceOp>(producerOp) {
    auto candidateList = llvm::map_to_vector(
        candidateExtractSliceOpList,
        [](tensor::ExtractSliceOp sliceOp) -> OffsetSizeAndStrideOpInterface {
          return sliceOp;
        });
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(producerOp)) {
      appendCandidateWithVerifyTileSizes(
          rewriter, candidateList,
          linalgOp.getIndexingMapMatchingResult(producerValue));
    } else if (isa<tensor::PackOp, tensor::UnPackOp, tensor::PadOp>(
                   producerOp)) {
      appendCandidateWithVerifyTileSizes(rewriter, candidateList);
    }
  }
};

// Target at both tensor.insert_slice and tensor.parallel_insert_slice
class ConsumerFusionAnchor
    : public FusionAnchorBase<OffsetSizeAndStrideOpInterface> {
public:
  ConsumerFusionAnchor(
      RewriterBase &rewriter, Operation *consumerOp, OpOperand &consumerValue,
      ArrayRef<OffsetSizeAndStrideOpInterface> candidateInsertSliceOpList)
      : FusionAnchorBase<OffsetSizeAndStrideOpInterface>(consumerOp) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {
      appendCandidateWithVerifyTileSizes(
          rewriter, candidateInsertSliceOpList,
          linalgOp.getMatchingIndexingMap(&consumerValue));
    } else if (isa<tensor::PackOp, tensor::UnPackOp, tensor::PadOp>(
                   consumerOp)) {
      appendCandidateWithVerifyTileSizes(rewriter, candidateInsertSliceOpList);
    }
  }
};

// maximum recursive time
#define MAX_DEPTH 5

/** Get the Root source of target ExtractSliceOp
 * %0 =
 * %1 = scf.for(%arg1 = %0)
 *  %2 = extract %arg1
 *  %3 = scf.for(%arg2 = %2)
 *      %4 = extract %args2
 *      ...
 *
 * @param targetSliceOp: %4 = extract %args2
 * @return Result Value: %0
 *         Collected insertExtractOp List during walk including targetSliceOp:
 *                %4 = extract %args2 and %2 = extract %arg1
 */
static FailureOr<std::pair<Value, SmallVector<tensor::ExtractSliceOp>>>
getRootSourceOfExtractSliceOp(tensor::ExtractSliceOp targetSliceOp,
                              int curDepth = 0) {
  // control recursive time in avoid of stack overflow
  if (curDepth > MAX_DEPTH)
    return failure();

  SmallVector<tensor::ExtractSliceOp> candidateSliceOpList;
  candidateSliceOpList.push_back(targetSliceOp);
  Value rootSource = targetSliceOp.getSourceMutable().get();

  while (true) {
    if (auto iterArg = dyn_cast<BlockArgument>(rootSource)) {
      if (auto outerLoop = dyn_cast<LoopLikeOpInterface>(
              iterArg.getOwner()->getParentOp())) {
        rootSource = outerLoop.getTiedLoopInit(iterArg)->get();
        continue;
      }
      return failure();
    } else if (auto sliceOp =
                   rootSource.getDefiningOp<tensor::ExtractSliceOp>()) {
      // walk up loop to find larger candidate extractSliceOp
      auto resultAndSliceOpsPair =
          getRootSourceOfExtractSliceOp(sliceOp, curDepth + 1);
      if (failed(resultAndSliceOpsPair))
        return failure();
      candidateSliceOpList.append((*resultAndSliceOpsPair).second.begin(),
                                  (*resultAndSliceOpsPair).second.end());
      return std::make_pair((*resultAndSliceOpsPair).first,
                            candidateSliceOpList);
    }
    break;
  }
  return std::make_pair(rootSource, candidateSliceOpList);
}

static FailureOr<tensor::ExtractSliceOp>
getFirstExtractSliceOpOfOperand(OpOperand &operand) {
  if (auto iterArg = dyn_cast<BlockArgument>(operand.get())) {
    if (auto loop =
            dyn_cast<LoopLikeOpInterface>(iterArg.getOwner()->getParentOp())) {
      return getFirstExtractSliceOpOfOperand(*loop.getTiedLoopInit(iterArg));
    }
    return failure();
  }

  Operation *defineOp = operand.get().getDefiningOp();
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defineOp)) {
    return sliceOp;
  } else if (isa<linalg::FillOp, tensor::ExpandShapeOp,
                 tensor::CollapseShapeOp>(defineOp)) {
    return getFirstExtractSliceOpOfOperand(defineOp->getOpOperand(0));
  }
  return failure();
}

/**
 * Find the untiled Producer op based on given OpOperand of Tiled Op, E.g.
 *
 * %1 = op1(...)
 * %2 = scf.for() {
 *  %3 = extract_slice %1 ...
 *  %4 = scf.for() {
 *     %5 = extract_slice %3 ...
 *     %t2 = tiledOp2(%5, ...)
 *  }
 * }
 *
 * @param operandOfTiledOp: %5
 * @return ProducerFusionAnchor including:
 *           1. fusableProducer:  %1 = op1(...)
 *           2. fusableResult: %1
 *           3. candidateSliceOpList： %5 = extract_slice %3 and
 *                                     %3 = extract_slice %1
 */
static FailureOr<ProducerFusionAnchor>
getProducerFusionAnchorFromOpOperand(RewriterBase &rewriter,
                                     OpOperand &operandOfTiledOp) {
  FailureOr<tensor::ExtractSliceOp> sliceOp =
      getFirstExtractSliceOpOfOperand(operandOfTiledOp);
  if (failed(sliceOp))
    return failure();

  auto resultAndSliceOpsPair = getRootSourceOfExtractSliceOp(*sliceOp);
  if (failed(resultAndSliceOpsPair))
    return failure();

  OpResult resultOfProducer =
      dyn_cast<OpResult>((*resultAndSliceOpsPair).first);
  // If producer is tilable
  if (isa<TilingInterface>(resultOfProducer.getOwner())) {
    return ProducerFusionAnchor(rewriter, resultOfProducer.getOwner(),
                                resultOfProducer,
                                (*resultAndSliceOpsPair).second);
  }

  return failure();
}

// Get the Result of top-level Loop which yield the target InsertSliceOp. E.g
// ```
// %1 = scf.for
//  %2 = scf.for
//   %3 = scf.for
//      ...
//      %4 = insert
//      yield %4
//   %5 = insert %3
//   yield %5
//  yield %2
// ```
// @param targetSliceOp: %4 = insert
// @return Result Value: %1
//         Collected insertSliceOp List during walk including targetSliceOp:
//                %4 = insert and %5 = insert %3
static FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
getResultOfTopLevelLoopYieldInsertSliceOp(
    OffsetSizeAndStrideOpInterface targetSliceOp, int curDepth = 0) {
  // control recursive time in avoid of stack overflow
  if (curDepth > MAX_DEPTH)
    return failure();

  SmallVector<OffsetSizeAndStrideOpInterface> candidateSliceOpList;
  candidateSliceOpList.push_back(targetSliceOp);
  Value resultOfLoop;
  if (auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(
          targetSliceOp.getOperation())) {
    Value destValue = sliceOp.getDest();
    auto iterArg = cast<BlockArgument>(destValue);
    auto forallOp = dyn_cast<scf::ForallOp>(iterArg.getOwner()->getParentOp());
    if (!forallOp)
      return failure();
    resultOfLoop = forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));
  } else if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(
                 targetSliceOp.getOperation())) {
    Value resultValue = sliceOp.getResult();
    for (auto &useOperand : resultValue.getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(useOperand.getOwner())) {
        if (llvm::detail::isPresent(resultOfLoop))
          return failure();
        auto forOp = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        if (!forOp)
          return failure();
        resultOfLoop = forOp->getResult(useOperand.getOperandNumber());
      }
    }
  }

  if (!llvm::detail::isPresent(resultOfLoop))
    return failure();

  while (true) {
    bool walkThroughOuterLoop = false;
    for (auto &useOperand : resultOfLoop.getUses()) {
      if (auto sliceOp =
              dyn_cast<OffsetSizeAndStrideOpInterface>(useOperand.getOwner())) {
        auto resultAndSliceOpsPair =
            getResultOfTopLevelLoopYieldInsertSliceOp(sliceOp, curDepth + 1);
        if (failed(resultAndSliceOpsPair))
          return failure();
        candidateSliceOpList.append((*resultAndSliceOpsPair).second.begin(),
                                    (*resultAndSliceOpsPair).second.end());
        return std::make_pair((*resultAndSliceOpsPair).first,
                              candidateSliceOpList);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(useOperand.getOwner())) {
        // walk through outer loop
        auto forOp = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        if (!forOp)
          return failure();
        resultOfLoop = forOp->getResult(useOperand.getOperandNumber());
        walkThroughOuterLoop = true;
        break;
      }
    }
    if (!walkThroughOuterLoop)
      break;
  }
  return std::make_pair(resultOfLoop, candidateSliceOpList);
}

/**
 * Find the untiled Consumer op based on given OpResult of Tiled Op, E.g.
 *
 * %1 = scf.for
 *  %2 = scf.for
 *   %3 = scf.for
 *      ...
 *      %t1 = tiledOp1
 *      %4 = insert %t1
 *      yield %4
 *   %5 = insert %3
 *   yield %5
 *  yield %2
 * %6 = op2(%1)
 *
 * @param resultOfTiledOp: %t1
 * @return ConsumerFusionAnchor including:
 *           1. fusableConsumer:  %6 = op2(%1)
 *           2. fusableOperand
 *           3. candidateSliceOpList： %4 = insert %t1 and
 *                                     %5 = insert %3
 */
static FailureOr<SmallVector<ConsumerFusionAnchor>>
getConsumerFusionAnchorFromOpResult(RewriterBase &rewriter,
                                    OpResult resultOfTiledOp) {
  OffsetSizeAndStrideOpInterface sliceOp;
  for (auto &useOfResult : resultOfTiledOp.getUses()) {
    if (isa<tensor::InsertSliceOp>(useOfResult.getOwner()) ||
        isa<tensor::ParallelInsertSliceOp>(useOfResult.getOwner())) {
      if (llvm::detail::isPresent(sliceOp))
        return failure();
      sliceOp =
          dyn_cast<OffsetSizeAndStrideOpInterface>(useOfResult.getOwner());
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(useOfResult.getOwner())) {
      if (auto loop = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp())) {
        return getConsumerFusionAnchorFromOpResult(
            rewriter, loop->getResult(useOfResult.getOperandNumber()));
      }
    }
  }

  if (!llvm::detail::isPresent(sliceOp))
    return failure();

  auto resultAndSliceOpsPair =
      getResultOfTopLevelLoopYieldInsertSliceOp(sliceOp);
  if (failed(resultAndSliceOpsPair))
    return failure();

  // support multiple consumers
  SmallVector<ConsumerFusionAnchor> consumerAnchorList;
  for (auto &useOperand : (*resultAndSliceOpsPair).first.getUses()) {
    // If consumer is tilable
    if (isa<TilingInterface>(useOperand.getOwner())) {
      consumerAnchorList.push_back(
          ConsumerFusionAnchor(rewriter, useOperand.getOwner(), useOperand,
                               (*resultAndSliceOpsPair).second));
    }
  }

  if (consumerAnchorList.empty())
    return failure();
  else
    return consumerAnchorList;
}

static Operation *preOpFuseProducerOfOpOperand(
    RewriterBase &rewriter, Location loc, OpOperand &operand, SystemDesc desc,
    llvm::SmallDenseSet<Operation *> &alreadyTiledOps) {
  FailureOr<ProducerFusionAnchor> prodAnchor =
      getProducerFusionAnchorFromOpOperand(rewriter, operand);
  if (failed(prodAnchor))
    return nullptr;

  if (alreadyTiledOps.count((*prodAnchor).getFusableOp()))
    return nullptr;

  FailureOr<tensor::ExtractSliceOp> candidateSliceOp =
      (*prodAnchor).selectCandidateByCostModel(rewriter, loc, desc);
  if (failed(candidateSliceOp)) {
    return nullptr;
  }
  std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
      scfX::tileAndFuseProducerOfSlice(rewriter, *candidateSliceOp);

  if (!fusedResult)
    return nullptr;

  // return tilable op
  return fusedResult.value().tiledOps[0];
}

static SmallVector<Operation *> postOpFuseConsumerOfOpResult(
    RewriterBase &rewriter, Location loc, OpResult result, SystemDesc desc,
    llvm::SmallDenseSet<Operation *> &alreadyTiledOps) {
  SmallVector<Operation *> tiledConsumerList;
  FailureOr<SmallVector<ConsumerFusionAnchor>> consAnchorList =
      getConsumerFusionAnchorFromOpResult(rewriter, result);
  if (failed(consAnchorList))
    return tiledConsumerList;

  for (auto &consAnchor : *consAnchorList) {
    if (alreadyTiledOps.count(consAnchor.getFusableOp()))
      continue;

    FailureOr<OffsetSizeAndStrideOpInterface> candidateSliceOp =
        consAnchor.selectCandidateByCostModel(rewriter, loc, desc);
    if (failed(candidateSliceOp))
      continue;

    std::optional<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        scfX::tileAndFuseConsumerOfSlice(rewriter, *candidateSliceOp);
    if (fusedResult) {
      auto tiledOp = fusedResult.value().tiledOps[0];
      tiledConsumerList.push_back(tiledOp);
      auto whileProducerOutOfBlock =
          [&tiledOp](LoopLikeOpInterface loop) -> LogicalResult {
        Block &body = loop->getRegion(0).front();
        return (tiledOp->getBlock() == &body) ? failure() : success();
      };
      SmallVector<LoopLikeOpInterface> outerLoops =
          scfX::getOuterNestLoopsWhile(
              (*candidateSliceOp)->getParentOfType<LoopLikeOpInterface>(),
              whileProducerOutOfBlock);
      // Manually run cse on region which contains top-level loop of candidate
      // slice in avoid of conflict with subsequent `tileAndFuseConsumerOfSlice`
      // get nest loops between next candidate sliceOp and tiled producer.
      auto region = outerLoops.front()->getParentRegion();
      (void)mlir::eraseUnreachableBlocks(rewriter, {*region});
      (void)mlir::runRegionDCE(rewriter, {*region});
    }
  }

  // return tilable op list
  return tiledConsumerList;
}

/**
 * Target at following general topology:
 *
 * producer1   producer2
 *   \          /
 *     anchor_op
 *   /          \
 * consumer1  consumer2
 *
 * where:
 *
 * 1. anchor op is responsible for providing scheduled parallel loops and
 * several FusionAnchor including both Producer and Consumer.
 * 2. support both pre-op and post-op fusion: try to fuse all of producers and
 * consumers of anchor op
 * 3. recursively call diffusion on either fused producer or consumer op based
 * on BFS.
 */
void diffusion(RewriterBase &rewriter, Location loc, Operation *anchorOp,
               SystemDesc desc) {
  llvm::SmallDenseSet<Operation *> alreadyTiledOps;
  std::deque<Operation *> anchorOpList = {anchorOp};

  while (!anchorOpList.empty()) {
    anchorOp = anchorOpList.front();
    anchorOpList.pop_front();
    // pre-op fuse
    for (OpOperand &operand : anchorOp->getOpOperands()) {
      if (auto tiledOp = preOpFuseProducerOfOpOperand(rewriter, loc, operand,
                                                      desc, alreadyTiledOps)) {
        alreadyTiledOps.insert(tiledOp);
        anchorOpList.push_back(tiledOp);
      }
    }
    // post-op fuse
    for (OpResult result : anchorOp->getResults()) {
      auto tiledOpList = postOpFuseConsumerOfOpResult(rewriter, loc, result,
                                                      desc, alreadyTiledOps);
      for (auto &tiledOp : tiledOpList) {
        alreadyTiledOps.insert(tiledOp);
        anchorOpList.push_back(tiledOp);
      }
    }
  }
}

/**
 * What is Anchor Op?
 * 1. located in a for loop
 * 2. it is the only one TilingInterface op in for loop
 * 3. has extract/insert slice
 *
 * E.g.
 * %1 = scf.for(){
 *   %2 = scf.for(){
 *       %3 = extract_slice
 *       %4 = anchor_op(%3)
 *       %5 = insert %4
 *       yield %5
 *   }
 * }
 *
 * */
static LogicalResult isAnchorOp(Operation *targetOp) {
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

static bool TilingAndFusionBasedOnAnchorOp(RewriterBase &rewriter, Location loc,
                                           func::FuncOp f, SystemDesc desc) {
  SmallVector<Operation *> anchorOpList;
  // Walk through func operation.
  f->walk([&anchorOpList](Operation *op) {
    if (succeeded(isAnchorOp(op))) {
      anchorOpList.push_back(op);
    }
  });

  for (auto &anchorOp : anchorOpList) {
    diffusion(rewriter, f->getLoc(), anchorOp, desc);
  }

  // Return whether need repartition
  return true;
}

static void FineGrainedFusion(RewriterBase &rewriter, Location loc,
                              func::FuncOp f, SystemDesc desc) {
  // Target at anchor op, like matmul/conv, and try to fuse any tilable
  // operation around it by diffusion
  TilingAndFusionBasedOnAnchorOp(rewriter, loc, f, desc);
}

struct AnyTilableFusion : public impl::AnyTilableFusionBase<AnyTilableFusion> {

public:
  void runOnOperation() final {
    auto &ctx = getContext();

    func::FuncOp func = getOperation();
    Location loc = func.getLoc();
    IRRewriter rewriter(&ctx);
    /// TODO: fetch from somewhere else
    SystemDesc desc;
    FineGrainedFusion(rewriter, loc, func, desc);

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