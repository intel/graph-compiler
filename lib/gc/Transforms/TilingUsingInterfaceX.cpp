//===- Implementation of tiling using TilingInterface -------===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

#include "TilingUsingInterfaceX.h"

#define DEBUG_TYPE "tile-using-interface-x"

using namespace mlir;

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

static std::tuple<OpResult, std::optional<OpOperand *>>
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
  // recursively find
  if (auto newExtractOp =
          source->get().getDefiningOp<tensor::ExtractSliceOp>()) {
    return getUntiledProducerFromSliceSource(
        &newExtractOp.getSourceMutable(),
        loops.drop_back(loopIt - loops.rbegin()));
  }
  if (loopIt == loops.rend())
    destinationIterArg = source;
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
}

std::optional<scf::SCFFuseProducerOfSliceResult>
mlir::scfX::tileAndFuseProducerOfSlice(
    RewriterBase &rewriter, tensor::ExtractSliceOp candidateSliceOp,
    MutableArrayRef<LoopLikeOpInterface> loops) {
  // 1. Get the producer of the source (potentially walking through
  // `iter_args` of nested `scf.for`)
  auto [fusableProducer, destinationInitArg] =
      getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                        loops);
  if (!fusableProducer)
    return std::nullopt;
  unsigned resultNumber = fusableProducer.getResultNumber();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(candidateSliceOp);

  // 2. Clone the fused producer
  // 2a. Compute the destination operands to use for the cloned operation.
  SmallVector<Value> origDestinationTensors, clonedOpDestinationTensors;
  Operation *fusableProducerOp = fusableProducer.getOwner();
  if (isa<DestinationStyleOpInterface>(fusableProducerOp) &&
      failed(tensor::getOrCreateDestinations(
          rewriter, fusableProducerOp->getLoc(), fusableProducerOp,
          origDestinationTensors)))
    return std::nullopt;

  clonedOpDestinationTensors = origDestinationTensors;
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp)) {
    // 2b. If the producer is also destination style, then to maintain the
    // destination passing style, update the destination of the producer to be
    // the source of the slice.
    clonedOpDestinationTensors[resultNumber] = candidateSliceOp.getSource();
  }
  // 2c. Clone the fused producer.
  Operation *clonedProducerOp = cloneOpAndUpdateDestinationArgs(
      rewriter, fusableProducerOp, clonedOpDestinationTensors);
  // 2d. Update the source of the candidateSlice to be the cloned producer.
  //     Easier to just clone the slice with different source since replacements
  //     and DCE of cloned ops becomes easier
  SmallVector<Value> candidateSliceOpOperands =
      llvm::to_vector(candidateSliceOp->getOperands());
  candidateSliceOpOperands[0] = clonedProducerOp->getResult(resultNumber);
  tensor::ExtractSliceOp clonedCandidateSliceOp =
      mlir::clone(rewriter, candidateSliceOp,
                  candidateSliceOp->getResultTypes(), candidateSliceOpOperands);

  // 3. Generate the tiled implementation of the producer of the source
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceExtractSliceWithTiledProducer(
          rewriter, clonedCandidateSliceOp,
          clonedProducerOp->getResult(resultNumber));
  if (failed(tileAndFuseResult))
    return std::nullopt;
  // Note: Do not delete the candidateSliceOp, since its passed in from the
  // caller.
  rewriter.replaceAllUsesWith(candidateSliceOp,
                              tileAndFuseResult->tiledValues[0]);
  rewriter.eraseOp(clonedCandidateSliceOp);
  rewriter.eraseOp(clonedProducerOp);

  // 3. If the slice is for a destination operand, for example,
  //
  // ```mlir
  // %0 = linalg.init
  // %1 = linalg.fill .. outs(%0 : )
  // %2 = scf.for .. iter_args(%arg0 = %1) {
  //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %4 = tensor.extract_slice %arg1 [..]
  //     .. = linalg.matmul .. outs(%4 : )
  //   }
  // }
  // ```
  //
  // the IR is currently
  //
  // ```
  // %0 = linalg.init
  // %1 = linalg.fill
  // %2 = scf.for .. iter_args(%arg0 = %1 /* incorrect value */ ) {
  //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %4 = tensor.extract_slice %arg1[..]
  //     %5 = linalg.fill .. outs(%4 : )
  //     .. = linalg.matmul .. outs(%5 : )
  //   }
  // }
  // ```
  //
  // The untiled `linalg.fill` is still used as the `init_value` since it
  // was originally a destination operand of the untiled `linalg.matmul`.
  // When fusing an operand that is a destination operand, the iter_arg of
  // the outer most loop should be changed to use the destination of the
  // fused operation. With this the IR will be.
  //
  // ```
  // %0 = linalg.init
  // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
  //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %3 = tensor.extract_slice %arg1[..]
  //     %4 = linalg.fill .. outs(%3 : )
  //     .. = linalg.matmul .. outs(%4 : )
  //   }
  // }
  // ```
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp) && !loops.empty()) {
    loops.front()
        ->getOpOperands()[destinationInitArg.value()->getOperandNumber()]
        .set(origDestinationTensors[resultNumber]);
  }
  return scf::SCFFuseProducerOfSliceResult{fusableProducer,
                                           tileAndFuseResult->tiledValues[0],
                                           tileAndFuseResult->tiledOps};
}

/// This utility currently checks whether the loop either :-
/// 1. Yields exactly one result.
/// 2. Has consumer op as its first user and other users to be in the same
/// containing block as that of consumer op's. Currently we clone the loop op
/// right before the consumer op in order to maintain a valid def-use chain.
/// This utility thus helps ensuring that no invalid IR is formed due to the
/// same.
static LogicalResult checkAssumptionForLoop(Operation *loopOp,
                                            Operation *consumerOp) {
  // Check if the loop op yields one result.
  if (loopOp->getNumResults() == 1)
    return success();
  // Check if the consumerOp is the first user of the loopOp and if other users
  // are in the same containing block as that of consumer op's.
  Block *parentBlock = consumerOp->getBlock();
  for (Operation *userOp : loopOp->getUsers()) {
    if (userOp == consumerOp)
      continue;
    if (parentBlock != userOp->getBlock() ||
        !consumerOp->isBeforeInBlock(userOp)) {
      return failure();
    }
  }
  return success();
}

SmallVector<LoopLikeOpInterface>
scfX::getOuterLoopsOfSliceOp(OffsetSizeAndStrideOpInterface sliceOp) {
  SmallVector<LoopLikeOpInterface> outerLoops;
  auto forOp = sliceOp->getParentOfType<LoopLikeOpInterface>();
  while (forOp) {
    outerLoops.push_back(forOp);
    forOp = forOp->getParentOfType<LoopLikeOpInterface>();
  }
  return {outerLoops.rbegin(), outerLoops.rend()};
}

// maximum recursive time
#define MAX_DEPTH 5

FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
scfX::getResultOfTopLevelLoopYieldInsertSliceOp(
    OffsetSizeAndStrideOpInterface targetSliceOp, int curDepth) {
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

static void
fixTerminatorSCFYield(RewriterBase &rewriter, scf::ForOp newForOp,
                      ResultRange tilingResult,
                      SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
                      SmallVector<SmallVector<OpFoldResult>> &resultSizes,
                      ArrayRef<BlockArgument> bbArgs) {
  scf::YieldOp oldTerminatorOp =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  unsigned totalOldResults = oldTerminatorOp->getNumResults();
  unsigned totalTiledResults = tilingResult.size();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(totalOldResults + totalTiledResults);
  for (auto oldResult : oldTerminatorOp.getResults()) {
    newYieldOperands.push_back(oldResult);
  }
  rewriter.setInsertionPointAfter(oldTerminatorOp);
  Location loc = newForOp.getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult, bbArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    Value newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, tiledResult, bbArg, resultOffset, resultSize, strides);
    newYieldOperands.push_back(newInsertSliceOp);
  }
  rewriter.create<scf::YieldOp>(loc, newYieldOperands);
  rewriter.eraseOp(oldTerminatorOp);
}

static void fixTerminatorSCFInParallel(
    RewriterBase &rewriter, scf::ForallOp newForallOp, ResultRange tilingResult,
    SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &resultSizes,
    ArrayRef<BlockArgument> bbArgs) {
  scf::InParallelOp newTerminatorOp = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(newTerminatorOp.getBody());
  Location firstYieldOpLoc =
      (*(newTerminatorOp.getYieldingOps().begin())).getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult, bbArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        firstYieldOpLoc, tiledResult, bbArg, resultOffset, resultSize, strides);
  }
}

static SmallVector<Value>
fixSharedOutSCFForall(RewriterBase &rewriter, scf::ForallOp outerLoop,
                      LoopLikeOpInterface innerLoop,
                      SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
                      SmallVector<SmallVector<OpFoldResult>> &resultSizes,
                      unsigned newInitSize) {
  SmallVector<Value> newInitAppend;
  rewriter.setInsertionPoint(innerLoop);
  Location Loc = outerLoop.getLoc();
  MutableArrayRef<BlockArgument> bbArgs = outerLoop.getBody()->getArguments();
  for (auto [bbArg, offset, sizes] : llvm::zip_equal(
           bbArgs.take_back(newInitSize), resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(offset.size(), rewriter.getIndexAttr(1));
    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        Loc, bbArg, offset, sizes, strides);
    newInitAppend.push_back(newExtractOp.getResult());
  }
  return newInitAppend;
}

static LogicalResult computeAllResultTileForOpGivenOperandSliceOp(
    RewriterBase &rewriter, TilingInterface tilableOp, unsigned operandNumber,
    OffsetSizeAndStrideOpInterface ossSliceOp,
    SmallVector<SmallVector<OpFoldResult>> &allResultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &allResultSizes) {
  // 1. check all stride all 1
  if (llvm::any_of(ossSliceOp.getMixedStrides(), [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(ossSliceOp, "ossSliceOp has stride");
  }
  // 2. compute iteration domain Tile from input position
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  if (failed(tilableOp.getIterationDomainTileFromOperandTile(
          rewriter, operandNumber, ossSliceOp.getMixedOffsets(),
          ossSliceOp.getMixedSizes(), iterDomainOffsets, iterDomainSizes))) {
    return rewriter.notifyMatchFailure(
        tilableOp, "can't get iter domain position from input position");
  }
  unsigned totalNumResultsOfConsumer = tilableOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(
      totalNumResultsOfConsumer);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(totalNumResultsOfConsumer);
  // 3. compute result Tile by resultNumber
  for (auto [idx, v] : llvm::enumerate(tilableOp->getResults())) {
    if (failed(tilableOp.getResultTilePosition(
            rewriter, idx, iterDomainOffsets, iterDomainSizes,
            resultOffsets[idx], resultSizes[idx]))) {
      return rewriter.notifyMatchFailure(
          tilableOp,
          "can't get result domain position from iter domain position");
    }
  }
  allResultOffsets = resultOffsets;
  allResultSizes = resultSizes;
  return success();
}

/// Implementation of fusing consumer of a single slice by computing the
/// slice of the consumer in-place for scf loop.
FailureOr<scf::SCFFuseConsumerOfSliceResult>
mlir::scfX::tileAndFuseConsumerOfSlice(RewriterBase &rewriter,
                                       Operation *candidateSliceOp) {
  if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
          candidateSliceOp))
    return failure();

  bool isInsertSliceOp = isa<tensor::InsertSliceOp>(candidateSliceOp);

  // 1. Get the real consumer of candidate
  // tensor.insert_slice/parallel_insert_slice by walking through
  // scf.for/scf.forall and collect all [Parallel]insertSliceOp(s) along the way
  FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
      resultAndSliceOpsPair = scfX::getResultOfTopLevelLoopYieldInsertSliceOp(
          cast<OffsetSizeAndStrideOpInterface>(candidateSliceOp));
  if (failed(resultAndSliceOpsPair)) {
    return rewriter.notifyMatchFailure(candidateSliceOp,
                                       "could not fetch consumer to fuse");
  }

  OpOperand &consumerOpOperand =
      *((*resultAndSliceOpsPair).first.getUses().begin());
  Operation *consumerOp = consumerOpOperand.getOwner();
  unsigned operandNumber = consumerOpOperand.getOperandNumber();
  unsigned resultNumber = 0;
  if (auto producerResult = dyn_cast<OpResult>(consumerOpOperand.get())) {
    resultNumber = producerResult.getResultNumber();
  } else {
    return rewriter.notifyMatchFailure(
        consumerOp, "consumer op's operand doesn't seem to be an OpResult");
  }

  // 2. Get all outer loops of candidateSliceOp
  SmallVector<LoopLikeOpInterface> outerLoops = getOuterLoopsOfSliceOp(
      cast<OffsetSizeAndStrideOpInterface>(candidateSliceOp));

  LoopLikeOpInterface outerMostLoop = outerLoops.front();
  if (failed(checkAssumptionForLoop(outerMostLoop, consumerOp))) {
    return rewriter.notifyMatchFailure(
        outerMostLoop,
        "containing loop op should either yield just one value or "
        "have the consumer op as its first user");
  }

  // 3. Check consumer is not using outerMostLoop's output as init.
  auto dstOp = cast<DestinationStyleOpInterface>(consumerOp);
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, outerMostLoop->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }
  ValueRange newInitAppend = dpsInits;

  // 4. reconstruct nested loop from outer to inner
  SmallVector<OffsetSizeAndStrideOpInterface> candidateSliceOpList =
      (*resultAndSliceOpsPair).second;
  SmallVector<LoopLikeOpInterface> newOuterLoops;
  SmallVector<SmallVector<OpFoldResult>> allResultOffsets, allResultSizes;

  Block *oldLoopBody = nullptr;
  Block *newLoopBody = nullptr;
  SmallVector<Value> newOuts;

  OpBuilder::InsertionGuard g(rewriter);

  for (auto [index, loop] :
       llvm::enumerate(MutableArrayRef(outerLoops).drop_back())) {
    if (index == 0)
      rewriter.setInsertionPoint(consumerOp);
    else
      rewriter.setInsertionPoint(loop);

    LoopLikeOpInterface newLoopOp;
    // Create a new loop with the new init values for this loop.
    if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
      newOuts = llvm::to_vector(forOp.getInits());
      newOuts.append(newInitAppend.begin(), newInitAppend.end());
      auto newLoop = rewriter.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newOuts);
      newLoopOp = newLoop;
      oldLoopBody = forOp.getBody();
      newLoopBody = newLoop.getBody();
      newInitAppend =
          newLoopBody->getArguments().take_back(newInitAppend.size());
    } else if (auto forallOp = dyn_cast<scf::ForallOp>(loop.getOperation())) {
      newOuts = llvm::to_vector(forallOp.getOutputs());
      newOuts.append(newInitAppend.begin(), newInitAppend.end());
      auto newLoop = rewriter.create<scf::ForallOp>(
          forallOp.getLoc(), forallOp.getMixedLowerBound(),
          forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOuts,
          forallOp.getMapping());
      rewriter.eraseOp(newLoop.getTerminator());
      newLoopOp = newLoop;
      oldLoopBody = forallOp.getBody();
      newLoopBody = newLoop.getBody();

      // create extractSliceOp for newInits
      assert(index == 0 && "Currently Only outerMostLoop assumed ForallOp");
      auto outerMostCandidate = candidateSliceOpList.back();
      assert(isa<tensor::ParallelInsertSliceOp>(outerMostCandidate));
      // set InsertPoint before next inner loop
      auto nextLoop = outerLoops[index + 1];
      rewriter.setInsertionPoint(nextLoop);
      if (failed(computeAllResultTileForOpGivenOperandSliceOp(
              rewriter, cast<TilingInterface>(consumerOp), operandNumber,
              outerMostCandidate, allResultOffsets, allResultSizes))) {
        return failure();
      }
      newInitAppend =
          fixSharedOutSCFForall(rewriter, forallOp, nextLoop, allResultOffsets,
                                allResultSizes, newInitAppend.size());
    }
    rewriter.mergeBlocks(
        oldLoopBody, newLoopBody,
        newLoopBody->getArguments().take_front(oldLoopBody->getNumArguments()));
    rewriter.replaceOp(
        loop, newLoopOp->getResults().take_front(loop->getNumResults()));
    newOuterLoops.push_back(newLoopOp);
  }

  // 5.a reconstruct inner-most loop
  LoopLikeOpInterface oldInnerMostLoop = outerLoops.back(), newInnerMostLoop;
  Location loc = oldInnerMostLoop->getLoc();
  rewriter.setInsertionPoint(oldInnerMostLoop);
  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldInnerMostLoop.getOperation());
    newOuts = llvm::to_vector(forOp.getInits());
    newOuts.append(newInitAppend.begin(), newInitAppend.end());
    oldLoopBody = forOp.getBody();
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newOuts);
    newInnerMostLoop = newForOp;
    newLoopBody = newForOp.getBody();
  } else {
    auto forallOp = cast<scf::ForallOp>(oldInnerMostLoop.getOperation());
    newOuts = llvm::to_vector(forallOp.getOutputs());
    newOuts.append(newInitAppend.begin(), newInitAppend.end());
    oldLoopBody = forallOp.getBody();
    auto newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), newOuts, forallOp.getMapping());
    newInnerMostLoop = newForallOp;
    rewriter.eraseOp(newForallOp.getTerminator());
    newLoopBody = newForallOp.getBody();
  }

  // 5.b Move the loop body to the new op.
  unsigned oldNumArguments = oldLoopBody->getNumArguments();
  rewriter.mergeBlocks(oldLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(oldNumArguments));
  // 5.c replace the result of old oldInnerMostLoop with newInnerMostLoop's
  // results
  rewriter.replaceOp(oldInnerMostLoop,
                     newInnerMostLoop->getResults().take_front(
                         oldInnerMostLoop->getNumResults()));

  // 6. Set insertion point before terminator op of the loop and create a new
  // tensor.insert_slice. In the scf.for case this is a clone of the
  // candidateSliceOp whereas in the scf.forall case this is created from the
  // operands of tensor.parallel_insert_slice.
  tensor::InsertSliceOp clonedInsertSliceOp;
  // we need to compute real offset and size for multi-level insertSliceOp
  // according the candidateSliceOpList
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    auto newForallOp = cast<scf::ForallOp>(newInnerMostLoop);
    rewriter.setInsertionPoint(newForallOp.getTerminator());
    clonedInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, sliceOp.getSource(), sliceOp.getDest(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  } else {
    rewriter.setInsertionPoint(candidateSliceOp);
    clonedInsertSliceOp =
        cast<tensor::InsertSliceOp>(rewriter.clone(*candidateSliceOp));
  }

  // 7.a. Clone consumer op.
  auto newForOpBlockArgsForConsumerDest =
      newLoopBody->getArguments().drop_front(oldNumArguments);
  auto clonedConsumerOp = cast<TilingInterface>(cloneOpAndUpdateDestinationArgs(
      rewriter, consumerOp, newForOpBlockArgsForConsumerDest));

  // 7.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    operandToReplace.set(clonedInsertSliceOp.getResult());
  });

  // 8 - Perform tiling of the cloned consumer and replace the operand at
  // `operandNumber` with the source of the cloned tensor.insert_slice op.
  auto ossSliceOp =
      cast<OffsetSizeAndStrideOpInterface>(clonedInsertSliceOp.getOperation());
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter, ossSliceOp, clonedConsumerOp->getOpOperand(operandNumber));
  if (failed(tileAndFuseResult)) {
    return failure();
  }
  rewriter.replaceAllUsesWith(
      tileAndFuseResult->tiledOps[0]->getOperand(operandNumber),
      clonedInsertSliceOp.getSource());

  // 9. Try to fetch the offset and size for all results of the cloned
  // consumer. This would then be used to form the corresponding
  // tensor.insert_slice/parallel_insert_slice later.
  if (failed(computeAllResultTileForOpGivenOperandSliceOp(
          rewriter, clonedConsumerOp, operandNumber,
          cast<OffsetSizeAndStrideOpInterface>(candidateSliceOp),
          allResultOffsets, allResultSizes))) {
    return failure();
  }

  if (isInsertSliceOp) {
    auto newForOp = cast<scf::ForOp>(newInnerMostLoop);
    fixTerminatorSCFYield(
        rewriter, newForOp, tileAndFuseResult->tiledOps[0]->getResults(),
        allResultOffsets, allResultSizes,
        newForOp.getBody()->getArguments().take_back(newInitAppend.size()));
  } else {
    auto newForallOp = cast<scf::ForallOp>(newInnerMostLoop);
    fixTerminatorSCFInParallel(
        rewriter, newForallOp, tileAndFuseResult->tiledOps[0]->getResults(),
        allResultOffsets, allResultSizes,
        newForallOp.getBody()->getArguments().take_back(newInitAppend.size()));
  }

  newOuterLoops.push_back(cast<LoopLikeOpInterface>(newInnerMostLoop));

  // 10. reconstruct terminator of outer loop by inner loop
  auto outerCandidateIter = candidateSliceOpList.rbegin();
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(MutableArrayRef(newOuterLoops).drop_back(),
                       MutableArrayRef(newOuterLoops).drop_front())) {
    // create insertSliceOp according outer candidateSliceOp
    if (outerCandidateIter != candidateSliceOpList.rend() &&
        outerCandidateIter->getOperation()
                ->getParentOfType<LoopLikeOpInterface>() == outerLoop) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(outerLoop.getOperation())) {
        rewriter.setInsertionPoint(forallOp.getTerminator());
      } else {
        rewriter.setInsertionPointAfter(*outerCandidateIter);
      }

      if (failed(computeAllResultTileForOpGivenOperandSliceOp(
              rewriter, clonedConsumerOp, operandNumber, *outerCandidateIter,
              allResultOffsets, allResultSizes))) {
        return failure();
      }

      if (auto forOp = dyn_cast<scf::ForOp>(outerLoop.getOperation())) {
        fixTerminatorSCFYield(
            rewriter, forOp,
            innerLoop->getResults().take_back(newInitAppend.size()),
            allResultOffsets, allResultSizes,
            forOp.getBody()->getArguments().take_back(newInitAppend.size()));
      } else if (auto forallOp =
                     dyn_cast<scf::ForallOp>(outerLoop.getOperation())) {
        fixTerminatorSCFInParallel(
            rewriter, forallOp,
            innerLoop->getResults().take_back(newInitAppend.size()),
            allResultOffsets, allResultSizes,
            forallOp.getBody()->getArguments().take_back(newInitAppend.size()));
      }
      outerCandidateIter++;
    } else {
      // yield additional new appended results of innerLoop
      assert(isa<scf::ForOp>(outerLoop));
      auto forOp = cast<scf::ForOp>(outerLoop);
      auto outerLoopYield =
          cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      SmallVector<Value> newYields =
          llvm::to_vector(outerLoopYield.getOperands());
      ValueRange additionalYields =
          innerLoop->getResults().take_back(newInitAppend.size());
      newYields.append(additionalYields.begin(), additionalYields.end());
      rewriter.setInsertionPoint(outerLoopYield);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(outerLoopYield, newYields);
    }
  }

  // 11. Replace the result of consumer op with new outerMost loop's
  // results.
  for (auto &&[oldResult, newResult] :
       llvm::zip(consumerOp->getResults(),
                 newOuterLoops.front()->getResults().take_back(
                     newInitAppend.size()))) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // 12. Need to erase the cloned consumer op.
  rewriter.eraseOp(clonedConsumerOp);

  return scf::SCFFuseConsumerOfSliceResult{
      &consumerOpOperand,
      &(tileAndFuseResult->tiledOps[0]->getOpOperand(operandNumber)),
      tileAndFuseResult->tiledOps};
}
