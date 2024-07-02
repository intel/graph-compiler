//===-- TilingUsingInterfaceX.cpp -  upstream eXtension ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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

/// Get the result of top-level loop which yields the target InsertSliceOp. E.g
/// ```
/// %1 = scf.for
///  %2 = scf.for
///   %3 = scf.for
///      ...
///      %4 = insert
///      yield %4
///   %5 = insert %3
///   yield %5
///  yield %2
/// ```
/// @param targetSliceOp: %4 = insert
/// @param insertSliceOpChain: chain of all related insert sliceOp
/// @return resultValue: %1
static FailureOr<Value> getResultOfTopLevelLoopYieldInsertSliceOp(
    Operation *targetSliceOp,
    SmallVectorImpl<OffsetSizeAndStrideOpInterface> &insertSliceOpChain,
    int curDepth = 0, int maxDepth = 5) {
  assert(isa<OffsetSizeAndStrideOpInterface>(targetSliceOp));
  // Control recursive time in avoid of stack overflow
  if (curDepth > maxDepth)
    return failure();

  insertSliceOpChain.push_back(
      cast<OffsetSizeAndStrideOpInterface>(targetSliceOp));
  Value resultOfLoop;
  if (auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(targetSliceOp)) {
    Value destValue = sliceOp.getDest();
    auto iterArg = cast<BlockArgument>(destValue);
    auto forallOp = dyn_cast<scf::ForallOp>(iterArg.getOwner()->getParentOp());
    if (!forallOp)
      return failure();
    resultOfLoop = forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));
  } else if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(targetSliceOp)) {
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
    for (OpOperand &useOperand : resultOfLoop.getUses()) {
      if (auto sliceOp =
              dyn_cast<OffsetSizeAndStrideOpInterface>(useOperand.getOwner())) {
        return getResultOfTopLevelLoopYieldInsertSliceOp(
            sliceOp, insertSliceOpChain, curDepth + 1);
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
  return resultOfLoop;
}

/// A utility function that checks whether the only use of the result of a
/// tensor.insert_slice op is in a scf.yield op.
static LogicalResult
checkAssumptionForFusingConsumer(tensor::InsertSliceOp candidateSliceOp) {
  Value result = candidateSliceOp.getResult();
  Value::use_range uses = result.getUses();
  if (!llvm::hasSingleElement(uses)) {
    LLVM_DEBUG(llvm::dbgs() << "Too many uses of the candidate slice op\n");
    return failure();
  }
  OpOperand &operandUse = (*uses.begin());
  Operation *userOp = operandUse.getOwner();
  if (!isa<scf::YieldOp>(userOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected scf.yield to be the only user, but got -> "
               << (*userOp));
    return failure();
  }
  if (result.getDefiningOp()->getBlock() != userOp->getBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Expected tensor.insert_slice and scf.yield to "
                               "be in the same block\n");
    return failure();
  }
  return success();
}

/// Fetches the OpOperand of the only user (and use) of the value `val` which
/// implements `TilingInterface` and `DestinationStyleOpInterface`. Returns
/// failure otherwise.
static FailureOr<OpOperand *> getConsumerFromUses(Value val,
                                                  Block *containingOpBlock) {
  // Step 1. Check that the value has exactly one use except for scf.yield.
  OpOperand *operand = nullptr;
  for (auto &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (isa<tensor::InsertSliceOp>(user) ||
        isa<tensor::ParallelInsertSliceOp>(user))
      continue;
    else {
      if (operand)
        return failure();
      else
        operand = &use;
    }
  }
  if (!operand)
    return failure();
  // Step 2. Get uses.
  Operation *consumerOp = operand->getOwner();
  // TODO: We have to init result of consumer before scf.for, use
  //       DestinationStyleOpInterface to get result shape from init for now.
  //       Add support for other op such as op has InferTypeOpInterface.
  if (!isa<TilingInterface>(consumerOp) ||
      !isa<DestinationStyleOpInterface>(consumerOp))
    return failure();
  if (containingOpBlock != consumerOp->getBlock())
    return failure();
  return operand;
}

/// Return perfectly outer loops of given ForOp(included), sorted from
/// outer to inner.
static SmallVector<scf::ForOp> getPerfectlyOuterLoops(scf::ForOp loop) {
  SmallVector<scf::ForOp> outerLoops = {loop};
  auto forOp = loop->getParentOfType<scf::ForOp>();
  while (forOp) {
    Block &body = forOp.getRegion().front();
    if (body.begin() != std::prev(body.end(), 2))
      break;
    outerLoops.push_back(forOp);
    forOp = forOp->getParentOfType<scf::ForOp>();
  }
  return {outerLoops.rbegin(), outerLoops.rend()};
}

/// Fetch the untiled consumer of a scf.for's result which is yielded by a
/// tensor.insert_slice. This function makes the following assumptions :
/// 1.  tensor.insert_slice has scf.yield as its only user.
/// 2.  scf.for's corresponding result has only one use.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(tensor::InsertSliceOp candidateSliceOp) {
  if (failed(checkAssumptionForFusingConsumer(candidateSliceOp)))
    return failure();
  Value sliceResult = candidateSliceOp.getResult();
  // Step 1. Fetch the corresponding output.
  OpOperand &yieldOpOperand = (*sliceResult.getUses().begin());
  unsigned resultNumber = yieldOpOperand.getOperandNumber();
  // Step 2. Check containing op is scf.for.
  Operation *containingOp = candidateSliceOp->getParentOp();
  auto forOp = dyn_cast<scf::ForOp>(containingOp);
  if (!forOp)
    return failure();
  scf::ForOp topLevelForOp = getPerfectlyOuterLoops(forOp).front();
  Value resultingValue = topLevelForOp->getResult(resultNumber);

  return getConsumerFromUses(resultingValue, topLevelForOp->getBlock());
}

/// Fetch the first untiled consumer of a scf.forall's result which is yielded
/// by a tensor.parallel_insert_slice.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(tensor::ParallelInsertSliceOp candidateSliceOp) {
  // Step 1. Fetch the corresponding output
  Value sliceDest = candidateSliceOp.getDest();
  auto iterArg = dyn_cast<BlockArgument>(sliceDest);
  if (!iterArg)
    return failure();
  Operation *containingOp = iterArg.getOwner()->getParentOp();
  if (containingOp != candidateSliceOp->getParentOp()->getParentOp())
    return failure();
  // Step 2. Check that the containing op is scf.forall.
  auto forallOp = dyn_cast<scf::ForallOp>(containingOp);
  if (!forallOp)
    return failure();
  Value resultingValue =
      forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));

  return getConsumerFromUses(resultingValue, containingOp->getBlock());
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
        !consumerOp->isBeforeInBlock(userOp))
      return failure();
  }
  return success();
}

/// A utility to fetch an untiled consumer of
/// tensor.insert_slice/tensor.parallel_insert_slice.
static FailureOr<OpOperand *> getUntiledConsumerFromSlice(Operation *sliceOp) {
  if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(insertSlice);
  } else if (auto parallelInsertSlice =
                 dyn_cast<tensor::ParallelInsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(parallelInsertSlice);
  } else {
    return failure();
  }
}

/// After fusing consumer into scf.for we want to modify the scf.yield operation
/// to reflect the same by returning the values yielded by the tiled consumer.
static void
fixTerminatorSCFYield(RewriterBase &rewriter, scf::ForOp newForOp,
                      TilingResult &tilingResult,
                      ArrayRef<SmallVector<OpFoldResult>> &resultOffsets,
                      ArrayRef<SmallVector<OpFoldResult>> &resultSizes,
                      ArrayRef<BlockArgument> bbArgs) {
  scf::YieldOp oldTerminatorOp =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  unsigned totalOldResults = oldTerminatorOp->getNumResults();
  unsigned totalTiledResults = tilingResult.tiledOps[0]->getNumResults();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(totalOldResults + totalTiledResults);
  for (auto oldResult : oldTerminatorOp.getResults()) {
    newYieldOperands.push_back(oldResult);
  }
  rewriter.setInsertionPointAfter(oldTerminatorOp);
  Location loc = newForOp.getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult.tiledOps[0]->getResults(), bbArgs,
                       resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    Value newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, tiledResult, bbArg, resultOffset, resultSize, strides);
    newYieldOperands.push_back(newInsertSliceOp);
  }
  rewriter.create<scf::YieldOp>(loc, newYieldOperands);
  rewriter.eraseOp(oldTerminatorOp);
}

/// After fusing consumer into scf.forall we want to yield each of the resulting
/// values by the tiled consumer within scf.forall.in_parallel region.
static void
fixTerminatorSCFInParallel(RewriterBase &rewriter, scf::ForallOp newForallOp,
                           SmallVector<Value> tiledResults,
                           ArrayRef<SmallVector<OpFoldResult>> &resultOffsets,
                           ArrayRef<SmallVector<OpFoldResult>> &resultSizes,
                           ArrayRef<BlockArgument> bbArgs) {
  scf::InParallelOp newTerminatorOp = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(newTerminatorOp.getBody());
  Location firstYieldOpLoc =
      (*(newTerminatorOp.getYieldingOps().begin())).getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, bbArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        firstYieldOpLoc, tiledResult, bbArg, resultOffset, resultSize, strides);
  }
}

/// Implementation of fusing consumer of a single slice by computing the
/// slice of the consumer in-place for scf loop.
static FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerOfSliceImpl(RewriterBase &rewriter,
                               Operation *candidateSliceOp) {
  if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
          candidateSliceOp))
    return failure();

  bool isInsertSliceOp = isa<tensor::InsertSliceOp>(candidateSliceOp);

  // 1. Get the consumer of scf.for for the result yielded by
  // tensor.insert_slice/parallel_insert_slice.
  FailureOr<OpOperand *> maybeConsumerOpOperand =
      getUntiledConsumerFromSlice(candidateSliceOp);
  if (failed(maybeConsumerOpOperand)) {
    return rewriter.notifyMatchFailure(candidateSliceOp,
                                       "could not fetch consumer to fuse");
  }
  OpOperand *consumerOpOperand = *maybeConsumerOpOperand;
  Operation *consumerOp = consumerOpOperand->getOwner();
  unsigned operandNumber = consumerOpOperand->getOperandNumber();
  unsigned resultNumber = 0;
  if (auto producerResult = dyn_cast<OpResult>(consumerOpOperand->get())) {
    resultNumber = producerResult.getResultNumber();
  } else {
    return rewriter.notifyMatchFailure(
        consumerOp, "consumer op's operand doesn't seem to be an OpResult");
  }

  Operation *oldLoopOp = nullptr;
  SmallVector<Value> newOuts;
  Block *oldLoopBody = nullptr;
  unsigned initSize = 0;
  unsigned rank = 1;
  if (isInsertSliceOp) {
    auto forOp = candidateSliceOp->getParentOfType<scf::ForOp>();
    oldLoopOp = forOp;
    initSize = forOp.getInits().size();
  } else {
    auto forallOp = candidateSliceOp->getParentOfType<scf::ForallOp>();
    oldLoopOp = forallOp;
    initSize = forallOp.getOutputs().size();
    rank = forallOp.getRank();
  }

  Operation *oldTopLevelLoop = oldLoopOp;
  SmallVector<scf::ForOp> oldNestedForOps, newNestedForOps;
  if (isInsertSliceOp) {
    oldNestedForOps = getPerfectlyOuterLoops(cast<scf::ForOp>(oldLoopOp));
    oldTopLevelLoop = oldNestedForOps.front();
  }
  if (failed(checkAssumptionForLoop(oldTopLevelLoop, consumerOp))) {
    return rewriter.notifyMatchFailure(
        oldTopLevelLoop,
        "containing loop op should either yield just one value or "
        "have the consumer op as its first user");
  }

  OpBuilder::InsertionGuard g(rewriter);

  // 2. Check consumer is not using scf loop's output as init.
  auto dstOp = cast<DestinationStyleOpInterface>(consumerOp);
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, oldTopLevelLoop->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }
  SmallVector<Value> newInitAppend = dpsInits;

  Location loc = oldLoopOp->getLoc();

  // 3. Create new scf loop op.
  rewriter.setInsertionPoint(consumerOp);

  // 3.a Create new outer scf loops if necessary
  bool isNestedForOps = isInsertSliceOp && oldNestedForOps.size() > 1;
  if (isNestedForOps) {
    for (auto &forOp : MutableArrayRef(oldNestedForOps).drop_back()) {
      SmallVector<Value> newInits;
      newInits = llvm::to_vector(forOp.getInits());
      newInits.append(newInitAppend.begin(), newInitAppend.end());
      auto newLoop = rewriter.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newInits);
      newInitAppend = llvm::map_to_vector(
          newLoop.getRegionIterArgs().take_back(newInitAppend.size()),
          [](BlockArgument bArg) -> Value { return bArg; });
      rewriter.mergeBlocks(
          forOp.getBody(), newLoop.getBody(),
          newLoop.getBody()->getArguments().take_front(initSize + 1));
      rewriter.replaceOp(
          forOp, newLoop->getResults().take_front(forOp->getNumResults()));
      newNestedForOps.push_back(newLoop);
    }
    rewriter.setInsertionPoint(oldNestedForOps.back());
  }

  // 3.b Create new inner most scf loop
  Operation *newLoopOp = nullptr;
  Block *newLoopBody = nullptr;
  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldLoopOp);
    llvm::append_range(newOuts, forOp.getInits());
    newOuts.append(newInitAppend);
    oldLoopBody = forOp.getBody();
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newOuts);
    newLoopOp = newForOp;
    newLoopBody = newForOp.getBody();
  } else {
    auto forallOp = cast<scf::ForallOp>(oldLoopOp);
    llvm::append_range(newOuts, forallOp.getOutputs());
    newOuts.append(newInitAppend);
    oldLoopBody = forallOp.getBody();
    auto newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), newOuts, forallOp.getMapping());
    newLoopOp = newForallOp;
    rewriter.eraseOp(newForallOp.getTerminator());
    newLoopBody = newForallOp.getBody();
  }

  // 4. Move the loop body to the new op.
  unsigned oldNumArguments = oldLoopBody->getNumArguments();
  rewriter.mergeBlocks(oldLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(oldNumArguments));

  // 5. Set insertion point before terminator op of the loop and create a new
  // tensor.insert_slice. In the scf.for case this is a clone of the
  // candidateSliceOp whereas in the scf.forall case this is created from the
  // operands of tensor.parallel_insert_slice.
  tensor::InsertSliceOp clonedInsertSliceOp;
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    auto newForallOp = cast<scf::ForallOp>(newLoopOp);
    rewriter.setInsertionPoint(newForallOp.getTerminator());
    clonedInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, sliceOp.getSource(), sliceOp.getDest(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  } else {
    rewriter.setInsertionPoint(candidateSliceOp);
    clonedInsertSliceOp =
        cast<tensor::InsertSliceOp>(rewriter.clone(*candidateSliceOp));
  }

  // 6.a. Clone consumer op.
  auto newForOpBlockArgsForConsumerDest =
      newLoopBody->getArguments().drop_front(oldNumArguments);
  auto clonedConsumerOp = cast<TilingInterface>(cloneOpAndUpdateDestinationArgs(
      rewriter, consumerOp, newForOpBlockArgsForConsumerDest));

  // 6.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    operandToReplace.set(clonedInsertSliceOp.getResult());
  });

  // 7 - Perform tiling of the cloned consumer and replace the operand at
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

  // 8 - Extract offset/sizes/strides required to create the
  // tensor.insert_slice/parallel_insert_slice for each result of the consumer.
  SmallVector<OpFoldResult> offsets = ossSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = ossSliceOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = ossSliceOp.getMixedStrides();

  // 9. Check all insert stride is 1.
  if (llvm::any_of(strides, [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "containingOp's result yield with stride");
  }

  // 10. Try to get iter domain position from input position.
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  if (failed(clonedConsumerOp.getIterationDomainTileFromOperandTile(
          rewriter, operandNumber, offsets, sizes, iterDomainOffsets,
          iterDomainSizes))) {
    return rewriter.notifyMatchFailure(
        clonedConsumerOp, "can't get iter domain position from input position");
  }

  // 11. Try to fetch the offset and size for all results of the cloned
  // consumer. This would then be used to form the corresponding
  // tensor.insert_slice/parallel_insert_slice later.
  unsigned totalNumResultsOfConsumer = clonedConsumerOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(
      totalNumResultsOfConsumer);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(totalNumResultsOfConsumer);
  for (auto [idx, v] : llvm::enumerate(clonedConsumerOp->getResults())) {
    if (failed(clonedConsumerOp.getResultTilePosition(
            rewriter, idx, iterDomainOffsets, iterDomainSizes,
            resultOffsets[idx], resultSizes[idx]))) {
      return rewriter.notifyMatchFailure(
          clonedConsumerOp,
          "can't get result domain position from iter domain position");
    }
  }

  auto arrayRefOffsets = ArrayRef<SmallVector<OpFoldResult>>(resultOffsets);
  auto arrayRefSizes = ArrayRef<SmallVector<OpFoldResult>>(resultSizes);
  if (isInsertSliceOp) {
    auto newForOp = cast<scf::ForOp>(newLoopOp);
    fixTerminatorSCFYield(
        rewriter, newForOp, *tileAndFuseResult, arrayRefOffsets, arrayRefSizes,
        newForOp.getBody()->getArguments().drop_front(1 + initSize));
  } else {
    auto newForallOp = cast<scf::ForallOp>(newLoopOp);
    fixTerminatorSCFInParallel(
        rewriter, newForallOp, tileAndFuseResult->tiledOps[0]->getResults(),
        arrayRefOffsets, arrayRefSizes,
        newForallOp.getBody()->getArguments().drop_front(rank + initSize));
  }

  // 12.  Restore outer loops from inner to outer
  if (isNestedForOps) {
    newNestedForOps.push_back(cast<scf::ForOp>(newLoopOp));
    for (auto [outerLoop, innerLoop] :
         llvm::zip_equal(MutableArrayRef(newNestedForOps).drop_back(),
                         MutableArrayRef(newNestedForOps).drop_front())) {
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

  // 13. Replace the result of scf loop and consumer op with new loop's results.
  for (auto &&[oldResult, newResult] :
       llvm::zip_first(oldLoopOp->getResults(), newLoopOp->getResults())) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  Operation *newTopLevelLoop =
      isNestedForOps ? newNestedForOps.front() : newLoopOp;
  for (auto &&[oldResult, newResult] :
       llvm::zip(consumerOp->getResults(),
                 newTopLevelLoop->getResults().drop_front(initSize))) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // 14. Need to erase the old scf loop and the cloned consumer op.
  rewriter.eraseOp(oldLoopOp);
  rewriter.eraseOp(clonedConsumerOp);

  // 15. Need to erase the cloned insertSliceOp and unused extractSliceOp in
  // avoid of complex domination analysis
  assert(clonedInsertSliceOp->hasOneUse());
  auto unUsedExtractOp =
      cast<tensor::ExtractSliceOp>((*clonedInsertSliceOp->getUsers().begin()));
  rewriter.eraseOp(unUsedExtractOp);
  rewriter.eraseOp(clonedInsertSliceOp);

  return scf::SCFFuseConsumerOfSliceResult{
      consumerOpOperand,
      &(tileAndFuseResult->tiledOps[0]->getOpOperand(operandNumber)),
      tileAndFuseResult->tiledOps};
}

/// Fusing real consumer of a single slice even within complex nested loops via
/// multiple application of `tileAndFuseConsumerOfSliceImpl`.
FailureOr<scf::SCFFuseConsumerOfSliceResult>
mlir::scfX::tileAndFuseConsumerOfSlice(RewriterBase &rewriter,
                                       Operation *candidateSliceOp) {
  SmallVector<OffsetSizeAndStrideOpInterface> sliceOpChain;
  if (failed(getResultOfTopLevelLoopYieldInsertSliceOp(candidateSliceOp,
                                                       sliceOpChain)))
    return failure();

  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseConsumerResult;
  // reverse from outer to inner
  std::reverse(sliceOpChain.begin(), sliceOpChain.end());
  // multiple application of `tileAndFuseConsumerOfSliceImpl`
  for (auto &sliceOp : sliceOpChain) {
    fuseConsumerResult = tileAndFuseConsumerOfSliceImpl(rewriter, sliceOp);
    if (failed(fuseConsumerResult)) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "could not fuse consumer of sliceOp");
    }
  }
  return fuseConsumerResult;
}