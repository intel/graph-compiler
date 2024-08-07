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
  if (!loops.empty()) {
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
  }
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
}

static std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfSliceImpl(RewriterBase &rewriter,
                               tensor::ExtractSliceOp candidateSliceOp,
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

/// Get the real producer of candidate ExtractSliceOp
///
/// ```
/// %0 = producer
/// %1 = scf.for(%arg1 = %0)
///   %2 = extract %arg1
///   %3 = scf.for(%arg2 = %2)
///      %4 = extract %args2
///      ...
/// ```
///
/// @param candidateSliceOp: %4 = extract %args2
/// @param backwardSlice: in-out parameter populated by backward extractSliceOps
/// @return OpResult Producer : %0 = producer
FailureOr<OpResult> mlir::scfX::getRealProducerOfExtractSliceOp(
    Operation *candidateSliceOp,
    SmallVector<tensor::ExtractSliceOp> &backwardSlice, unsigned curDepth,
    unsigned maxDepth) {
  if (!isa<tensor::ExtractSliceOp>(candidateSliceOp))
    return failure();
  // control recursive time in avoid of stack overflow
  if (curDepth > maxDepth)
    return failure();

  auto extractOp = cast<tensor::ExtractSliceOp>(candidateSliceOp);
  backwardSlice.push_back(extractOp);
  Value rootSource = extractOp.getSourceMutable().get();

  while (true) {
    if (auto iterArg = dyn_cast<BlockArgument>(rootSource)) {
      if (auto outerLoop = dyn_cast<LoopLikeOpInterface>(
              iterArg.getOwner()->getParentOp())) {
        rootSource = outerLoop.getTiedLoopInit(iterArg)->get();
        continue;
      }
      return failure();
    }
    if (auto sliceOp = rootSource.getDefiningOp<tensor::ExtractSliceOp>()) {
      // walk up loop to find larger candidate extractSliceOp
      return getRealProducerOfExtractSliceOp(sliceOp, backwardSlice,
                                             curDepth + 1);
    }
    break;
  }
  return dyn_cast<OpResult>(rootSource);
}

/// Recursively find the outer nest loops of given loop(included) while the
/// predict function succeed, sorted from outer to inner.
///
/// @param loop: target loop, note that this loop will be also included. I.e.
///              if no other nest loops were found, just return itself.
/// @param pred: predict function, the termination condition of recursive
/// process.
/// @return Outer Nest Loops: nest loops outside given target loop(included).
///
/// E.g.
///
/// ```
///  %0 = scf.for()
///    %1 = scf.for()
///      %2 = scf.for()
/// ```
///
/// If `%2 = scf.for` is given without specific prediction function, this
/// function will return three nest loops: %0 + %1 + %2.
SmallVector<LoopLikeOpInterface> mlir::scfX::getOuterNestLoopsWhile(
    LoopLikeOpInterface loop,
    const std::function<LogicalResult(LoopLikeOpInterface)> &pred) {
  SmallVector<LoopLikeOpInterface> nestLoops = {loop};
  auto outerLoop = dyn_cast<LoopLikeOpInterface>(loop->getParentOp());
  while (outerLoop && succeeded(pred(outerLoop))) {
    nestLoops.push_back(outerLoop);
    outerLoop = dyn_cast<LoopLikeOpInterface>(outerLoop->getParentOp());
  }
  // sorted from outer to inner
  return {nestLoops.rbegin(), nestLoops.rend()};
}

/// Enhanced version of `tileAndFuseProducerOfSliceImpl`, which can deal with
/// multi-level `extractSliceOp`. E.g.
///
/// ```
/// %0 = untiled_producer
/// %1 = scf.for(%arg1 = %0)
///   %2 = extract %arg1
///   %3 = scf.for(%arg2 = %2)
///      %4 = extract %args2
///      %5 = tiled_consumer ins(%4)
/// ```
std::optional<scf::SCFFuseProducerOfSliceResult>
mlir::scfX::tileAndFuseProducerOfSlice(RewriterBase &rewriter,
                                       Operation *candidateSliceOp) {
  SmallVector<tensor::ExtractSliceOp> backwardSlice;
  if (failed(getRealProducerOfExtractSliceOp(candidateSliceOp, backwardSlice)))
    return std::nullopt;

  std::optional<scf::SCFFuseProducerOfSliceResult> fuseProducerResult;
  // reverse from outer to inner
  std::reverse(backwardSlice.begin(), backwardSlice.end());
  // multiple application of `tileAndFuseProducerOfSliceImpl`
  for (auto &&[index, sliceOp] : llvm::enumerate(backwardSlice)) {
    // get nest loops between next candidate sliceOp and tiled producer.
    auto whileProducerOutOfLoopBlock =
        [&fuseProducerResult](LoopLikeOpInterface loop) -> LogicalResult {
      if (fuseProducerResult) {
        Block &body = loop->getRegion(0).front();
        if (fuseProducerResult->tiledAndFusedProducer.getDefiningOp()
                ->getBlock() == &body)
          return failure();
      }
      return success();
    };
    SmallVector<LoopLikeOpInterface> outerLoops =
        getOuterNestLoopsWhile(sliceOp->getParentOfType<LoopLikeOpInterface>(),
                               whileProducerOutOfLoopBlock);
    fuseProducerResult =
        tileAndFuseProducerOfSliceImpl(rewriter, sliceOp, outerLoops);
    if (!fuseProducerResult)
      return std::nullopt;
  }
  return fuseProducerResult;
}

/// Get the real consumers from candidate InsertSliceOp. E.g
///
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
/// %6 = consumerOp ins(%1)
/// ```
///
/// @param candidateSliceOp: %4 = insert
/// @param forwardSlice: in-out parameter populated by forward insertSliceOps
/// @return OpOperand consumers: %6 = consumerOp ins(%1)
FailureOr<SmallVector<OpOperand *>>
mlir::scfX::getRealConsumersFromInsertSliceOp(
    Operation *candidateSliceOp,
    SmallVector<OffsetSizeAndStrideOpInterface> &forwardSlice,
    unsigned curDepth, unsigned maxDepth) {
  if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
          candidateSliceOp))
    return failure();
  // Control recursive time in avoid of stack overflow
  if (curDepth > maxDepth)
    return failure();

  forwardSlice.push_back(
      cast<OffsetSizeAndStrideOpInterface>(candidateSliceOp));
  Value resultOfLoop;
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    Value destValue = sliceOp.getDest();
    auto iterArg = cast<BlockArgument>(destValue);
    auto forallOp = dyn_cast<scf::ForallOp>(iterArg.getOwner()->getParentOp());
    if (!forallOp)
      return failure();
    resultOfLoop = forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));
  } else if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(candidateSliceOp)) {
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

  bool traverseUpperLoop;
  do {
    traverseUpperLoop = false;
    for (OpOperand &useOperand : resultOfLoop.getUses()) {
      if (auto sliceOp =
              dyn_cast<OffsetSizeAndStrideOpInterface>(useOperand.getOwner())) {
        return getRealConsumersFromInsertSliceOp(sliceOp, forwardSlice,
                                                 curDepth + 1);
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(useOperand.getOwner())) {
        // walk through outer loop
        auto forOp = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        if (!forOp)
          return failure();
        resultOfLoop = forOp->getResult(useOperand.getOperandNumber());
        traverseUpperLoop = true;
        break;
      }
    }
  } while (traverseUpperLoop);
  // Return all operands using result of top level loop.
  return llvm::map_to_vector(resultOfLoop.getUses(),
                             [](OpOperand &u) -> OpOperand * { return &u; });
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

/// Fetches the FIRST OpOperand of the tilable user (and use) of the value `val`
/// within the same block, which implements `TilingInterface` and
/// `DestinationStyleOpInterface` and has non-empty user list.
/// Returns failure otherwise.
static FailureOr<OpOperand *> getConsumerFromUses(Value val,
                                                  Block *containingOpBlock) {
  OpOperand *operand = nullptr;
  for (auto &use : val.getUses()) {
    Operation *user = use.getOwner();
    // Step 1. Check if the user is tilable.
    if (!isa<TilingInterface, DestinationStyleOpInterface>(user)) {
      // TODO: We have to init result of consumer before scf.for, use
      //       DestinationStyleOpInterface to get result shape from init for
      //       now. Add support for other op such as op has
      //       InferTypeOpInterface.
      continue;
    } else {
      // Step 2. Check if user stay in the same block.
      if (containingOpBlock != user->getBlock())
        continue;
      // Step 3. Check if user has succeeding user. Otherwise, it usually
      // represents already tiled.
      if (user->use_empty())
        continue;
      operand = &use;
      break;
    }
  }
  if (!operand)
    return failure();

  return operand;
}

/// Check if it is the ForOp that yield the result of inner loop
static LogicalResult isForOpYieldResultOfInnerLoop(LoopLikeOpInterface loop) {
  if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
    for (auto &&[index, op] :
         llvm::enumerate(forOp.getBody()->getOperations())) {
      // If the orderIndex of inner loop is the last second one before the
      // yieldOp of ForOp, the given loop must yield the result of inner loop.
      if (isa<LoopLikeOpInterface>(op)) {
        return success((index + 2) == forOp.getBody()->getOperations().size());
      }
    }
  }
  return failure();
}

/// Fetch the untiled consumer of a scf.for's result which is yielded by a
/// tensor.insert_slice. This function makes the following assumptions that
/// tensor.insert_slice has scf.yield as its only user.
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
  LoopLikeOpInterface topLevelForOp =
      scfX::getOuterNestLoopsWhile(forOp, isForOpYieldResultOfInnerLoop)
          .front();
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

/// This utility currently checks whether the first userOp of loop is NOT before
/// the last defineOp of consumer. Currently we clone the loop op right before
/// a certain op in order to maintain a valid def-use chain. This utility thus
/// helps ensuring that no invalid IR is formed due to the same. E.g.
///
/// ```
/// %0 = scf.for() {
///
/// }
/// ...
/// %1 = firstUserOfLoop(%0)
/// ...
/// %2 = lastDefOfConsumer
/// ...
/// %3 = consumerOp(%2)
/// ```
///
/// If the `firstUserOfLoop`is before `lastDefOfConsumer`, then it would be
/// invalid to clone the loop op right before the `firstUserOfLoop`:
///
/// ```
/// %0:2 = scf.for() {
///    %3 = tiledConsumerOp(%2)
/// }
/// %1 = firstUserOfLoop(%0)
/// ...
/// %2 = lastDefOfConsumer
/// ```
///
/// To address this issue, this utility would double-check there is no user of
/// `firstUserOfLoop` before `lastDefOfConsumer`. If so, move `firstUserOfLoop`
/// after `lastDefOfConsumer`. Then, it turns out valid as follow:
///
/// ```
/// %2 = lastDefOfConsumer
/// %0:2 = scf.for() {
///    %3 = tiledConsumerOp(%2)
/// }
/// %1 = firstUserOfLoop(%0)
/// ```
///
/// @param loopOp: loop operation
/// @param consumerOp: consumer operation
/// @param insertPointBefore: which operation we clone the looOp right before
static LogicalResult checkAssumptionForLoop(Operation *loopOp,
                                            Operation *consumerOp,
                                            Operation **insertPointBefore) {
  Block *parentBlock = consumerOp->getBlock();
  // loopOp and consumerOp should stay in the same block.
  if (loopOp->getBlock() != parentBlock)
    return failure();

  Operation *firstUserOfLoop = consumerOp, *lastDefOfConsumer = loopOp;
  // Find the first user of loopOp
  for (Operation *userOp : loopOp->getUsers()) {
    if (userOp == consumerOp)
      continue;
    // `ParallelInsertSlice` located inside `InParallelOp` has no same parent
    // block with any other types of operation. Thus, just redirecting to its
    // parent `InParallelOp`.
    if (isa<tensor::ParallelInsertSliceOp>(userOp))
      userOp = userOp->getParentOfType<scf::InParallelOp>();

    if (parentBlock != userOp->getBlock())
      return failure();

    if (userOp->isBeforeInBlock(firstUserOfLoop))
      firstUserOfLoop = userOp;
  }
  // Find the last define of consumer
  for (Value operand : consumerOp->getOperands()) {
    // If the operand is `BlockArgument`, auto skip.
    if (isa<BlockArgument>(operand))
      continue;
    auto defineOp = operand.getDefiningOp();
    if (defineOp == loopOp)
      continue;
    if (!defineOp || parentBlock != defineOp->getBlock())
      return failure();
    if (lastDefOfConsumer->isBeforeInBlock(defineOp))
      lastDefOfConsumer = defineOp;
  }
  if (firstUserOfLoop->isBeforeInBlock(lastDefOfConsumer)) {
    // Try to move if possible
    if (llvm::all_of(firstUserOfLoop->getUsers(),
                     [&lastDefOfConsumer, &parentBlock](Operation *userOp) {
                       return userOp->getBlock() == parentBlock &&
                              lastDefOfConsumer->isBeforeInBlock(userOp);
                     })) {
      // Safely moving
      firstUserOfLoop->moveAfter(lastDefOfConsumer);
    } else {
      return failure();
    }
  }
  // Set InsertPoint
  *insertPointBefore = firstUserOfLoop;
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
/// As for `insertSlice`, it also supports nest outer loop structure without
/// any other slice inside. E.g.
///
/// ```
/// scf.for()
///   scf.for()
///      scf.for()
///         ...
///         insert_slice
///         yield
///      yield
///   yield
/// ```
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
  SmallVector<LoopLikeOpInterface> oldNestedForOps, newNestedForOps;
  if (isInsertSliceOp) {
    oldNestedForOps =
        scfX::getOuterNestLoopsWhile(cast<LoopLikeOpInterface>(oldTopLevelLoop),
                                     isForOpYieldResultOfInnerLoop);
    oldTopLevelLoop = oldNestedForOps.front();
  }

  // 2.a Check assumption for loop and find suitable insertPoint that loop
  // structure would be cloned right before.
  Operation *insertPointBefore = nullptr;
  if (failed(checkAssumptionForLoop(oldTopLevelLoop, consumerOp,
                                    &insertPointBefore))) {
    return rewriter.notifyMatchFailure(
        oldTopLevelLoop, "containing loop op does not satisfy the assumption "
                         "and no suitable insertPoint is found");
  }

  OpBuilder::InsertionGuard g(rewriter);

  // 2.b Check consumer is not using scf loop's output as init.
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(consumerOp);
  if (!dstOp)
    return rewriter.notifyMatchFailure(consumerOp,
                                       "consumer op is not DPS operation");
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
  rewriter.setInsertionPoint(insertPointBefore);

  // 3.a Create new outer scf loops if necessary
  bool isNestedForOps = isInsertSliceOp && oldNestedForOps.size() > 1;
  if (isNestedForOps) {
    for (auto &&[index, loopOp] :
         llvm::enumerate(MutableArrayRef(oldNestedForOps).drop_back())) {
      auto forOp = cast<scf::ForOp>(loopOp);
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
      rewriter.setInsertionPointAfter(oldNestedForOps[index + 1]);
    }
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
  SmallVector<OffsetSizeAndStrideOpInterface> forwardSlice;
  if (failed(getRealConsumersFromInsertSliceOp(candidateSliceOp, forwardSlice)))
    return failure();

  FailureOr<scf::SCFFuseConsumerOfSliceResult> fuseConsumerResult;
  // reverse from outer to inner
  std::reverse(forwardSlice.begin(), forwardSlice.end());
  // multiple application of `tileAndFuseConsumerOfSliceImpl`
  for (auto &sliceOp : forwardSlice) {
    fuseConsumerResult = tileAndFuseConsumerOfSliceImpl(rewriter, sliceOp);
    if (failed(fuseConsumerResult)) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "could not fuse consumer of sliceOp");
    }
  }
  return fuseConsumerResult;
}