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

static LogicalResult checkAssumptionForFusingConsumer(Value result) {
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

/// Fetch the first untiled consumer of a scf.for's result which is yielded by
/// a tensor.insert_slice. This function makes the following assumptions :-
/// 1.  tensor.insert_slice has scf.yield as its only user.
/// 2.  scf.for's corresponding result has only one use.
static OpOperand *
getUntiledConsumerFromSlice(tensor::InsertSliceOp candidateSliceOp) {
  Value sliceResult = candidateSliceOp.getResult();
  if (failed(checkAssumptionForFusingConsumer(candidateSliceOp.getResult()))) {
    return nullptr;
  }
  // Step 1. Fetch the corresponding output.
  OpOperand &yieldOpOperand = (*sliceResult.getUses().begin());
  unsigned resultNumber = yieldOpOperand.getOperandNumber();
  // Step 2. Check containing op is scf.for.
  Operation *containingOp = candidateSliceOp->getParentOp();
  auto forOp = dyn_cast<scf::ForOp>(containingOp);
  if (!forOp) {
    return nullptr;
  }
  Value resultingValue = forOp->getResult(resultNumber);

  // Step 3. Check resulting value of scf.for has exactly one use.
  if (!llvm::hasSingleElement(resultingValue.getUses())) {
    return nullptr;
  }

  // Step 4. Get uses.
  OpOperand &operand = (*resultingValue.getUses().begin());
  Operation *consumerOp = operand.getOwner();
  // TODO: We have to init result of consumer before scf.for, use
  //       DestinationStyleOpInterface to get result shape from init for now.
  //       Add support for other op such as op has InferTypeOpInterface.
  if (!isa<TilingInterface>(consumerOp) ||
      !isa<DestinationStyleOpInterface>(consumerOp)) {
    return nullptr;
  }
  if (containingOp->getBlock() != consumerOp->getBlock()) {
    return nullptr;
  }
  return &operand;
}

/// Fetch the first untiled consumer of a scf.forall's result which is yielded
/// by a tensor.parallel_insert_slice.
static OpOperand *
getUntiledConsumerFromSlice(tensor::ParallelInsertSliceOp candidateSliceOp) {
  // Step 1. Fetch the corresponding output
  Value sliceDest = candidateSliceOp.getDest();
  auto iterArg = cast<BlockArgument>(sliceDest);
  Operation *containingOp = iterArg.getOwner()->getParentOp();
  // Step 2. Check that the containing op is scf.forall.
  auto forallOp = dyn_cast<scf::ForallOp>(containingOp);
  if (!forallOp) {
    return nullptr;
  }
  Value resultingValue =
      forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));
  // Step 3. Check resulting value of scf.forall has exactly one use.
  Value::use_range uses = resultingValue.getUses();
  if (!llvm::hasSingleElement(uses)) {
    return nullptr;
  }

  // Step 4. Get uses.
  OpOperand &operand = (*resultingValue.getUses().begin());
  Operation *consumerOp = operand.getOwner();
  // TODO: We have to init result of consumer before scf.forall, use
  //       DestinationStyleOpInterface to get result shape from init for now.
  //       Add support for other op such as op has InferTypeOpInterface.
  if (!isa<TilingInterface>(consumerOp) ||
      !isa<DestinationStyleOpInterface>(consumerOp)) {
    return nullptr;
  }
  if (containingOp->getBlock() != consumerOp->getBlock()) {
    return nullptr;
  }
  return &operand;
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

static OpOperand *getUntiledConsumerFromSlice(Operation *sliceOp) {
  if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(insertSlice);
  } else if (auto parallelInsertSlice =
                 dyn_cast<tensor::ParallelInsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(parallelInsertSlice);
  } else {
    return nullptr;
  }
}

static void
fixTerminatorSCFYield(RewriterBase &rewriter, scf::ForOp newForOp,
                      TilingResult tilingResult,
                      SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
                      SmallVector<SmallVector<OpFoldResult>> &resultSizes,
                      SmallVector<OpFoldResult> &strides, unsigned initSize) {
  scf::YieldOp oldTerminatorOp =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands(oldTerminatorOp.getResults());
  rewriter.setInsertionPointAfter(oldTerminatorOp);
  MutableArrayRef<BlockArgument> bbArgs = newForOp.getBody()->getArguments();
  Location loc = newForOp.getLoc();
  for (auto [idx, v] :
       llvm::enumerate(tilingResult.tiledOps[0]->getResults())) {
    SmallVector<OpFoldResult> strides(resultOffsets[idx].size(),
                                      rewriter.getIndexAttr(1));
    Value newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, v, bbArgs[1 + initSize + idx], resultOffsets[idx],
        resultSizes[idx], strides);
    newYieldOperands.push_back(newInsertSliceOp);
  }
  rewriter.create<scf::YieldOp>(loc, newYieldOperands);
  rewriter.eraseOp(oldTerminatorOp);
}

static void fixTerminatorSCFInParallel(
    RewriterBase &rewriter, scf::ForallOp newForallOp,
    TilingResult tilingResult,
    SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &resultSizes,
    SmallVector<OpFoldResult> &strides, unsigned initSize, unsigned rank) {
  scf::InParallelOp newTerminatorOp = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(newTerminatorOp.getBody());
  Location firstYieldOpLoc =
      (*(newTerminatorOp.getYieldingOps().begin())).getLoc();
  MutableArrayRef<BlockArgument> bbArgs = newForallOp.getBody()->getArguments();
  for (auto [idx, v] :
       llvm::enumerate(tilingResult.tiledOps[0]->getResults())) {
    SmallVector<OpFoldResult> strides(resultOffsets[idx].size(),
                                      rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        firstYieldOpLoc, v, bbArgs[rank + initSize + idx], resultOffsets[idx],
        resultSizes[idx], strides);
  }
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

  // 1. Get the consumer of scf.for for the result yielded by
  // tensor.insert_slice/parallel_insert_slice.
  OpOperand *consumerOpOperand = getUntiledConsumerFromSlice(candidateSliceOp);
  if (!consumerOpOperand) {
    return rewriter.notifyMatchFailure(candidateSliceOp,
                                       "could not fetch consumer to fuse");
  }
  Operation *consumerOp = consumerOpOperand->getOwner();
  unsigned operandNumber = consumerOpOperand->getOperandNumber();
  unsigned resultNumber =
      cast<OpResult>(consumerOpOperand->get()).getResultNumber();

  Operation *oldLoopOp = nullptr;
  SmallVector<Value> newOuts;
  Block *oldLoopBody = nullptr;
  unsigned initSize = 0;
  unsigned rank = 1;
  if (isInsertSliceOp) {
    auto forOp = candidateSliceOp->template getParentOfType<scf::ForOp>();
    SmallVector<Value> forOpOuts(forOp.getInits());
    oldLoopOp = forOp;
    newOuts = forOpOuts;
    oldLoopBody = forOp.getBody();
    initSize = forOp.getInits().size();
  } else {
    auto forallOp = candidateSliceOp->template getParentOfType<scf::ForallOp>();
    SmallVector<Value> forallOpOuts(forallOp.getOutputs());
    oldLoopOp = forallOp;
    newOuts = forallOpOuts;
    oldLoopBody = forallOp.getBody();
    initSize = forallOp.getOutputs().size();
    rank = forallOp.getRank();
  }

  if (failed(checkAssumptionForLoop(oldLoopOp, consumerOp))) {
    llvm::dbgs() << "failed\n";
    return rewriter.notifyMatchFailure(
        oldLoopOp, "containing loop op should either yield just one value or "
                   "have the consumer op as its first user");
  }

  OpBuilder::InsertionGuard g(rewriter);

  // 2. Check consumer is not using scf loop's output as init.
  auto dstOp = cast<DestinationStyleOpInterface>(consumerOp);
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, oldLoopOp->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }
  newOuts.append(dpsInits);

  Location loc = oldLoopOp->getLoc();

  // 3. Create new scf loop op.
  rewriter.setInsertionPoint(consumerOp);
  Operation *newLoopOp = nullptr;
  Block *newLoopBody = nullptr;
  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldLoopOp);
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newOuts);
    newLoopOp = newForOp;
    newLoopBody = newForOp.getBody();
  } else {
    auto forallOp = cast<scf::ForallOp>(oldLoopOp);
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

  // 5.a. Clone consumer after the cloned
  // tensor.insert_slice/parallel_insert_slice op.
  rewriter.setInsertionPointAfter(candidateSliceOp);
  auto newForOpBlockArgsForConsumerDest =
      newLoopBody->getArguments().drop_front(oldNumArguments);
  auto clonedConsumerOp = cast<TilingInterface>(cloneOpAndUpdateDestinationArgs(
      rewriter, consumerOp, newForOpBlockArgsForConsumerDest));

  // 5.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice/parallel_insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(candidateSliceOp)) {
      operandToReplace.set(sliceOp.getResult());
    } else if (auto sliceOp =
                   dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
      operandToReplace.set(sliceOp.getSource());
    }
  });

  // 6 - Perform tiling of the cloned consumer.
  if (isInsertSliceOp) {
    rewriter.setInsertionPointAfter(clonedConsumerOp);
  } else {
    rewriter.setInsertionPoint(cast<scf::ForallOp>(newLoopOp).getTerminator());
  }
  auto ossSliceOp = cast<OffsetSizeAndStrideOpInterface>(candidateSliceOp);
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter, ossSliceOp, clonedConsumerOp->getOpOperand(operandNumber));
  if (failed(tileAndFuseResult)) {
    return rewriter.notifyMatchFailure(clonedConsumerOp,
                                       "failed to tile consumer op: ");
  }

  // 7 - Extract offset/sizes/strides required to create the
  // tensor.insert_slice/parallel_insert_slice for each result of the consumer.
  SmallVector<OpFoldResult> offsets = ossSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = ossSliceOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = ossSliceOp.getMixedStrides();

  // 8. Check all insert stride is 1.
  if (llvm::any_of(strides, [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "containingOp's result yield with stride");
  }

  // 9. Try to get iter domain position from input position.
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;

  if (isInsertSliceOp) {
    rewriter.setInsertionPointAfter(clonedConsumerOp);
  } else {
    rewriter.setInsertionPointAfter(tileAndFuseResult->tiledOps[0]);
  }
  if (failed(clonedConsumerOp.getIterationDomainTileFromOperandTile(
          rewriter, operandNumber, offsets, sizes, iterDomainOffsets,
          iterDomainSizes))) {
    return rewriter.notifyMatchFailure(
        clonedConsumerOp, "can't get iter domain position from input position");
  }

  // 10. Try to fetch the offset and size for all results of the cloned
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

  if (isInsertSliceOp) {
    fixTerminatorSCFYield(rewriter, cast<scf::ForOp>(newLoopOp),
                          *tileAndFuseResult, resultOffsets, resultSizes,
                          strides, initSize);
  } else {
    fixTerminatorSCFInParallel(rewriter, cast<scf::ForallOp>(newLoopOp),
                               *tileAndFuseResult, resultOffsets, resultSizes,
                               strides, initSize, rank);
  }

  // 12. Replace the result of scf loop and consumer op with new loop's results.
  for (auto &&[oldResult, newResult] :
       llvm::zip_first(oldLoopOp->getResults(), newLoopOp->getResults())) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  for (auto &&[index, oldValue] : llvm::enumerate(consumerOp->getResults())) {
    rewriter.replaceAllUsesWith(oldValue,
                                newLoopOp->getResult(initSize + index));
  }

  // 13. Need to erase the old scf loop and the cloned consumer op.
  rewriter.eraseOp(oldLoopOp);
  rewriter.eraseOp(clonedConsumerOp);

  return scf::SCFFuseConsumerOfSliceResult{
      consumerOp, tileAndFuseResult->tiledOps[0], {}};
}
