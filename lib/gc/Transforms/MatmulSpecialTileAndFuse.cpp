//=== MatmulSpecialTileAndFuse.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass finds two consecutive matmuls,
// tiles them and fuses them.
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_MATMULSPECIALTILEANDFUSE
#include "gc/Transforms/Passes.h.inc"

#define DEBUG_TYPE "linalg-transforms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

/// Add new operands to the forall op for users of the producerOp
/// that are dominated by the containing scf.forall op.
static Operation *replaceForAllWithNewSignature(
    RewriterBase &rewriter, Operation *producerOp, Operation *containingOp,
    TilingResult &tileAndFuseResult, int64_t resultNumber,
    SmallVector<OpFoldResult> &offsets, SmallVector<OpFoldResult> &sizes) {

  // Count number of users not including the containing op
  SetVector<Operation *> dominatedUsers;
  DominanceInfo domInfo(containingOp);
  for (Operation *user : producerOp->getResult(resultNumber).getUsers()) {
    if (!containingOp->isAncestor(user) &&
        (domInfo.dominates(containingOp, user))) {
      dominatedUsers.insert(user);
    }
  }
  if (dominatedUsers.empty())
    return nullptr;

  // Create new scf.forall op
  auto forallOp = cast<scf::ForallOp>(containingOp);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Get new output
  Location loc = forallOp.getLoc();
  auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
  if (!genericOp)
    return nullptr;
  SmallVector<Value> outputs = genericOp.getOutputs();
  SmallVector<Value> newOuts(forallOp.getOutputs());
  newOuts.push_back(outputs[resultNumber]);

  // Create new scf.forall op
  auto newforallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newOuts, forallOp.getMapping());
  rewriter.eraseBlock(newforallOp.getBody());
  newforallOp.getRegion().takeBody(forallOp.getRegion());

  // Add additional block argument for new value being returned
  // and replaces all uses of the new output with corresponding bbArg
  // inside the scf.forall to enable fusion into this new scf.forall.
  newforallOp.getBody()->addArgument(newOuts.back().getType(),
                                     newOuts.back().getLoc());
  auto bbArgs = newforallOp.getBody()->getArguments();
  rewriter.replaceUsesWithIf(newOuts.back(), bbArgs.back(),
                             [&](OpOperand &use) {
                               Operation *op = use.getOwner();
                               return newforallOp->isProperAncestor(op);
                             });

  // Fix terminator
  scf::InParallelOp terminatorOp = newforallOp.getTerminator();
  SmallVector<Operation *> yieldingOps = llvm::to_vector<4>(llvm::map_range(
      terminatorOp.getYieldingOps(), [](Operation &op) { return &op; }));
  Operation *firstYieldOp = yieldingOps.front();
  rewriter.setInsertionPoint(firstYieldOp);
  Value src = tileAndFuseResult.tiledValues[0];
  Value dst = newforallOp.getRegionIterArgs().back();
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  rewriter.create<tensor::ParallelInsertSliceOp>(firstYieldOp->getLoc(), src,
                                                 dst, offsets, sizes, strides);

  for (auto result : llvm::enumerate(forallOp.getResults())) {
    rewriter.replaceAllUsesWith(result.value(),
                                newforallOp->getResult(result.index()));
  }
  rewriter.replaceUsesWithIf(producerOp->getResult(resultNumber),
                             newforallOp->getResults().back(),
                             [&](OpOperand &use) {
                               Operation *user = use.getOwner();
                               return dominatedUsers.contains(user);
                             });
  return newforallOp;
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
/// If tiled op has uses that are dominated by `containingOp`, return
/// a new `containingOp` with results of the fused op appended to
/// results of the `containingOp` or nullptr if there are no dominated uses.
static std::tuple<SmallVector<Operation *>, Operation *>
tileAndFuseFirstExtractUse(RewriterBase &rewriter, Operation *producerOp,
                           Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    return {};
  }

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto it = llvm::find_if(tileableProducer->getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (it == tileableProducer->getUsers().end()) {
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  int64_t resultNumber =
      cast<OpResult>(sliceOpToTile.getSource()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  SmallVector<OpFoldResult> offsets = sliceOpToTile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOpToTile.getMixedSizes();

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducer.generateResultTileValue(rewriter, resultNumber, offsets,
                                               sizes);

  if (failed(tileAndFuseResult)) {
    return {};
  }

  for (auto *tiledOp : tileAndFuseResult->tiledOps) {
    LLVM_DEBUG(DBGS() << "tiledProducer: " << *tiledOp << "\n");
  }

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  if (failed(maybeRankReduced)) {
    return {};
  }
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Add new outputs to containing op, if required
  Operation *newContainingOp = replaceForAllWithNewSignature(
      rewriter, producerOp, containingOp, *tileAndFuseResult, resultNumber,
      offsets, sizes);

  return std::make_tuple(tileAndFuseResult->tiledOps, newContainingOp);
}

/// First, find the first "scf::ForallOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static SmallVector<Operation *>
tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Operation *producerOp, Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    return {};
  }

  // Search the first use by a "scf::ForallOp" user.
  scf::ForallOp forallOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        forallOp = dyn_cast<scf::ForallOp>(use.getOwner());
        return forallOp;
      });
  // If it's not from the containing op, return.
  if (!forallOp || forallOp != containingOp) {
    return {};
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = forallOp.getTiedBlockArgument(pUse);

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (itBBArgUsers == bbArg.getUsers().end()) {
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = cast<OpResult>(pUse->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    return {};
  }

  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tileAndFuseResult)) {
    return {};
  }

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  assert(succeeded(maybeRankReduced) && "unexpected shape");
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Replace the use in containingOp.
  rewriter.modifyOpInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return tileAndFuseResult->tiledOps;
}

static Operation *cloneAndFuseFirstUse(RewriterBase &rewriter,
                                       Operation *producerOp,
                                       Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an use by cloning\n");

  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      if (containingOp->isProperAncestor(use.getOwner())) {
        uses.push_back(&use);
        continue;
      }
      // Cannot clone and fuse if the use is by the containing op itself: fail
      // immediately.
      if (containingOp == use.getOwner()) {
        return nullptr;
      }
    }
  }

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty()) {
    return nullptr;
  }

  // Clone and fuse inside the containing op.
  Operation *fusedOp = nullptr;
  OpOperand *use = uses.front();
  // Parallel insert slice is not a valid clone destination.
  // TODO: Generalize to other type of ops.
  assert(!isa<tensor::ParallelInsertSliceOp>(use->getOwner()) &&
         "Parallel insert slice is not a valid clone destination");
  unsigned resultNumber = cast<OpResult>(use->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use->getOwner());
  fusedOp = rewriter.clone(*producerOp);
  rewriter.modifyOpInPlace(
      use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });

  return fusedOp;
}

// Copied from transform.structured.fuse_into_containing_op.
void fuse(IRRewriter &rewriter, Operation *producerOp,
          Operation *containingOp) {
  SmallVector<Operation *> fusedOps;
  llvm::dbgs() << "Fuse " << *producerOp << " into " << *containingOp << '\n';

  // TODO: If there are multiple uses of the producer in the containing op,
  // we currently tile/clone the op multiple times (once per use). In some
  // cases, we can tile/clone once and reuse the value for each use.
  // Futhermore, producers should then be traversed according to a
  // topological sorting.
  auto [tiledOps, newContainingOp] =
      tileAndFuseFirstExtractUse(rewriter, producerOp, containingOp);
  if (!tiledOps.empty()) {
    LLVM_DEBUG(DBGS() << "\nFused a direct extract use\n" << *containingOp);
    fusedOps.append(tiledOps);
    if (newContainingOp) {
      // Update handles associated with the containing op so we don't need to
      // invalidate them. This is a hack to support better composability
      // between tiling and fusion while a proper mechanism is being
      // investigated.
      rewriter.eraseOp(containingOp);
      containingOp = newContainingOp;
    }
  }

  SmallVector<Operation *> tiledContainingOpOperand =
      tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
          rewriter, producerOp, containingOp);
  if (!tiledContainingOpOperand.empty()) {
    LLVM_DEBUG(DBGS() << "\nFused an extract use through block argument\n"
                      << *containingOp);
    fusedOps.append(tiledContainingOpOperand);
  }

  Operation *cloned = cloneAndFuseFirstUse(rewriter, producerOp, containingOp);
  if (cloned) {
    LLVM_DEBUG(DBGS() << "\nFused an use by cloning\n" << *containingOp);
    fusedOps.push_back(cloned);
  }
}

//===----------------------------------------------------------------------===//
// MatmulSpecialTileAndFuse Pass
//===----------------------------------------------------------------------===//

namespace {
struct MatmulSpecialTileAndFuse
    : public impl::MatmulSpecialTileAndFuseBase<MatmulSpecialTileAndFuse> {
  void runOnOperation() override;
};

void MatmulSpecialTileAndFuse::runOnOperation() {
  Operation *topOp = getOperation();
  MLIRContext *context = topOp->getContext();
  auto &topFunc =
      topOp->getRegions().front().getBlocks().front().getOperations().front();
  Region &region = topFunc.getRegions().front();
  Block &block = region.getBlocks().front();
  IRRewriter rewriter(context);

  // tile the matmul to a scf::ForallOp
  getOperation()->walk([&](linalg::MatmulOp mm) {
    Operation *op = mm.getOperation();
    if (auto attr = op->getAttr("matmul_special_tiling")) {
      // if (cast<BoolAttr>(attr).getValue() == true) {}
      SmallVector<int64_t> numThreadsInt = {8, 0, 8};
      SmallVector<OpFoldResult> numThreads;
      numThreads.reserve(numThreadsInt.size());
      for (size_t idx = 0; idx < numThreadsInt.size(); ++idx) {
        numThreads.push_back(
            OpFoldResult{rewriter.getI64IntegerAttr(numThreadsInt[idx])});
      }

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mm);
      auto tilableOp = dyn_cast<TilingInterface>(op);
      auto tilingResult =
          linalg::tileToForallOp(rewriter, tilableOp, numThreads, std::nullopt);
      if (failed(tilingResult))
        return signalPassFailure();
      rewriter.replaceOp(mm, tilingResult->tileOp->getResults());
      tilingResult->tileOp->setAttr("matmul_special_tiled_forall",
                                    rewriter.getBoolAttr(true));
    }
  });

  getOperation()->walk([&](scf::ForallOp forall) {
    Operation *forallOp = forall.getOperation();
    if (auto attr = forallOp->getAttr("matmul_special_tiled_forall")) {
      // fuse parent ops into the forall
      while (true) {
        bool canFuse = false;
        for (auto &op : block.getOperations()) {
          // Do not try to fuse EmptyOp.
          if (auto emptyParent = dyn_cast<tensor::EmptyOp>(op)) {
            continue;
          }
          for (Operation *user : op.getUsers()) {
            auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
            if (sliceOp && forallOp->isProperAncestor(sliceOp)) {
              // The user of 'op' is an ExtractSliceOp, which is inside of
              // the 'forallOp'. We can fuse 'op' into 'forallOp'.
              canFuse = true;
              fuse(rewriter, &op, forallOp);
              // End fusing after the matmul
              if (auto mmParent = dyn_cast<linalg::MatmulOp>(op)) {
                return WalkResult::interrupt();
              }
            }
          }
        }
        if (!canFuse) {
          break;
        }
      }
    }
  });

  // Delete dead operations by dialects' canonicalizer
  RewritePatternSet owningPatterns(context);
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(owningPatterns);

  ArrayRef<std::string> disabledPatterns, enabledPatterns;
  std::shared_ptr<const FrozenRewritePatternSet> patterns =
      std::make_shared<FrozenRewritePatternSet>(
          std::move(owningPatterns), disabledPatterns, enabledPatterns);
  GreedyRewriteConfig config;
  LogicalResult converged =
      applyPatternsAndFoldGreedily(topOp, *patterns, config);
  (void)converged;
}
} // namespace

std::unique_ptr<Pass> createMatmulSpecialTileAndFusePass() {
  return std::make_unique<MatmulSpecialTileAndFuse>();
}

} // namespace gc
} // namespace mlir