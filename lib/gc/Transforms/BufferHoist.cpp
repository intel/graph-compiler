//===- BufferHoist.cpp - Buffer hoist in nested parallel loop ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "gc/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir::gc {

#define GEN_PASS_DEF_BUFFERNESTEDPARALLELLOOPHOISTING
#include "gc/Transforms/Passes.h.inc"
namespace {

/// Returns true if the given operation implements a known high-level region-
/// based control-flow interface.
static bool isKnownControlFlowInterface(Operation *op) {
  return isa<LoopLikeOpInterface, RegionBranchOpInterface>(op);
}

/// Returns true if the given operation represents a loop by testing whether it
/// implements the `LoopLikeOpInterface` or the `RegionBranchOpInterface`. In
/// the case of a `RegionBranchOpInterface`, it checks all region-based control-
/// flow edges for cycles.
static bool isLoop(Operation *op) {
  // If the operation implements the `LoopLikeOpInterface` it can be considered
  // a loop.
  if (isa<LoopLikeOpInterface>(op))
    return true;

  // If the operation does not implement the `RegionBranchOpInterface`, it is
  // (currently) not possible to detect a loop.
  auto regionInterface = dyn_cast<RegionBranchOpInterface>(op);
  if (!regionInterface)
    return false;

  return regionInterface.hasLoop();
}
/// Return whether the given operation is a loop with sequential execution
/// semantics.
static bool isSequentialLoop(Operation *op) {
  return !op->hasTrait<OpTrait::HasParallelRegion>() && isLoop(op);
}

/// Return whether the given operation is a loop with parallel execution
/// semantics.
static bool isParallelLoop(Operation *op) {
  return isLoop(op) && op->hasTrait<OpTrait::HasParallelRegion>();
}

static bool allowHoisting(Operation *op) {
  auto allocOp = dyn_cast<AllocationOpInterface>(op);
  return allocOp &&
         static_cast<uint8_t>(allocOp.getHoistingKind() & HoistingKind::Loop);
}

/// A state implementation compatible with the `BufferAllocationHoisting` class
/// that hoists allocations out of loops.
struct BufferHoistingState {
  /// A pointer to the current dominance info.
  DominanceInfo *dominators;

  /// The current allocation value.
  Value allocValue;

  /// The current placement block (if any).
  Block *placementBlock;

  /// Initializes the state base.
  BufferHoistingState(DominanceInfo *dominators, Value allocValue,
                      Block *placementBlock)
      : dominators(dominators), allocValue(allocValue),
        placementBlock(placementBlock) {}

  /// Remembers the dominator block of all aliases.
  Block *aliasDominatorBlock = nullptr;

  /// Computes the upper bound for the placement block search.
  Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
    aliasDominatorBlock = dominatorBlock;
    // If there is a dependency block, we have to use this block as an upper
    // bound to satisfy all allocation value dependencies.
    return dependencyBlock ? dependencyBlock : nullptr;
  }

  /// Returns true if the given operation represents a loop with sequential
  /// execution semantics and one of the aliases caused the
  /// `aliasDominatorBlock` to be "above" the block of the given loop operation.
  /// If this is the case, it indicates that the allocation is passed via a back
  /// edge.
  bool isLegalPlacement(Operation *op) {
    return isSequentialLoop(op) &&
           !dominators->dominates(aliasDominatorBlock, op->getBlock());
  }

  /// Returns true if the given operation should be considered for hoisting.
  static bool shouldHoistOpType(Operation *op) { return allowHoisting(op); }

  /// Does not change the internal placement block, as we want to move
  /// operations out of loops only.
  void recordMoveToDominator(Block *block) {}

  /// Sets the current placement block to the given block.
  void recordMoveToParent(Block *block) { placementBlock = block; }
};

template <typename StateT>
class BufferHoisting : public BufferPlacementTransformationBase {
public:
  BufferHoisting(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op), scopeOp(op) {}

  LogicalResult hoist() {

    SmallVector<Value> allocsAndAllocas;
    for (BufferPlacementAllocs::AllocEntry &entry : allocs)
      allocsAndAllocas.push_back(std::get<0>(entry));
    scopeOp->walk([&](memref::AllocaOp op) {
      allocsAndAllocas.push_back(op.getMemref());
    });

    for (auto allocValue : allocsAndAllocas) {
      // check if the current alloc Op can be hoist or not
      if (!StateT::shouldHoistOpType(allocValue.getDefiningOp()))
        continue;
      Operation *definingOp = allocValue.getDefiningOp();
      assert(definingOp && "No defining op");
      auto operands = definingOp->getOperands();

      auto resultAliases = aliases.resolve(allocValue);
      // Determine the common dominator block of all aliases.
      Block *dominatorBlock =
          findCommonDominator(allocValue, resultAliases, dominators);
      // Init the initial hoisting state.
      StateT state(&dominators, allocValue, allocValue.getParentBlock());
      // Check for additional allocation dependencies to compute an upper bound
      // for hoisting.
      Block *dependencyBlock = nullptr;
      // If this node has dependencies, check all dependent nodes. This ensures
      // that all dependency values have been computed before allocating the
      // buffer.
      for (Value depValue : operands) {
        Block *depBlock = depValue.getParentBlock();
        if (!dependencyBlock || dominators.dominates(dependencyBlock, depBlock))
          dependencyBlock = depBlock;
      }

      bool shouldHoist = false;
      int64_t numThreads = 0;
      Value inductVar;
      Block *curBlock = state.placementBlock;
      Operation *parentOp = curBlock->getParentOp();
      if (isParallelLoop(parentOp)) {
        // Only hoist the allocators that are between nested parallel loops and
        // used inside the inner parallel loop.
        OpBuilder builder(parentOp);
        // check if curBlock contains any forall ops
        SmallVector<Operation *, 4> parallelOpsInCurBlock;
        for (auto &op : curBlock->getOperations()) {
          if (isParallelLoop(&op)) {
            parallelOpsInCurBlock.push_back(&op);
          }
        }
        MemRefType memrefType = dyn_cast<MemRefType>(allocValue.getType());
        bool isStaticShape = memrefType && memrefType.hasStaticShape();
        if (!parallelOpsInCurBlock.empty() && isStaticShape) {
          for (auto *use : allocValue.getUsers()) {
            for (auto *operation : parallelOpsInCurBlock) {
              if (operation->isAncestor(use)) {

                // only support scf.forall for now
                if (auto forallOp = dyn_cast<scf::ForallOp>(parentOp)) {
                  SmallVector<Value> ubs = getValueOrCreateConstantIndexOp(
                      builder, forallOp.getLoc(),
                      forallOp.getMixedUpperBound());
                  if (std::optional<int64_t> ubs0_int =
                          getConstantIntValue(ubs[0])) {
                    numThreads = ubs0_int.value();
                    shouldHoist = true;
                  }
                  inductVar = forallOp.getInductionVar(0);
                }
              }
            }
          }
        }
      }

      // Find the actual placement block and determine the start operation using
      // an upper placement-block boundary. The idea is that placement block
      // cannot be moved any further upwards than the given upper bound.
      Block *placementBlock = findPlacementBlock(
          state, state.computeUpperBound(dominatorBlock, dependencyBlock),
          shouldHoist);
      Operation *startOperation = BufferPlacementAllocs::getStartOperation(
          allocValue, placementBlock, liveness);

      // Move the alloc in front of the start operation.
      Operation *allocOperation = allocValue.getDefiningOp();
      if (shouldHoist) {
        OpBuilder builder(allocOperation);
        // 1) allocate larger buffer...
        //   1.1) query the forall op to get the inductor var and the upperBound
        //   1.2) create new memref type with larger size based on upperBound
        //   1.3) create new allocOp with the new memref type
        //   1.4) move the new allocOp in the correct insertpoint.
        // 2) replace with new memref::subViewOp
        //   2.1) create a new memref::subViewOp op based on the new allocOp
        //   2.2) replace the original allocOp and its use with the new
        //   memref::subViewOp
        //   2.2) remove the original allocOp
        MemRefType memrefType = dyn_cast<MemRefType>(allocValue.getType());
        if (memrefType && memrefType.hasStaticShape()) {
          Type elementType = memrefType.getElementType();
          int64_t memrefTypeRank = memrefType.getRank();

          ArrayRef<int64_t> originalShape = memrefType.getShape();
          // Create a new shape with the desired larger size
          SmallVector<int64_t, 4> newShape(originalShape.begin(),
                                           originalShape.end());
          newShape[0] *= numThreads;

          // Create a new memref type with the larger size
          MemRefType newElementType = MemRefType::get(newShape, elementType);

          auto loc = allocOperation->getLoc();
          Value newAlloc = builder.create<memref::AllocOp>(loc, newElementType);
          newAlloc.getDefiningOp()->moveBefore(startOperation);

          SmallVector<OpFoldResult> size = llvm::to_vector<4>(
              llvm::map_range(originalShape, [&](int64_t v) -> OpFoldResult {
                return builder.getI64IntegerAttr(v);
              }));

          SmallVector<OpFoldResult> offset(memrefTypeRank,
                                           builder.getI64IntegerAttr(0));
          // compile-time const
          Value originShapeValue =
              builder.create<arith::ConstantIndexOp>(loc, originalShape[0]);
          // runtime var
          Value result =
              builder.create<arith::MulIOp>(loc, originShapeValue, inductVar);
          offset[0] = result;

          SmallVector<OpFoldResult> strides(memrefTypeRank,
                                            builder.getI64IntegerAttr(1));

          auto subView = builder.create<memref::SubViewOp>(
              allocOperation->getLoc(), newAlloc, offset, size, strides);
          allocOperation->replaceAllUsesWith(subView->getResults());
          allocOperation->remove();
        } else {
          return failure();
        }
      } else {
        allocOperation->moveBefore(startOperation);
      }
    }
    return success();
  }

private:
  /// Finds a valid placement block by walking upwards in the CFG until we
  /// either cannot continue our walk due to constraints (given by the StateT
  /// implementation) or we have reached the upper-most dominator block.
  Block *findPlacementBlock(StateT &state, Block *upperBound,
                            bool shouldHoist = false) {
    Block *currentBlock = state.placementBlock;
    // ppropriate placement block that satisfies the constraint of the
    // current StateT implementation. Walk until we reach the upperBound
    // block (if any).

    // If we are not able to find a valid parent operation or an
    // associated parent block, break the walk loop.
    Operation *parentOp;
    Block *parentBlock;
    while ((parentOp = currentBlock->getParentOp()) &&
           (parentBlock = parentOp->getBlock()) &&
           (!upperBound ||
            dominators.properlyDominates(upperBound, currentBlock))) {
      // Try to find an immediate dominator and check whether the parent block
      // is above the immediate dominator (if any).
      DominanceInfoNode *idom = nullptr;

      // DominanceInfo doesn't support getNode queries for single-block regions.
      if (!currentBlock->isEntryBlock())
        idom = dominators.getNode(currentBlock)->getIDom();

      if (idom && dominators.properlyDominates(parentBlock, idom->getBlock())) {
        // If the current immediate dominator is below the placement block, move
        // to the immediate dominator block.
        currentBlock = idom->getBlock();
        state.recordMoveToDominator(currentBlock);
      } else {
        // We have to move to our parent block since an immediate dominator does
        // either not exist or is above our parent block. If we cannot move to
        // our parent operation due to constraints given by the StateT
        // implementation, break the walk loop. Furthermore, we should not move
        // allocations out of unknown region-based control-flow operations.
        if ((!isKnownControlFlowInterface(parentOp) ||
             !state.isLegalPlacement(parentOp)) &&
            !shouldHoist)
          break;
        shouldHoist = false;

        // Move to our parent block by notifying the current StateT
        // implementation.
        currentBlock = parentBlock;
        state.recordMoveToParent(currentBlock);
      }
    }
    // Return the finally determined placement block.
    return state.placementBlock;
  }

  /// The dominator info to find the appropriate start operation to move the
  /// allocs.
  DominanceInfo dominators;

  /// The map storing the final placement blocks of a given alloc value.
  llvm::DenseMap<Value, Block *> placementBlocks;

  /// The operation that this transformation is working on. It is used to also
  /// gather allocas.
  Operation *scopeOp;
};

static LogicalResult hoistBuffersFromNestedParallelLoop(Operation *op) {
  BufferHoisting<BufferHoistingState> optimizer(op);
  return optimizer.hoist();
};

class BufferNestedParallelLoopHoisting
    : public impl::BufferNestedParallelLoopHoistingBase<
          BufferNestedParallelLoopHoisting> {
public:
  friend struct PassHelper;
  using impl::BufferNestedParallelLoopHoistingBase<
      BufferNestedParallelLoopHoisting>::BufferNestedParallelLoopHoistingBase;

  void runOnOperation() final {
    auto op = getOperation();
    if (failed(hoistBuffersFromNestedParallelLoop(op)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createBufferNestedParallelLoopHoistingPass() {
  return std::make_unique<BufferNestedParallelLoopHoisting>();
}

} // namespace mlir::gc
