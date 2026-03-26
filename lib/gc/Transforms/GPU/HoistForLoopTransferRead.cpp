//===--------- RemoveAllocs.cpp - Remove unnecessary allocs --------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_HOISTFORLOOPTRANSFERREAD
#define GEN_PASS_DEF_HOISTFORLOOPTRANSFERREAD
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

static bool isConstantZero(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantIndexOp>();
  return cst && cst.value() == 0;
}

static bool isAllZeroIndices(ValueRange indices) {
  for (Value idx : indices)
    if (!isConstantZero(idx))
      return false;
  return true;
}

static bool tensorVectorShapesMatch(RankedTensorType tTy, VectorType vTy) {
  if (tTy.getRank() != (int64_t)vTy.getRank())
    return false;
  for (int64_t i = 0; i < tTy.getRank(); ++i) {
    if (tTy.isDynamicDim(i) || vTy.isDynamicDim(i))
      return false; // keep conservative
    if (tTy.getDimSize(i) != vTy.getDimSize(i))
      return false;
  }
  return tTy.getElementType() == vTy.getElementType();
}

/// Conservative "full transfer_read":
/// - source is the given tensor value
/// - no mask
/// - indices are all constant 0
/// - vector type matches tensor shape and element type
static bool isFullTransferReadFrom(vector::TransferReadOp readOp,
                                   Value tensor) {
  if (readOp.getBase() != tensor)
    return false;

  if (readOp.getMask())
    return false;

  if (!isAllZeroIndices(readOp.getIndices()))
    return false;

  auto tTy = dyn_cast<RankedTensorType>(tensor.getType());
  auto vTy = dyn_cast<VectorType>(readOp.getVectorType());
  if (!tTy || !vTy)
    return false;

  return tensorVectorShapesMatch(tTy, vTy);
}

/// Conservative "full transfer_write":
/// - destination is the given tensor value
/// - no mask
/// - indices are all constant 0
/// - vector type matches tensor shape and element type
static bool isFullTransferWriteTo(vector::TransferWriteOp writeOp,
                                  Value tensor) {
  if (writeOp.getMask())
    return false;

  if (!isAllZeroIndices(writeOp.getIndices()))
    return false;

  auto tTy = dyn_cast<RankedTensorType>(tensor.getType());
  auto vTy = dyn_cast<VectorType>(writeOp.getVector().getType());
  if (!tTy || !vTy)
    return false;

  return tensorVectorShapesMatch(tTy, vTy);
}

/// Try to optimize a single tensor iter-arg:
/// - tensor iter-arg is only read via full transfer_read
/// - yielded tensor is produced by full transfer_write into that iter-arg
/// If successful, rewrite loop to carry vector instead, and reconstruct tensor
/// after the loop.
static LogicalResult tryHoistTensorAsVector(scf::ForOp forOp,
                                            unsigned iterArgIdx,
                                            PatternRewriter &rewriter) {
  // --- Basic sanity checks ---
  auto oldIterArgs = forOp.getRegionIterArgs();
  if (iterArgIdx >= oldIterArgs.size())
    return failure();

  Value tensorIterArg = oldIterArgs[iterArgIdx];
  auto tensorTy = dyn_cast<RankedTensorType>(tensorIterArg.getType());
  if (!tensorTy)
    return failure(); // only ranked tensors in this conservative impl

  // The corresponding yield operand is "carried value" for this iter_arg.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yieldedTensor = yieldOp.getResults()[iterArgIdx];

  // Must be produced by a transfer_write.
  auto writeOp = yieldedTensor.getDefiningOp<vector::TransferWriteOp>();
  if (!writeOp)
    return failure();

  // Must write fully into the tensor iter-arg.
  if (!isFullTransferWriteTo(writeOp, tensorIterArg))
    return failure();

  // Ensure the produced tensor from transfer_write is not used elsewhere
  // (besides yield), otherwise we would need to keep it and semantics become
  // more complex.
  if (!yieldedTensor.hasOneUse())
    return failure();

  // Vector that we want to carry/yield instead of tensor.
  Value vecToYield = writeOp.getVector();
  auto vecTy = dyn_cast<VectorType>(vecToYield.getType());
  if (!vecTy)
    return failure();

  // --- Check that the tensor iter-arg is only read via full transfer_read (and
  // written by that writeOp) --- Allowed uses:
  //  - vector.transfer_read (full) from tensorIterArg
  //  - vector.transfer_write (full) to tensorIterArg (the one that defines
  //  yieldedTensor)
  for (OpOperand &use : tensorIterArg.getUses()) {
    Operation *user = use.getOwner();

    if (auto r = dyn_cast<vector::TransferReadOp>(user)) {
      if (!isFullTransferReadFrom(r, tensorIterArg))
        return failure();
      continue;
    }

    if (auto w = dyn_cast<vector::TransferWriteOp>(user)) {
      // Only allow the specific full write we matched for the yield.
      if (w != writeOp)
        return failure();
      if (!isFullTransferWriteTo(w, tensorIterArg))
        return failure();
      continue;
    }

    // Any other use -> not safe
    return failure();
  }

  // --- Build init vector outside the loop ---
  // Replace init tensor (iter_arg initial value) with
  // transfer_read(initTensor).
  Value initTensor = forOp.getInitArgs()[iterArgIdx];

  // Create constants 0 indices right before the loop (same rank as tensor).
  Location loc = forOp.getLoc();
  rewriter.setInsertionPoint(forOp);

  SmallVector<Value> zeroIdx;
  zeroIdx.reserve(tensorTy.getRank());
  for (int64_t i = 0; i < tensorTy.getRank(); ++i)
    zeroIdx.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
  SmallVector<bool> inBounds;
  for (int64_t i = 0; i < tensorTy.getRank(); ++i)
    inBounds.push_back(true); // we know these are in bounds statically
  Value initVec = vector::TransferReadOp::create(
      rewriter, loc, vecTy, initTensor, zeroIdx, /*padding=*/std::nullopt,
      /*permutationMap*/
      AffineMap::getMultiDimIdentityMap(tensorTy.getRank(), loc.getContext()),
      inBounds);

  // --- Create new scf.for with vector as the iter-arg ---
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  newInitArgs[iterArgIdx] = initVec;

  // New result types: same as old, but tensor result becomes vector result for
  // this slot.
  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(forOp.getNumResults());
  for (unsigned r = 0; r < forOp.getNumResults(); ++r) {
    if (r == iterArgIdx)
      newResultTypes.push_back(vecTy);
    else
      newResultTypes.push_back(forOp.getResultTypes()[r]);
  }

  // Create an empty new loop, we will clone body ops with a mapping.
  auto newFor =
      scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newInitArgs);

  // Build mapping from old block args to new block args.
  Block *oldBody = forOp.getBody();
  Block *newBody = newFor.getBody();

  IRMapping mapping;
  mapping.map(oldBody->getArgument(0),
              newBody->getArgument(0)); // induction var

  // Map iter args: tensor iter arg -> vector iter arg, others stay the same.
  auto oldRegionIterArgs = forOp.getRegionIterArgs();
  auto newRegionIterArgs = newFor.getRegionIterArgs();
  for (unsigned i = 0; i < oldRegionIterArgs.size(); ++i) {
    if (i == iterArgIdx)
      mapping.map(oldRegionIterArgs[i],
                  newRegionIterArgs[i]); // tensor -> vector
    else
      mapping.map(oldRegionIterArgs[i], newRegionIterArgs[i]);
  }

  // Clone body ops (excluding terminator), while:
  //  - replacing full transfer_read from old tensor iter-arg with the new
  //  vector iter-arg
  //  - skipping the matched transfer_write (we will yield its vector operand
  //  instead)
  rewriter.setInsertionPointToStart(newBody);

  for (Operation &op : oldBody->without_terminator()) {
    // Replace transfer_read(tensorIterArg) with the carried vector directly.
    if (auto r = dyn_cast<vector::TransferReadOp>(&op)) {
      if (isFullTransferReadFrom(r, tensorIterArg)) {
        // The read result becomes exactly the carried vector iter arg.
        mapping.map(r.getResult(), newRegionIterArgs[iterArgIdx]);
        continue; // do not clone this op
      }
    }

    // Skip the matched transfer_write that only exists to produce
    // yieldedTensor.
    if (&op == writeOp.getOperation()) {
      // We *don't* need to map its tensor result because we required
      // yieldedTensor.hasOneUse() (only used by scf.yield). If you relax this,
      // you must handle extra uses.
      continue;
    }

    // Default: clone op with mapping.
    rewriter.clone(op, mapping);
  }

  // Create new yield: for iterArgIdx yield vecToYield (mapped), others yield
  // mapped old operands.
  auto oldYieldOperands = yieldOp.getResults();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYieldOperands.size());

  for (unsigned i = 0; i < oldYieldOperands.size(); ++i) {
    if (i == iterArgIdx) {
      // Yield the vector that would have been written into the tensor.
      newYieldOperands.push_back(mapping.lookup(vecToYield));
    } else {
      newYieldOperands.push_back(mapping.lookup(oldYieldOperands[i]));
    }
  }

  rewriter.setInsertionPointToEnd(newBody);
  scf::YieldOp::create(rewriter, loc, newYieldOperands);

  // --- Reconstruct the tensor result outside the loop ---
  // We must preserve original loop result types for users: tensor result is
  // recreated.
  rewriter.setInsertionPointAfter(newFor);

  Value newVecResult = newFor.getResult(iterArgIdx);
  Value reconstructedTensor =
      vector::TransferWriteOp::create(rewriter, loc, newVecResult, initTensor,
                                      zeroIdx)
          .getResult();

  // Replace old loop results:
  // - tensor result slot replaced by reconstructed tensor
  // - other result slots replaced by corresponding newFor result
  SmallVector<Value> replacements;
  replacements.reserve(forOp.getNumResults());

  for (unsigned r = 0; r < forOp.getNumResults(); ++r) {
    if (r == iterArgIdx)
      replacements.push_back(reconstructedTensor);
    else
      replacements.push_back(newFor.getResult(r));
  }

  rewriter.replaceOp(forOp, replacements);
  return success();
}

struct ForOpHoistTensorToVectorPattern final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Try to apply for any tensor iter-arg; succeed on first successful
    // rewrite.
    auto iterArgs = forOp.getRegionIterArgs();
    for (unsigned i = 0; i < iterArgs.size(); ++i) {
      if (!isa<RankedTensorType>(iterArgs[i].getType()))
        continue;

      if (succeeded(tryHoistTensorAsVector(forOp, i, rewriter)))
        return success();
    }
    return failure();
  }
};

struct HoistForLoopTransferRead final
    : gc::impl::HoistForLoopTransferReadBase<HoistForLoopTransferRead> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForOpHoistTensorToVectorPattern>(ctx);

    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

} // namespace
