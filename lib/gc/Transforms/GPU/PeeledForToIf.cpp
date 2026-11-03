//===-- PeeledForToIf.cpp - Convert at-most-one-trip for to if --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "peeled-for-to-if"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_PEELEDFORTOIF
#define GEN_PASS_DEF_PEELEDFORTOIF
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();
  return std::nullopt;
}

/// Strip index casts inserted by loop peeling for non-index loops.
static Value skipIndexCasts(Value v) {
  while (auto cast = v.getDefiningOp<arith::IndexCastOp>()) v = cast.getIn();
  return v;
}

/// Determine whether `map`, applied to `operands`, produces the peeling split
/// bound of a loop with the given `ub` and `step`, i.e.
///
///   split = ub - (ub - x) mod step
///
/// for any `x`. For a multi-result map (affine.max) one matching result is
/// enough: the maximum is greater than or equal to each of its results, so
/// ub - max(...) <= ub - split < step.
static bool isSplitBoundMap(AffineMap map, ValueRange operands, Value ub,
                            Value step, std::optional<int64_t> stepConst,
                            MLIRContext *ctx) {
  unsigned numDims = map.getNumDims();
  auto exprOf = [&](unsigned i) {
    return i < numDims ? getAffineDimExpr(i, ctx)
                       : getAffineSymbolExpr(i - numDims, ctx);
  };

  SmallVector<AffineExpr> ubExprs, stepExprs, allExprs;
  for (auto [i, operand] : llvm::enumerate(operands)) {
    AffineExpr expr = exprOf(i);
    allExprs.push_back(expr);
    if (skipIndexCasts(operand) == ub) ubExprs.push_back(expr);
    if (skipIndexCasts(operand) == step) stepExprs.push_back(expr);
  }
  // A constant step is folded into the map instead of being an operand.
  if (stepConst && *stepConst > 0)
    stepExprs.push_back(getAffineConstantExpr(*stepConst, ctx));

  for (AffineExpr result : map.getResults()) {
    for (AffineExpr ubExpr : ubExprs) {
      for (AffineExpr stepExpr : stepExprs) {
        // ub - (ub - x) mod step, x an operand of the map.
        for (AffineExpr xExpr : allExprs)
          if (result == ubExpr - (ubExpr - xExpr) % stepExpr) return true;
        // Same with a constant lb = 0 folded into the map, plus the
        // post-canonicalization spelling of it.
        if (result == ubExpr - ubExpr % stepExpr) return true;
        if (auto stepValue = dyn_cast<AffineConstantExpr>(stepExpr))
          if (result ==
              ubExpr.floorDiv(stepValue.getValue()) * stepValue.getValue())
            return true;
      }
    }
  }
  return false;
}

/// Determine whether a scf.for loop has at most one iteration.
///
/// All constants: ub - lb <= step.
///
/// Otherwise the loop must be the partial iteration of a peeled loop, whose
/// lower bound is the split bound `ub - (ub - lb) mod step`, possibly clamped
/// to lb. `scf::peelForLoopAndSimplifyBounds` has spelled that bound in
/// several ways over time, all of which are accepted here:
///
///   affine.apply <()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>()[lb, ub, step]
///   affine.max   <()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2, s0)>()[lb, ub,
///   step] affine.apply/affine.max with a constant step folded into the map,
///   e.g.
///     <()[s0] -> ((s0 floordiv C) * C)>()[ub] with step = C
///   arith: ub - ((ub - min(lb, ub)) remui step)
static bool isAtMostOneIteration(scf::ForOp forOp) {
  Value lb = skipIndexCasts(forOp.getLowerBound());
  Value ub = skipIndexCasts(forOp.getUpperBound());
  Value step = skipIndexCasts(forOp.getStep());

  auto lbConst = getConstantIndex(lb);
  auto ubConst = getConstantIndex(ub);
  auto stepConst = getConstantIndex(step);
  if (lbConst && ubConst && stepConst) {
    int64_t diff = *ubConst - *lbConst;
    if (diff >= 0 && diff <= *stepConst) return true;
  }

  MLIRContext *ctx = forOp.getContext();
  if (auto applyOp = lb.getDefiningOp<affine::AffineApplyOp>())
    return isSplitBoundMap(applyOp.getAffineMap(), applyOp.getMapOperands(), ub,
                           step, stepConst, ctx);

  if (auto maxOp = lb.getDefiningOp<affine::AffineMaxOp>())
    return isSplitBoundMap(maxOp.getAffineMap(), maxOp.getMapOperands(), ub,
                           step, stepConst, ctx);

  // arith spelling: lb = ub - ((ub - min(lb, ub)) remui step).
  if (auto subOp = lb.getDefiningOp<arith::SubIOp>()) {
    if (skipIndexCasts(subOp.getLhs()) != ub) return false;
    auto remOp = skipIndexCasts(subOp.getRhs()).getDefiningOp<arith::RemUIOp>();
    if (!remOp || skipIndexCasts(remOp.getRhs()) != step) return false;
    auto innerSub =
        skipIndexCasts(remOp.getLhs()).getDefiningOp<arith::SubIOp>();
    return innerSub && skipIndexCasts(innerSub.getLhs()) == ub;
  }

  return false;
}

static void convertForToIf(scf::ForOp forOp, IRRewriter &rewriter) {
  Location loc = forOp.getLoc();
  rewriter.setInsertionPoint(forOp);

  Value cond =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                            forOp.getLowerBound(), forOp.getUpperBound());

  auto ifOp = scf::IfOp::create(
      rewriter, loc, cond,
      [&](OpBuilder &builder, Location loc) {
        IRMapping mapping;
        mapping.map(forOp.getInductionVar(), forOp.getLowerBound());
        for (auto [blockArg, initArg] :
             llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs()))
          mapping.map(blockArg, initArg);

        for (Operation &op : forOp.getBody()->without_terminator())
          builder.clone(op, mapping);

        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        SmallVector<Value> yieldValues;
        for (Value v : yieldOp.getOperands())
          yieldValues.push_back(mapping.lookupOrDefault(v));
        scf::YieldOp::create(builder, loc, yieldValues);
      },
      [&](OpBuilder &builder, Location loc) {
        scf::YieldOp::create(builder, loc, forOp.getInitArgs());
      });

  rewriter.replaceOp(forOp, ifOp.getResults());
}

struct PeeledForToIf final : gc::impl::PeeledForToIfBase<PeeledForToIf> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    SmallVector<scf::ForOp> candidates;
    funcOp->walk([&](scf::ForOp forOp) {
      if (isAtMostOneIteration(forOp)) candidates.push_back(forOp);
    });

    for (scf::ForOp forOp : candidates) convertForToIf(forOp, rewriter);
  }
};

} // namespace
