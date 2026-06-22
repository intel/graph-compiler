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

/// Determine whether a scf.for loop has at most one iteration.
///
/// Pattern 1 (pre-canonicalization):
///   lb = affine.apply <()[s0, s1, s2] -> (s1 - (s1 - s0) % s2)>()[_, ub, step]
///
/// Pattern 2 (post-canonicalization with constant step):
///   lb = affine.apply <()[s0] -> ((s0 floordiv C) * C)>()[ub], step = C
///
/// Pattern 3: all constants and ub - lb <= step.
static bool isAtMostOneIteration(scf::ForOp forOp) {
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();

  auto lbConst = getConstantIndex(lb);
  auto ubConst = getConstantIndex(ub);
  auto stepConst = getConstantIndex(step);
  if (lbConst && ubConst && stepConst) {
    int64_t diff = *ubConst - *lbConst;
    if (diff >= 0 && diff <= *stepConst) return true;
  }

  auto affineApply = lb.getDefiningOp<affine::AffineApplyOp>();
  if (!affineApply) return false;

  AffineMap map = affineApply.getAffineMap();
  if (map.getNumResults() != 1) return false;

  // Pattern 1: map = s1 - (s1 - s0) % s2
  if (map.getNumDims() == 0 && map.getNumSymbols() == 3) {
    AffineExpr s0 = getAffineSymbolExpr(0, forOp.getContext());
    AffineExpr s1 = getAffineSymbolExpr(1, forOp.getContext());
    AffineExpr s2 = getAffineSymbolExpr(2, forOp.getContext());
    AffineExpr expected = s1 - (s1 - s0) % s2;
    if (map.getResult(0) == expected) {
      auto operands = affineApply.getMapOperands();
      if (operands.size() == 3 && operands[1] == ub && operands[2] == step)
        return true;
    }
  }

  // Pattern 2: lb = (ub floordiv C) * C, step = C
  if (!stepConst || *stepConst <= 0) return false;
  int64_t C = *stepConst;

  auto operands = affineApply.getMapOperands();
  if (operands.size() != 1) return false;

  AffineExpr var;
  if (map.getNumDims() == 1 && map.getNumSymbols() == 0)
    var = getAffineDimExpr(0, forOp.getContext());
  else if (map.getNumDims() == 0 && map.getNumSymbols() == 1)
    var = getAffineSymbolExpr(0, forOp.getContext());
  else return false;

  AffineExpr expected = (var.floorDiv(C)) * C;
  if (map.getResult(0) == expected && operands[0] == ub) return true;

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
