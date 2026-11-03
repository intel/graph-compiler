//===-- GuardEmptyTiles.cpp - Skip empty work-group tiles -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::gc {
#define GEN_PASS_DECL_GUARDEMPTYTILES
#define GEN_PASS_DEF_GUARDEMPTYTILES
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

static bool isAtLeastOne(const ValueBoundsConstraintSet::Variable &var) {
  FailureOr<int64_t> lb = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, var);
  return succeeded(lb) && *lb >= 1;
}

/// Whether `v` is provably at least 1. The value bounds analysis only records
/// upper bounds for a minimum, so the clamped tile sizes produced by tiling,
/// affine.min(tile, ub - iv), are handled term by term here.
static bool isProvablyPositive(Value v) {
  if (auto min = v.getDefiningOp<affine::AffineMinOp>()) {
    AffineMap map = min.getAffineMap();
    SmallVector<Value> operands(min.getMapOperands());
    return llvm::all_of(map.getResults(), [&](AffineExpr term) {
      AffineMap termMap =
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), term);
      return isAtLeastOne({termMap, operands});
    });
  }
  if (auto min = v.getDefiningOp<arith::MinSIOp>())
    return isProvablyPositive(min.getLhs()) && isProvablyPositive(min.getRhs());
  if (auto max = v.getDefiningOp<arith::MaxSIOp>())
    return isProvablyPositive(max.getLhs()) || isProvablyPositive(max.getRhs());
  return isAtLeastOne({v});
}

/// A dynamic tile size that cannot be shown to be at least 1.
static bool mayBeZero(OpFoldResult size) {
  auto value = dyn_cast<Value>(size);
  return value && !isProvablyPositive(value);
}

/// Whether two tile sizes are the same: the same value, the same side-effect
/// free computation on the same operands (the condition CSE merges under; an
/// op that reads memory may see different values), or equal by the value
/// bounds analysis.
static bool isSameSize(Value a, Value b) {
  if (a == b) return true;
  Operation *defA = a.getDefiningOp(), *defB = b.getDefiningOp();
  if (defA && defB && defA->getNumResults() == 1 &&
      defB->getNumResults() == 1 && isMemoryEffectFree(defA) &&
      isMemoryEffectFree(defB) &&
      OperationEquivalence::isEquivalentTo(
          defA, defB,
          [](Value x, Value y) { return success(isSameSize(x, y)); },
          /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations))
    return true;
  FailureOr<bool> equal = ValueBoundsConstraintSet::areEqual({a}, {b});
  return succeeded(equal) && *equal;
}

/// Adds `op` and the ops of `block` it transitively depends on to `chain`.
/// Fails if any of them has memory effects, as the chain is moved ahead of
/// the guard.
static LogicalResult collectIndexChain(Operation *op, Block *block,
                                       SetVector<Operation *> &chain) {
  BackwardSliceOptions opts;
  opts.inclusive = true;
  opts.filter = [block](Operation *o) { return o->getBlock() == block; };
  SetVector<Operation *> slice;
  if (failed(getBackwardSlice(op, &slice, opts))) return failure();
  for (Operation *o : slice)
    if (!isMemoryEffectFree(o)) return failure();
  chain.insert_range(slice);
  return success();
}

/// Wraps the body of `forall` in `scf.if` when one of its output tiles has a
/// dynamic size that may be zero:
///
///   scf.forall (...) shared_outs(%out = ...) {
///     <tile position: offsets, sizes>
///     %r = scf.if <all sizes > 0> -> (tensor<?x64xf16>) {
///       <the whole tile: slices, fills, producers, matmul, consumers>
///       scf.yield %tile
///     } else {
///       %same = tensor.extract_slice %out[offsets] [sizes] [1, 1]
///       scf.yield %same
///     }
///     scf.forall.in_parallel {
///       tensor.parallel_insert_slice %r into %out[offsets] [sizes] [1, 1]
///     }
///   }
///
/// The guard covers all computations fused into the tile. Skipping the body
/// is only correct when every output tile is empty under the same condition
/// and the body has no side effects; otherwise the forall is left unchanged
/// with a warning.
static void guardForall(scf::ForallOp forall) {
  Block *body = forall.getBody();
  Operation *terminator = forall.getTerminator();

  // The sizes that may be zero, one representative per distinct size, and
  // the ones each output tile has.
  SmallVector<tensor::ParallelInsertSliceOp> inserts;
  SmallVector<Value> sizes;
  SmallVector<SmallVector<Value>> tileSizes;
  for (Operation &op : forall.getTerminator().getYieldingOps()) {
    auto ins = dyn_cast<tensor::ParallelInsertSliceOp>(op);
    if (!ins) return;
    inserts.push_back(ins);
    SmallVector<Value> &own = tileSizes.emplace_back();
    for (OpFoldResult size : ins.getMixedSizes()) {
      if (!mayBeZero(size)) continue;
      Value v = cast<Value>(size);
      own.push_back(v);
      if (llvm::none_of(sizes, [&](Value s) { return isSameSize(s, v); }))
        sizes.push_back(v);
    }
  }
  if (sizes.empty()) return;

  // The body is skipped when any of the sizes is zero, so every output tile
  // must be empty whenever any one of them is: each tile has to carry each
  // size. Otherwise one empty tile would drop the others' work.
  for (ArrayRef<Value> own : tileSizes) {
    for (Value size : sizes) {
      if (llvm::any_of(own, [&](Value s) { return isSameSize(s, size); }))
        continue;
      forall.emitWarning()
          << "output tiles are empty under different conditions, not guarded";
      return;
    }
  }

  // The condition and the else branch need the tile positions ahead of the
  // guard, so their index arithmetic is kept outside of it.
  SetVector<Operation *> hoisted;
  for (auto ins : inserts) {
    for (Value v : llvm::concat<Value>(ins.getOffsets(), ins.getSizes(),
                                       ins.getStrides())) {
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != body) continue;
      if (failed(collectIndexChain(def, body, hoisted))) {
        forall.emitWarning() << "tile position has side effects, not guarded";
        return;
      }
    }
  }

  // Everything from the first op outside the hoisted chains to the terminator
  // only runs when the tile is non-empty.
  Operation *guardPt = nullptr;
  for (Operation &op : *body) {
    if (&op == terminator) break;
    if (!hoisted.contains(&op)) {
      guardPt = &op;
      break;
    }
  }
  if (!guardPt) return;
  // The guarded ops are skipped for an empty tile, which is only correct when
  // the tile itself is all they do.
  SmallVector<Operation *> toHoist;
  for (Operation *op = guardPt; op != terminator; op = op->getNextNode()) {
    if (hoisted.contains(op)) {
      toHoist.push_back(op);
    } else if (!isMemoryEffectFree(op)) {
      forall.emitWarning() << "tile body has side effects, not guarded";
      return;
    }
  }
  for (Operation *op : toHoist) op->moveBefore(guardPt);

  OpBuilder b(guardPt);
  Location loc = forall.getLoc();
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value cond;
  for (Value size : sizes) {
    Value hasElems =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, size, zero);
    cond = cond ? arith::AndIOp::create(b, loc, cond, hasElems) : hasElems;
  }

  auto types = llvm::map_to_vector(
      inserts, [](tensor::ParallelInsertSliceOp ins) -> Type {
        return ins.getSourceType();
      });
  auto guard = scf::IfOp::create(b, loc, types, cond, /*withElseRegion=*/true);

  Block *thenBlock = guard.thenBlock();
  thenBlock->getOperations().splice(thenBlock->end(), body->getOperations(),
                                    guardPt->getIterator(),
                                    terminator->getIterator());
  b.setInsertionPointToEnd(thenBlock);
  SmallVector<Value> tiles = llvm::map_to_vector(
      inserts, [](tensor::ParallelInsertSliceOp ins) -> Value {
        return ins.getSource();
      });
  scf::YieldOp::create(b, loc, tiles);

  b.setInsertionPointToEnd(guard.elseBlock());
  SmallVector<Value> untouched;
  for (auto ins : inserts)
    untouched.push_back(tensor::ExtractSliceOp::create(
        b, loc, ins.getSourceType(), ins.getDest(), ins.getMixedOffsets(),
        ins.getMixedSizes(), ins.getMixedStrides()));
  scf::YieldOp::create(b, loc, untouched);

  for (auto [ins, res] : llvm::zip_equal(inserts, guard.getResults()))
    ins.getSourceMutable().assign(res);
}

struct GuardEmptyTiles final
    : mlir::gc::impl::GuardEmptyTilesBase<GuardEmptyTiles> {
  void runOnOperation() override {
    SmallVector<scf::ForallOp> foralls;
    getOperation()->walk([&](scf::ForallOp op) { foralls.push_back(op); });
    for (scf::ForallOp forall : foralls) guardForall(forall);
  }
};

} // namespace
