//===- FactorDequant.cpp - Apply dequant scales per output ---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A K loop step that contracts with dequantized weights
//
//   %q    = arith.sitofp (arith.subi (arith.extui %w), %z)
//   %b    = arith.mulf %q, %s
//   %acc' = vector.contract %x, %b, %acc      // %acc is the loop-carried value
//
// where the scale %s and the zero point %z do not change along the
// contraction dimension k computes, for every output,
//
//   sum_k x[m,k] * (w[n,k] - z[n]) * s[n]
//     = s[n] * sum_k x[m,k] * (w[n,k] - z[n])
//
// This pass applies the scale once per output instead of once per weight:
//
//   %p    = vector.contract %x, %q, 0.0       // f32
//   %out' = math.fma (broadcast %s), %p, %out // f32 loop-carried value
//
// The loop result is converted back to the original accumulator type.
//
// This changes the rounding, so results differ from the input program. The
// product (w - z) * s is no longer rounded to the weight type, the sum over K
// steps is kept in f32, and the scale is applied with an fma.
//
// With `fold-zero-point`, the zero point also leaves the weights. For an
// unsigned weight w < 512, the f16 value with the bits `w | 0x6600` is exactly
// 1536 + w. So the weights are built with one integer `or`, and the
// contraction starts at `vector.contract %x, -(1536 + z)` instead of zero:
//
//   %start = vector.contract %x, (broadcast -(1536 + z)), 0.0
//   %p     = vector.contract %x, bitcast(extui(%w) | 0x6600), %start
//
// The biased weights and the offsets -(1536 + z) are exact f16 values, but the
// f32 sums still round. They now carry the offset 1536 * sum_k x, which
// cancels at the end, so the error grows with sum_k |x| instead of with the
// result, and an infinite x gives NaN.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_FACTORDEQUANT
#define GEN_PASS_DEF_FACTORDEQUANT
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Returns `v` with dimension `dim` reduced to size 1, for a `v` whose value
// does not change along `dim`. Returns null when that is not known. With a
// null `rewriter`, only checks and returns `v` on success.
Value collapseAlong(RewriterBase *rewriter, Value v, int64_t dim) {
  auto ty = dyn_cast<VectorType>(v.getType());
  if (!ty || ty.isScalable() || dim < 0 || dim >= ty.getRank()) return {};
  if (ty.getDimSize(dim) == 1) return v;
  SmallVector<int64_t> shape(ty.getShape());
  shape[dim] = 1;
  auto newTy = VectorType::get(shape, ty.getElementType());

  DenseElementsAttr cst;
  if (matchPattern(v, m_Constant(&cst))) {
    if (!cst.isSplat()) return {};
    if (!rewriter) return v;
    OpBuilder::InsertionGuard guard(*rewriter);
    rewriter->setInsertionPointAfterValue(v);
    return arith::ConstantOp::create(*rewriter, v.getLoc(), newTy,
                                     cst.resizeSplat(newTy));
  }

  Operation *def = v.getDefiningOp();
  if (!def) return {};
  Location loc = def->getLoc();
  // The new value is built next to the old one.
  auto build = [&](auto &&fn) -> Value {
    if (!rewriter) return v;
    OpBuilder::InsertionGuard guard(*rewriter);
    rewriter->setInsertionPointAfter(def);
    return fn();
  };

  // A read that broadcasts along `dim`.
  if (auto read = dyn_cast<vector::TransferReadOp>(def)) {
    // Trailing dims past the transfer rank come from vector elements.
    if (dim >= static_cast<int64_t>(read.getTransferRank()) || read.getMask())
      return {};
    auto expr =
        dyn_cast<AffineConstantExpr>(read.getPermutationMap().getResult(dim));
    if (!expr) return {};
    return build([&] {
      return vector::TransferReadOp::create(
          *rewriter, loc, newTy, read.getBase(), read.getIndices(),
          read.getPadding(), read.getPermutationMapAttr(),
          read.getInBoundsAttr());
    });
  }

  if (auto bcast = dyn_cast<vector::BroadcastOp>(def)) {
    Value src = bcast.getSource();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    int64_t lead = ty.getRank() - (srcTy ? srcTy.getRank() : 0);
    // A new leading dimension, or a unit one stretched, is constant.
    if (dim >= lead) {
      src = collapseAlong(rewriter, src, dim - lead);
      if (!src) return {};
    }
    return build([&] {
      return vector::BroadcastOp::create(*rewriter, loc, newTy, src);
    });
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(def)) {
    ArrayRef<int64_t> perm = transpose.getPermutation();
    Value src = collapseAlong(rewriter, transpose.getVector(), perm[dim]);
    if (!src) return {};
    return build(
        [&] { return vector::TransposeOp::create(*rewriter, loc, src, perm); });
  }

  // A cast that only adds or drops unit dimensions: `dim` is the j-th
  // non-unit dimension of both shapes.
  if (auto cast = dyn_cast<vector::ShapeCastOp>(def)) {
    VectorType srcTy = cast.getSourceVectorType();
    auto nonUnit = [](ArrayRef<int64_t> s) {
      SmallVector<int64_t> dims;
      for (auto [i, d] : llvm::enumerate(s))
        if (d != 1) dims.push_back(i);
      return dims;
    };
    SmallVector<int64_t> resDims = nonUnit(ty.getShape());
    SmallVector<int64_t> srcDims = nonUnit(srcTy.getShape());
    if (resDims.size() != srcDims.size()) return {};
    for (auto [r, s] : llvm::zip(resDims, srcDims))
      if (ty.getDimSize(r) != srcTy.getDimSize(s)) return {};
    int64_t j = llvm::find(resDims, dim) - resDims.begin();
    Value src = collapseAlong(rewriter, cast.getSource(), srcDims[j]);
    if (!src) return {};
    return build([&] {
      return vector::ShapeCastOp::create(*rewriter, loc, newTy, src);
    });
  }

  // Elementwise operations of values that are constant along `dim`.
  if (OpTrait::hasElementwiseMappableTraits(def) && def->getNumResults() == 1 &&
      def->getNumRegions() == 0) {
    SmallVector<Value> operands;
    for (Value operand : def->getOperands()) {
      auto operandTy = dyn_cast<VectorType>(operand.getType());
      if (!operandTy || operandTy.getShape() != ty.getShape()) return {};
      Value c = collapseAlong(rewriter, operand, dim);
      if (!c) return {};
      operands.push_back(c);
    }
    return build([&] {
      OperationState state(loc, def->getName(), operands, {newTy},
                           def->getAttrs());
      return rewriter->create(state)->getResult(0);
    });
  }
  return {};
}

// The vector that `read` gets from the whole-vector `vector.transfer_write`
// that produced its tensor, when `read` reads exactly that vector without its
// leading unit dimensions. Returns null otherwise.
Value forwardedWrite(vector::TransferReadOp read) {
  auto write = read.getBase().getDefiningOp<vector::TransferWriteOp>();
  if (!write || !write.getResult() || read.getMask() || write.getMask())
    return {};
  if (!llvm::equal(read.getIndices(), write.getIndices())) return {};
  if (!read.getPermutationMap().isMinorIdentity() ||
      !write.getPermutationMap().isMinorIdentity())
    return {};
  if (llvm::is_contained(read.getInBoundsValues(), false) ||
      llvm::is_contained(write.getInBoundsValues(), false))
    return {};
  VectorType wTy = write.getVectorType(), rTy = read.getVectorType();
  // Scalable sizes are not compared below.
  if (wTy.isScalable() || rTy.isScalable()) return {};
  int64_t lead = wTy.getRank() - rTy.getRank();
  if (lead < 0 || wTy.getElementType() != rTy.getElementType() ||
      wTy.getShape().drop_front(lead) != rTy.getShape() ||
      !llvm::all_of(
          wTy.getShape().take_front(lead), [](int64_t d) { return d == 1; }))
    return {};
  return write.getVector();
}

// Erases the defining op of `v` and the ops feeding it, as far as they have
// become dead.
void eraseDeadChain(RewriterBase &rewriter, Value v) {
  // A set, so an op that feeds two dead ops is queued and erased once.
  llvm::SetVector<Operation *> worklist;
  if (Operation *def = v.getDefiningOp()) worklist.insert(def);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op)) continue;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp()) worklist.insert(def);
    rewriter.eraseOp(op);
  }
}

unsigned bitWidth(Value v) {
  return getElementTypeOrSelf(v).getIntOrFloatBitWidth();
}

// The `arith.extui` that `v` comes from, through operations that only move
// elements around. Null if there is none.
arith::ExtUIOp extensionOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto ext = dyn_cast<arith::ExtUIOp>(def)) return ext;
    if (!isa<vector::TransposeOp, vector::BroadcastOp, vector::ShapeCastOp>(
            def))
      return {};
    v = def->getOperand(0);
  }
  return {};
}

// Rewrites one contraction; see the file comment.
LogicalResult factor(RewriterBase &rewriter, vector::ContractionOp contract,
                     bool foldZeroPoint) {
  MLIRContext *ctx = contract.getContext();
  // The contraction accumulates into a value carried by its loop.
  if (contract.getKind() != vector::CombiningKind::ADD) return failure();
  auto acc = dyn_cast<BlockArgument>(contract.getAcc());
  auto loop =
      acc ? dyn_cast<scf::ForOp>(acc.getOwner()->getParentOp()) : scf::ForOp();
  if (!loop || acc.getOwner() != loop.getBody() || acc.getArgNumber() == 0 ||
      !acc.hasOneUse() || !contract->hasOneUse())
    return failure();
  unsigned idx = acc.getArgNumber() - loop.getNumInductionVars();
  if (loop.getBody()->getTerminator()->getOperand(idx) != contract.getResult())
    return failure();
  auto accTy = dyn_cast<VectorType>(contract.getAccType());
  // The loop value is carried in f32, and the scale is extended to f32.
  auto supported = [](Type t) { return t.isF16() || t.isBF16() || t.isF32(); };
  if (!accTy || !supported(accTy.getElementType()) ||
      !supported(contract.getRhsType().getElementType()))
    return failure();

  // One reduction dimension k, a rank-2 right-hand side (n, k) or (k, n).
  SmallVector<vector::IteratorType> iters = contract.getIteratorTypesArray();
  if (llvm::count(iters, vector::IteratorType::reduction) != 1)
    return failure();
  unsigned kIter =
      llvm::find(iters, vector::IteratorType::reduction) - iters.begin();
  SmallVector<AffineMap> maps = contract.getIndexingMapsArray();
  AffineMap rhsMap = maps[1], accMap = maps[2];
  VectorType rhsTy = contract.getRhsType();
  if (rhsTy.getRank() != 2 || !rhsMap.isProjectedPermutation() ||
      !accMap.isProjectedPermutation())
    return failure();
  std::optional<unsigned> kPos =
      rhsMap.getResultPosition(getAffineDimExpr(kIter, ctx));
  if (!kPos) return failure();
  std::optional<unsigned> nInAcc =
      accMap.getResultPosition(rhsMap.getResult(1 - *kPos));
  if (!nInAcc) return failure();

  // The right-hand side, seen through a rank-reducing round trip or cast:
  // `b` has the same shape plus leading unit dimensions, k is its `kDim`.
  Value b = contract.getRhs();
  if (auto read = b.getDefiningOp<vector::TransferReadOp>())
    b = forwardedWrite(read);
  else if (auto cast = b.getDefiningOp<vector::ShapeCastOp>())
    b = cast.getSource();
  if (!b) return failure();
  auto bTy = cast<VectorType>(b.getType());
  int64_t lead = bTy.getRank() - rhsTy.getRank();
  if (lead < 0 || bTy.getShape().drop_front(lead) != rhsTy.getShape())
    return failure();
  int64_t kDim = lead + *kPos;

  // b = q * s, with s constant along k.
  auto mul = b.getDefiningOp<arith::MulFOp>();
  if (!mul) return failure();
  Value q, s;
  for (int i : {0, 1}) {
    if (collapseAlong(nullptr, mul->getOperand(i), kDim)) {
      s = mul->getOperand(i);
      q = mul->getOperand(1 - i);
      break;
    }
  }
  if (!s) return failure();

  // q = sitofp(extui(w) - z) with z constant along k, or q = itofp(w).
  Value qi;
  if (auto c = q.getDefiningOp<arith::SIToFPOp>()) qi = c.getIn();
  else if (auto c = q.getDefiningOp<arith::UIToFPOp>()) qi = c.getIn();
  if (!qi) return failure();
  Value z;
  if (auto sub = qi.getDefiningOp<arith::SubIOp>();
      sub && q.getDefiningOp<arith::SIToFPOp>()) {
    if (!collapseAlong(nullptr, sub.getRhs(), kDim)) return failure();
    z = sub.getRhs();
    qi = sub.getLhs();
  }
  // The magic number needs exact f16 values 1536 + w and 1536 + z.
  auto wExt = qi.getDefiningOp<arith::ExtUIOp>();
  arith::ExtUIOp zExt = z ? extensionOf(z) : arith::ExtUIOp();
  bool magic = foldZeroPoint && wExt && zExt && bitWidth(wExt.getIn()) <= 9 &&
               bitWidth(zExt.getIn()) <= 9 && getElementTypeOrSelf(q).isF16() &&
               getElementTypeOrSelf(contract.getLhs()).isF16();

  // Carry the loop value in f32.
  Type f32 = rewriter.getF32Type();
  auto acc32Ty = accTy.clone(f32);
  Value acc32 = acc;
  Operation *yieldCast = nullptr;
  if (!accTy.getElementType().isF32()) {
    rewriter.setInsertionPoint(loop);
    Value init32 =
        arith::ExtFOp::create(rewriter, loop.getLoc(), acc32Ty,
                              loop.getInitArgs()[idx], /*fastmath=*/{});
    auto castFn = [](OpBuilder &bld, Location loc, Type to,
                     Value from) -> Value {
      if (bitWidth(from) > getElementTypeOrSelf(to).getIntOrFloatBitWidth())
        return arith::TruncFOp::create(bld, loc, to, from);
      return arith::ExtFOp::create(bld, loc, to, from, /*fastmath=*/{});
    };
    SmallVector<Value> results = scf::replaceAndCastForOpIterArg(
        rewriter, loop, loop.getInitArgsMutable()[idx], init32, castFn);
    rewriter.replaceOp(loop, results);
    // The contraction now reads truncf(arg) and its result goes to an extf.
    acc32 = contract.getAcc().getDefiningOp<arith::TruncFOp>().getIn();
    yieldCast = *contract->getUsers().begin();
  }

  rewriter.setInsertionPoint(contract);
  Location loc = contract.getLoc();
  auto toRhs = [&](Value v) -> Value {
    auto ty = rhsTy.clone(getElementTypeOrSelf(v));
    if (v.getType() == ty) return v;
    return vector::ShapeCastOp::create(rewriter, loc, ty, v);
  };
  auto contractWith = [&](Value rhs, Value start) -> Value {
    return vector::ContractionOp::create(rewriter, loc, contract.getLhs(), rhs,
                                         start, contract.getIndexingMaps(),
                                         contract.getIteratorTypes(),
                                         vector::CombiningKind::ADD);
  };
  // A value that is constant along k as a 1xN row. A shape_cast, unlike a
  // transpose, keeps the read it comes from contiguous.
  int64_t n = rhsTy.getDimSize(1 - *kPos);
  auto rowOf = [&](Value v) -> Value {
    Value c = collapseAlong(&rewriter, v, kDim);
    return vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get({1, n}, getElementTypeOrSelf(c)), c);
  };
  Value start = arith::ConstantOp::create(
      rewriter, loc, acc32Ty,
      DenseElementsAttr::get(acc32Ty, rewriter.getF32FloatAttr(0.0f)));
  Value newB;
  if (!magic) {
    newB = toRhs(q);
  } else {
    Type f16 = rewriter.getF16Type();
    auto i16Ty = bTy.clone(rewriter.getI16Type());
    Value w16 = arith::ExtUIOp::create(rewriter, loc, i16Ty, wExt.getIn());
    Value bias = arith::ConstantOp::create(
        rewriter, loc, i16Ty, DenseElementsAttr::get(i16Ty, APInt(16, 0x6600)));
    Value bits = arith::OrIOp::create(rewriter, loc, w16, bias);
    newB = toRhs(arith::BitcastOp::create(rewriter, loc, bTy.clone(f16), bits));
    // -(1536 + z) per column, then broadcast along k into the (n, k) or
    // (k, n) shape of the right-hand side.
    auto rowTy = VectorType::get({1, n}, f16);
    Value zf = arith::UIToFPOp::create(rewriter, loc, rowTy, rowOf(z));
    Value off = arith::ConstantOp::create(
        rewriter, loc, rowTy,
        DenseElementsAttr::get(rowTy, rewriter.getFloatAttr(f16, 1536.0)));
    Value zNeg = arith::NegFOp::create(
        rewriter, loc, arith::AddFOp::create(rewriter, loc, zf, off));
    int64_t k = rhsTy.getDimSize(*kPos);
    Value bz = vector::BroadcastOp::create(rewriter, loc,
                                           VectorType::get({k, n}, f16), zNeg);
    if (*kPos == 1) bz = vector::TransposeOp::create(rewriter, loc, bz, {1, 0});
    start = contractWith(bz, start);
  }
  Value p = contractWith(newB, start);

  // The scale as an accumulator-shaped value: N along n, broadcast along m.
  Value row = rowOf(s);
  SmallVector<int64_t> rowShape(accTy.getRank(), 1);
  rowShape[*nInAcc] = n;
  auto rowTy = VectorType::get(rowShape, getElementTypeOrSelf(row));
  if (row.getType() != rowTy)
    row = vector::ShapeCastOp::create(rewriter, loc, rowTy, row);
  if (!getElementTypeOrSelf(row).isF32())
    row = arith::ExtFOp::create(rewriter, loc, VectorType::get(rowShape, f32),
                                row, /*fastmath=*/{});
  Value scale = vector::BroadcastOp::create(rewriter, loc, acc32Ty, row);
  Value out = math::FmaOp::create(rewriter, loc, scale, p, acc32);

  Value oldB = contract.getRhs();
  if (yieldCast) {
    rewriter.replaceOp(yieldCast, out);
    Operation *accCast = contract.getAcc().getDefiningOp();
    rewriter.eraseOp(contract);
    rewriter.eraseOp(accCast);
  } else {
    rewriter.replaceOp(contract, out);
  }
  eraseDeadChain(rewriter, oldB);
  return success();
}

struct FactorDequant final
    : ::mlir::gc::impl::FactorDequantBase<FactorDequant> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<vector::ContractionOp> contracts;
    getOperation()->walk(
        [&](vector::ContractionOp op) { contracts.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (vector::ContractionOp op : contracts)
      (void)factor(rewriter, op, foldZeroPoint);
  }
};

} // namespace
