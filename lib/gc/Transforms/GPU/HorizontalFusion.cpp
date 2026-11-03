//===- HorizontalFusion.cpp - Fuse sibling matmuls --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Utils/Transform.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_HORIZONTALFUSIONPASS
#define GEN_PASS_DEF_HORIZONTALFUSIONPASS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Matrix values, not destinations or the grouped offsets, establish sharing.
static SmallVector<Value> matrixInputs(Operation *op) {
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
    return llvm::to_vector(matmul.getInputs());
  auto grouped = cast<linalgx::GroupedMatmulOp>(op);
  return {grouped.getLhs().front(), grouped.getWeights().front()};
}

static Value destination(Operation *op) {
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
    return matmul.getOutputs().front();
  return cast<linalgx::GroupedMatmulOp>(op).getOutputs().front();
}

static LogicalResult isValidMatmulCandidate(Operation *op,
                                            PatternRewriter &rw) {
  if (op->hasAttr(mlir::gc::GC_ATTR_LEVEL))
    return rw.notifyMatchFailure(op, "matmul has already been tiled");
  if (auto grouped = dyn_cast<linalgx::GroupedMatmulOp>(op)) {
    if (!grouped.hasPureTensorSemantics())
      return rw.notifyMatchFailure(op, "expected tensor semantics");
    if (grouped.getNumBranches() != 1)
      return rw.notifyMatchFailure(op, "grouped matmul is already fused");
    if (llvm::any_of(
            llvm::concat<Value>(grouped.getLhs(), grouped.getWeights(),
                                grouped.getOutputs()),
            [](Value operand) {
              return !isa<FloatType>(
                  cast<ShapedType>(operand.getType()).getElementType());
            }))
      return rw.notifyMatchFailure(op, "grouped matmul fusion only supports "
                                       "floating-point inputs and outputs");
    return success();
  }
  auto matmul = dyn_cast<linalg::MatmulOp>(op);
  if (!matmul)
    return rw.notifyMatchFailure(op, "expected a regular or grouped matmul");
  if (!matmul.hasPureTensorSemantics())
    return rw.notifyMatchFailure(op, "expected tensor semantics");
  if (matmul.getCast() != linalg::TypeFn::cast_signed)
    return rw.notifyMatchFailure(op, "expected signed operand casting");
  if (llvm::any_of(matmul.getOperands(), [](Value value) {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        return !type || type.getRank() != 2;
      }))
    return rw.notifyMatchFailure(op, "expected rank-2 tensor operands");
  AffineExpr m, n, k;
  bindDims(op->getContext(), m, n, k);
  auto map = [&](ArrayRef<AffineExpr> exprs) {
    return AffineMap::get(3, 0, exprs, op->getContext());
  };
  auto maps = matmul.getIndexingMapsArray();
  if ((maps[0] != map({m, k}) && maps[0] != map({k, m})) ||
      (maps[1] != map({k, n}) && maps[1] != map({n, k})) ||
      maps[2] != map({m, n}))
    return rw.notifyMatchFailure(op, "unsupported matmul indexing maps");
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(matmul->getRegions(), captures);
  if (!captures.empty())
    return rw.notifyMatchFailure(op,
                                 "matmul scalar body captures external values");
  for (Operation &scalarOp : matmul.getRegion().front().without_terminator()) {
    if (scalarOp.getNumRegions() != 0)
      return rw.notifyMatchFailure(op, "matmul scalar body has nested regions");
    if (!isPure(&scalarOp))
      return rw.notifyMatchFailure(op, "matmul scalar body is not pure");
  }
  return success();
}

static LogicalResult checkMatrixShape(Value lhs, Value rhs,
                                      Operation *candidate,
                                      PatternRewriter &rw) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  if (lhsType.getRank() != rhsType.getRank())
    return rw.notifyMatchFailure(candidate, "matrix operand ranks differ");
  if (lhsType.getElementType() != rhsType.getElementType())
    return rw.notifyMatchFailure(candidate,
                                 "matrix operand element types differ");
  if (lhs == rhs) return success();
  for (int64_t dim = 0; dim < lhsType.getRank(); ++dim) {
    if (!lhsType.isDynamicDim(dim) && !rhsType.isDynamicDim(dim)) {
      if (lhsType.getDimSize(dim) != rhsType.getDimSize(dim))
        return rw.notifyMatchFailure(candidate,
                                     "matrix operand static dimensions differ");
      continue;
    }
    auto equal = ValueBoundsConstraintSet::areEqual(
        ValueBoundsConstraintSet::Variable(lhs, dim),
        ValueBoundsConstraintSet::Variable(rhs, dim));
    if (failed(equal) || !*equal)
      return rw.notifyMatchFailure(
          candidate, "cannot prove equal dynamic matrix dimensions");
  }
  return success();
}

static LogicalResult isValidSiblingMatmul(Operation *lhs, Operation *rhs,
                                          PatternRewriter &rw) {
  if (lhs->getName() != rhs->getName())
    return rw.notifyMatchFailure(rhs, "matmul operation kinds differ");
  if (lhs->getDiscardableAttrDictionary() !=
      rhs->getDiscardableAttrDictionary())
    return rw.notifyMatchFailure(rhs, "matmul discardable attributes differ");
  auto lhsInputs = matrixInputs(lhs), rhsInputs = matrixInputs(rhs);
  for (auto [a, b] : llvm::zip_equal(lhsInputs, rhsInputs))
    if (failed(checkMatrixShape(a, b, rhs, rw))) return failure();
  // Equal matrix domains imply equal destination extents for valid matmuls.
  // Dynamic initialization tensors need not have the same defining operation.
  auto lhsOut = cast<RankedTensorType>(destination(lhs).getType());
  auto rhsOut = cast<RankedTensorType>(destination(rhs).getType());
  if (lhsOut.getElementType() != rhsOut.getElementType())
    return rw.notifyMatchFailure(rhs, "destination element types differ");
  for (auto [a, b] : llvm::zip_equal(lhsOut.getShape(), rhsOut.getShape()))
    if (!ShapedType::isDynamic(a) && !ShapedType::isDynamic(b) && a != b)
      return rw.notifyMatchFailure(rhs, "destination static dimensions differ");
  if (auto matmul = dyn_cast<linalg::MatmulOp>(lhs)) {
    if (matmul.getIndexingMapsArray() !=
        cast<linalg::MatmulOp>(rhs).getIndexingMapsArray())
      return rw.notifyMatchFailure(rhs, "matmul indexing maps differ");
    return success();
  }
  auto a = cast<linalgx::GroupedMatmulOp>(lhs);
  auto b = cast<linalgx::GroupedMatmulOp>(rhs);
  if (a.getOffsets() != b.getOffsets())
    return rw.notifyMatchFailure(rhs, "grouped matmul offsets differ");
  Value aBound = a.getMaxGroupSize(), bBound = b.getMaxGroupSize();
  if (aBound == bBound) return success();
  if (!aBound || !bBound)
    return rw.notifyMatchFailure(rhs,
                                 "only one grouped matmul has max_group_size");
  auto equal = ValueBoundsConstraintSet::areEqual(aBound, bBound);
  if (failed(equal) || !*equal)
    return rw.notifyMatchFailure(rhs,
                                 "cannot prove equal max_group_size bounds");
  return success();
}

// Check whether the group can be fused at its first matmul without crossing
// effects or non-speculatable operations. Compute backward slices of operand
// producers, including captures, reject dependencies between fusion members,
// and collect pure producers to hoist in block order. Report match failures
// without modifying the IR.
static LogicalResult
isLegalToHoistProducersAndFuse(ArrayRef<Operation *> group, PatternRewriter &rw,
                               SmallVectorImpl<Operation *> &toHoist) {
  Operation *anchor = group.front();
  SmallPtrSet<Operation *, 8> members(group.begin(), group.end());
  // Do not move a computation across memory effects or unknown operations.
  for (Operation *op = anchor->getNextNode(); op != group.back();
       op = op->getNextNode()) {
    if (!isPure(op))
      return rw.notifyMatchFailure(
          anchor, "intervening operation has effects or cannot be speculated");
  }
  BackwardSliceOptions options;
  options.inclusive = true;
  options.omitBlockArguments = true;
  options.omitUsesFromAbove = false;
  StringRef rejectionReason;
  options.filter = [&](Operation *def) {
    if (!rejectionReason.empty()) return false;
    if (members.contains(def)) {
      rejectionReason = "fusion branches depend on each other";
      return false;
    }
    // In valid SSA IR, only same-block producers at or after the anchor need
    // hoisting; block arguments and outside-block definitions are available.
    if (def->getBlock() != anchor->getBlock() || def->isBeforeInBlock(anchor))
      return false;
    if (!isPure(def)) {
      rejectionReason = "operand producer cannot be safely hoisted";
      return false;
    }
    return true;
  };
  SetVector<Operation *> slice;
  for (Operation *op : group) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def) continue;
      if (failed(getBackwardSlice(def, &slice, options)))
        return rw.notifyMatchFailure(
            anchor, "could not compute operand producer backward slice");
      if (!rejectionReason.empty())
        return rw.notifyMatchFailure(anchor, rejectionReason);
    }
  }
  // All producers are in one SSA block, so block order is a dependency order
  // even when several operand slices are combined.
  toHoist.assign(slice.begin(), slice.end());
  llvm::sort(toHoist, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return success();
}

// Build one gc.fused_matmul-marked generic from a legal group of matmuls.
// Deduplicate input value/map pairs, preserve destinations and results in
// branch order, and clone the scalar bodies into a shared (M, N, K) iteration
// domain. The caller replaces the original ops with the corresponding fused
// results.
static linalg::GenericOp fuseMatmuls(ArrayRef<Operation *> group,
                                     PatternRewriter &rw, Location loc) {
  SmallVector<Value> inputs, outputs;
  SmallVector<AffineMap> maps;
  SmallVector<std::pair<unsigned, unsigned>> branchInputs;
  auto addInput = [&](Value input, AffineMap map) -> unsigned {
    for (unsigned i = 0; i < inputs.size(); ++i)
      if (inputs[i] == input && maps[i] == map) return i;
    inputs.push_back(input);
    maps.push_back(map);
    return inputs.size() - 1;
  };
  for (Operation *op : group) {
    auto matmul = cast<linalg::MatmulOp>(op);
    auto branchMaps = matmul.getIndexingMapsArray();
    unsigned a = addInput(matmul.getInputs()[0], branchMaps[0]);
    unsigned b = addInput(matmul.getInputs()[1], branchMaps[1]);
    branchInputs.emplace_back(a, b);
    outputs.push_back(matmul.getOutputs().front());
  }
  maps.append(outputs.size(),
              cast<linalg::MatmulOp>(group.front()).getIndexingMapsArray()[2]);
  SmallVector<utils::IteratorType> iterators = {utils::IteratorType::parallel,
                                                utils::IteratorType::parallel,
                                                utils::IteratorType::reduction};
  auto fused = linalg::GenericOp::create(
      rw, loc, TypeRange(outputs), inputs, outputs, maps, iterators,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
        SmallVector<Value> results;
        for (auto [i, op] : llvm::enumerate(group)) {
          Block &body = cast<linalg::MatmulOp>(op).getRegion().front();
          auto [a, b] = branchInputs[i];
          IRMapping mapping;
          mapping.map(body.getArguments(),
                      ValueRange{args[a], args[b], args[inputs.size() + i]});
          // Clone the existing matmuls' scalar arithmetic, not temporary ops.
          for (Operation &scalarOp : body.without_terminator())
            builder.clone(scalarOp, mapping);
          results.push_back(
              mapping.lookup(body.getTerminator()->getOperand(0)));
        }
        linalg::YieldOp::create(builder, bodyLoc, results);
      });
  for (NamedAttribute attr : group.front()->getDiscardableAttrDictionary())
    fused->setDiscardableAttr(attr.getName(), attr.getValue());
  fused->setDiscardableAttr(mlir::gc::GC_ATTR_FUSED_MATMUL, rw.getUnitAttr());
  return fused;
}

// Build one multi-branch grouped_matmul from a legal group, preserving each
// (lhs, weights) pair and destination/result in branch order. Use the first
// op's common offsets and compatible max_group_size bound. Tiling later creates
// the gc.fused_grouped_matmul-marked generic; the caller replaces the original
// ops.
static linalgx::GroupedMatmulOp fuseGroupedMatmuls(ArrayRef<Operation *> group,
                                                   PatternRewriter &rw,
                                                   Location loc) {
  auto first = cast<linalgx::GroupedMatmulOp>(group.front());
  SmallVector<Value> lhs, weights, outputs;
  for (Operation *op : group) {
    auto grouped = cast<linalgx::GroupedMatmulOp>(op);
    lhs.push_back(grouped.getLhs().front());
    weights.push_back(grouped.getWeights().front());
    outputs.push_back(grouped.getOutputs().front());
  }
  SmallVector<AffineMap> maps(group.size(), first.getLhsMap());
  maps.append(group.size(), first.getWeightsMap());
  maps.push_back(first.getOffsetsMap());
  maps.append(group.size(), first.getOutputMap());
  auto fused = linalgx::GroupedMatmulOp::create(
      rw, loc, TypeRange(outputs), lhs, weights, first.getOffsets(),
      first.getMaxGroupSize(), outputs, rw.getAffineMapArrayAttr(maps));
  for (NamedAttribute attr : first->getDiscardableAttrDictionary())
    fused->setDiscardableAttr(attr.getName(), attr.getValue());
  return fused;
}

// The two pattern instantiations have distinct roots; each only combines
// siblings of its own operation kind in the same block.
template <typename OpTy>
struct HorizontalFusionPattern final : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy root, PatternRewriter &rw) const override {
    Operation *seed = root.getOperation();
    if (failed(isValidMatmulCandidate(seed, rw))) return failure();
    SmallVector<Operation *> group{seed}, toHoist;
    SmallVector<Value> commonInputs = matrixInputs(seed);
    // A fusion partner must directly use one of the seed's matrix inputs.
    // Users are unordered and may repeat when multiple operands are shared.
    SetVector<Operation *> candidateSet;
    for (Value input : commonInputs)
      for (Operation *user : input.getUsers())
        if (isa<OpTy>(user) && user->getBlock() == seed->getBlock() &&
            seed->isBeforeInBlock(user))
          candidateSet.insert(user);
    auto candidates = candidateSet.takeVector();
    llvm::sort(candidates, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    for (Operation *candidate : candidates) {
      if (failed(isValidMatmulCandidate(candidate, rw)) ||
          failed(isValidSiblingMatmul(seed, candidate, rw)))
        continue;
      auto inputs = matrixInputs(candidate);
      auto shared = llvm::filter_to_vector(commonInputs, [&](Value input) {
        return llvm::is_contained(inputs, input);
      });
      if (shared.empty()) {
        (void)rw.notifyMatchFailure(
            candidate, "no matrix input is shared by all fusion branches");
        continue;
      }
      auto proposed = group;
      proposed.push_back(candidate);
      SmallVector<Operation *> candidateDefs;
      if (failed(isLegalToHoistProducersAndFuse(proposed, rw, candidateDefs)))
        continue;
      group = std::move(proposed);
      commonInputs = std::move(shared);
      toHoist = std::move(candidateDefs);
    }
    if (group.size() < 2)
      return rw.notifyMatchFailure(root, "no compatible independent sibling");

    for (Operation *op : toHoist) rw.moveOpBefore(op, seed);
    rw.setInsertionPoint(seed);
    SmallVector<Location> locations;
    for (Operation *op : group) locations.push_back(op->getLoc());
    Location loc = FusedLoc::get(seed->getContext(), locations);
    Operation *fused = isa<linalg::MatmulOp>(seed)
                           ? fuseMatmuls(group, rw, loc).getOperation()
                           : fuseGroupedMatmuls(group, rw, loc).getOperation();
    for (auto [i, op] : llvm::enumerate(group))
      rw.replaceOp(op, fused->getResult(i));

    return success();
  }
};

struct HorizontalFusionPass final
    : mlir::gc::impl::HorizontalFusionPassBase<HorizontalFusionPass> {
  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) return;
    RewritePatternSet patterns(&getContext());
    patterns.add<HorizontalFusionPattern<linalg::MatmulOp>,
                 HorizontalFusionPattern<linalgx::GroupedMatmulOp>>(
        &getContext());
    GreedyRewriteConfig config;
    // Start with the earliest sibling so one rewrite can collect all branches.
    config.setUseTopDownTraversal();
    if (failed(applyPatternsGreedily(func, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace
