//===--------- Decomposition.cpp - Decompose aggregated ops ------*- C++
//-*-===//
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
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_DECOMPOSITION
#define GEN_PASS_DEF_DECOMPOSITION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Share only branch extraction and reconstruction. Each concrete pattern owns
// its fusion marker and operand-map rules. Neither rewrite changes the loops.
struct FusedMatmulDecompositionBase
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult decompose(linalg::GenericOp op, PatternRewriter &rw,
                          ArrayRef<AffineMap> lhsMaps,
                          ArrayRef<AffineMap> rhsMaps, AffineMap outputMap,
                          StringRef fusionMarker) const {
    if (!op.hasPureTensorSemantics()) return failure();
    SmallVector<utils::IteratorType> matmulIterators = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};
    if (op.getIteratorTypesArray() != matmulIterators)
      return rw.notifyMatchFailure(op, "expected a 2D matmul iteration domain");
    Block &body = op.getRegion().front();
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    auto maps = op.getIndexingMapsArray();
    unsigned numInputs = op.getNumDpsInputs();
    // Slice only scalar definitions in this body. Block arguments are leaves
    // that are matched to each branch's matrix inputs and accumulator below.
    BackwardSliceOptions options;
    options.inclusive = true;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *scalarOp) {
      return scalarOp->getBlock() == &body;
    };
    struct Branch {
      unsigned lhs, rhs;
      SetVector<Operation *> operations;
    };
    SmallVector<Branch> branches;
    for (auto item : llvm::enumerate(yield.getValues())) {
      unsigned i = item.index();
      Value result = item.value();
      Branch branch{numInputs, numInputs, {}};
      Operation *root = result.getDefiningOp();
      if (maps[numInputs + i] != outputMap || !root ||
          root->getBlock() != &body)
        return rw.notifyMatchFailure(op,
                                     "expected independent matmul branches");
      if (failed(getBackwardSlice(root, &branch.operations, options)))
        return rw.notifyMatchFailure(
            op, "could not compute scalar branch backward slice");
      bool usesInit = false;
      auto matchInput = [&](Value value) {
        auto arg = dyn_cast<BlockArgument>(value);
        if (!arg || arg.getOwner() != &body) return false;
        unsigned index = arg.getArgNumber();
        if (index == numInputs + i) {
          usesInit = true;
          return true;
        }
        if (index >= numInputs) return false;
        unsigned *input = nullptr;
        if (llvm::is_contained(lhsMaps, maps[index])) {
          input = &branch.lhs;
        } else if (llvm::is_contained(rhsMaps, maps[index])) {
          input = &branch.rhs;
        }
        if (!input || (*input != numInputs && *input != index)) return false;
        *input = index;
        return true;
      };
      for (Operation *scalarOp : branch.operations) {
        if (scalarOp->getNumRegions() || !isMemoryEffectFree(scalarOp))
          return rw.notifyMatchFailure(
              op, "expected region-less, memory-effect-free branch arithmetic");
        for (Value operand : scalarOp->getOperands())
          if (!branch.operations.contains(operand.getDefiningOp()) &&
              !matchInput(operand))
            return rw.notifyMatchFailure(
                op, "expected independent matmul branches");
      }
      if (!usesInit || branch.lhs == numInputs || branch.rhs == numInputs)
        return rw.notifyMatchFailure(op,
                                     "expected independent matmul branches");
      branches.push_back(std::move(branch));
    }

    SmallVector<Value> results;
    for (auto [i, branch] : llvm::enumerate(branches)) {
      auto indexingMaps = rw.getNamedAttr(
          "indexing_maps",
          rw.getAffineMapArrayAttr(
              {maps[branch.lhs], maps[branch.rhs], maps[numInputs + i]}));
      auto matmul = linalg::MatmulOp::create(
          rw, op.getLoc(), TypeRange{op.getResult(i).getType()},
          ValueRange{op.getInputs()[branch.lhs], op.getInputs()[branch.rhs]},
          ValueRange{op.getOutputs()[i]},
          ArrayRef<NamedAttribute>{indexingMaps});
      for (NamedAttribute attr : op->getDiscardableAttrDictionary())
        if (attr.getName() != fusionMarker)
          matmul->setDiscardableAttr(attr.getName(), attr.getValue());
      // Preserve the original scalar casts and arithmetic, including mixed
      // precision accumulation, rather than re-inferring their semantics.
      Block &target = matmul.getRegion().front();
      while (!target.empty()) rw.eraseOp(&target.back());
      IRMapping mapping;
      mapping.map(body.getArgument(branch.lhs), target.getArgument(0));
      mapping.map(body.getArgument(branch.rhs), target.getArgument(1));
      mapping.map(body.getArgument(numInputs + i), target.getArgument(2));
      {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToEnd(&target);
        // A single-root backward slice is already in dependency order.
        for (Operation *scalarOp : branch.operations)
          rw.clone(*scalarOp, mapping);
        linalg::YieldOp::create(rw, op.getLoc(),
                                mapping.lookup(yield.getValues()[i]));
      }
      results.push_back(matmul.getResult(0));
    }
    rw.replaceOp(op, results);
    return success();
  }
};

struct DecomposeFusedMatmul final : FusedMatmulDecompositionBase {
  using FusedMatmulDecompositionBase::FusedMatmulDecompositionBase;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rw) const override {
    if (!mlir::gc::isFusedMatmulOp(op)) return failure();
    AffineExpr m, n, k;
    bindDims(op.getContext(), m, n, k);
    SmallVector<AffineMap> lhsMaps = {
        AffineMap::get(3, 0, {m, k}, op.getContext()),
        AffineMap::get(3, 0, {k, m}, op.getContext())};
    SmallVector<AffineMap> rhsMaps = {
        AffineMap::get(3, 0, {k, n}, op.getContext()),
        AffineMap::get(3, 0, {n, k}, op.getContext())};
    return decompose(op, rw, lhsMaps, rhsMaps,
                     AffineMap::get(3, 0, {m, n}, op.getContext()),
                     mlir::gc::GC_ATTR_FUSED_MATMUL);
  }
};

struct DecomposeFusedGroupedMatmul final : FusedMatmulDecompositionBase {
  using FusedMatmulDecompositionBase::FusedMatmulDecompositionBase;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rw) const override {
    if (!mlir::gc::isFusedGroupedMatmulOp(op)) return failure();
    // Grouped tiling has already selected the expert and clamped its rows.
    // Its LHS uses (m, k); its transposed weights use (n, k).
    AffineExpr m, n, k;
    bindDims(op.getContext(), m, n, k);
    return decompose(op, rw, {AffineMap::get(3, 0, {m, k}, op.getContext())},
                     {AffineMap::get(3, 0, {n, k}, op.getContext())},
                     AffineMap::get(3, 0, {m, n}, op.getContext()),
                     mlir::gc::GC_ATTR_FUSED_GROUPED_MATMUL);
  }
};

// Decomposes any operation that implements AggregatedOpInterface by calling
// its decomposeOperation method.
struct DecomposeAggregatedOp : public RewritePattern {
  explicit DecomposeAggregatedOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto decomposableOp = dyn_cast<linalg::AggregatedOpInterface>(op);
    if (!decomposableOp || !isa<linalgx::AttentionOp>(op)) return failure();

    FailureOr<SmallVector<Value>> maybeNewResults =
        decomposableOp.decomposeOperation(rewriter);
    if (failed(maybeNewResults)) return failure();

    rewriter.replaceOp(op, maybeNewResults.value()[0]);
    return success();
  }
};

// pack src into dst inner_dims_pos=[gi] inner_tiles=[B] (no outer perm)
// → expand_shape src [[0,..,gi-1, gi, gi+1], [gi+2,..]] (trailing 1:2 split)
struct PackToExpandShape : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::PackOp op,
                                PatternRewriter &rw) const override {
    if (!op.getOuterDimsPerm().empty()) return failure();
    if (op.getPaddingValue()) return failure();
    auto innerDims = op.getInnerDimsPos();
    auto innerTiles = op.getMixedTiles();
    if (innerDims.size() != 1) return failure();
    auto tileSize = getConstantIntValue(innerTiles[0]);
    if (!tileSize) return failure();
    int64_t gi = innerDims[0];
    auto srcType = op.getSourceType();
    auto dstType = op.getDestType();
    // Only trailing dim: tile is appended at dst end, expand_shape needs
    // contiguous indices.
    if (gi != srcType.getRank() - 1) return failure();
    // Verify dst shape matches expected pack layout.
    if (dstType.getDimSize(gi) != srcType.getDimSize(gi) / *tileSize)
      return failure();
    if (dstType.getDimSize(dstType.getRank() - 1) != *tileSize)
      return failure();

    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0, r = srcType.getRank(); i < r; ++i) {
      if (i == gi) reassoc.push_back({i, r});
      else reassoc.push_back({i});
    }
    rw.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, dstType, op.getSource(),
                                                 reassoc);
    return success();
  }
};

// unpack src into dst inner_dims_pos=[gi] inner_tiles=[B] (no outer perm)
// → collapse_shape src (trailing 2:1 merge)
struct UnPackToCollapseShape : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::UnPackOp op,
                                PatternRewriter &rw) const override {
    if (!op.getOuterDimsPerm().empty()) return failure();
    auto innerDims = op.getInnerDimsPos();
    auto innerTiles = op.getMixedTiles();
    if (innerDims.size() != 1) return failure();
    auto tileSize = getConstantIntValue(innerTiles[0]);
    if (!tileSize) return failure();
    int64_t gi = innerDims[0];
    auto srcType = op.getSourceType();
    auto dstType = op.getDestType();
    // Compute expected collapsed dim size from source.
    int64_t collapsedDim = srcType.getDimSize(gi) * *tileSize;
    if (ShapedType::isDynamic(collapsedDim)) return failure();
    // dst[gi] must match or be dynamic.
    int64_t dstDim = dstType.getDimSize(gi);
    if (!ShapedType::isDynamic(dstDim) && dstDim != collapsedDim)
      return failure();
    // Build static result type (may differ from dstType if dstType has dynamic
    SmallVector<int64_t> resultShape(dstType.getShape());
    resultShape[gi] = collapsedDim;
    auto resultType =
        RankedTensorType::get(resultShape, dstType.getElementType());
    // Build reassociation: merge gi with last src dim, all others trivial.
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < dstType.getRank(); ++i) {
      if (i == gi) reassoc.push_back({i, srcType.getRank() - 1});
      else reassoc.push_back({i});
    }
    rw.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, resultType,
                                                   op.getSource(), reassoc);
    return success();
  }
};

struct Decomposition final : mlir::gc::impl::DecompositionBase<Decomposition> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<DecomposeAggregatedOp>(ctx);
    patterns.add<DecomposeFusedMatmul, DecomposeFusedGroupedMatmul>(ctx);
    patterns.add<PackToExpandShape, UnPackToCollapseShape>(ctx, /*benefit=*/2);
    linalg::populateDecomposePackUnpackPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns),
                                     GreedyRewriteConfig()))) {
      signalPassFailure();
    }
  }
};

} // namespace
