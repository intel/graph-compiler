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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Utils/Transform.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_REMOVEALLOCS
#define GEN_PASS_DEF_REMOVEALLOCS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Check whether memref has no read uses, transitively through view-like
// ops (subview, collapse_shape, expand_shape)
static bool hasNoReads(Value memref, Operation *excludeOp) {
  for (OpOperand &use : memref.getUses()) {
    Operation *user = use.getOwner();
    if (user == excludeOp) continue;

    // View-like ops: recursively check their results.
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      if (!hasNoReads(viewLike->getResult(0), excludeOp)) return false;
      continue;
    }

    // transfer_write with the memref as the base is a write-only use.
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (write.getBase() == memref) continue;
      return false;
    }

    // copy target is write-only.
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyOp.getTarget() == memref) continue;
      return false; // source of copy = read
    }

    // dealloc is not a read.
    if (isa<memref::DeallocOp>(user)) continue;

    // Conservatively treat any other use as a read.
    return false;
  }
  return true;
}

// Redirect all vector.transfer_write ops targeting `from` to write to `to`
// instead, setting in_bounds=false for dynamic dimensions.
static void redirectTransferWrites(PatternRewriter &rewriter, Value from,
                                   Value to) {
  auto toType = cast<MemRefType>(to.getType());
  SmallVector<vector::TransferWriteOp> writes;
  for (OpOperand &use : from.getUses())
    if (auto tw = dyn_cast<vector::TransferWriteOp>(use.getOwner()))
      writes.push_back(tw);

  for (auto tw : writes) {
    SmallVector<bool> newInBounds;
    for (auto [i, dim] : llvm::enumerate(toType.getShape()))
      newInBounds.push_back(!ShapedType::isDynamic(dim) &&
                            tw.getInBoundsValues()[i]);
    rewriter.setInsertionPoint(tw);
    IRMapping mapping;
    mapping.map(tw.getBase(), to);
    auto *newOp = rewriter.clone(*tw, mapping);
    cast<vector::TransferWriteOp>(newOp).setInBoundsAttr(
        rewriter.getBoolArrayAttr(newInBounds));
    rewriter.eraseOp(tw);
  }
}

// Erase deallocs of `alloc`, then erase `copy` and (if dead) `viewOps`.
// Finally erase alloc itself if dead.
static void cleanupAllocCopyChain(PatternRewriter &rewriter,
                                  memref::AllocOp alloc, memref::CopyOp copy,
                                  ArrayRef<Operation *> viewOps) {
  SmallVector<Operation *> deallocsToErase;
  for (OpOperand &use : alloc.getResult().getUses())
    if (isa<memref::DeallocOp>(use.getOwner()))
      deallocsToErase.push_back(use.getOwner());

  rewriter.eraseOp(copy);
  for (Operation *op : viewOps)
    if (op->use_empty()) rewriter.eraseOp(op);
  for (Operation *op : deallocsToErase) rewriter.eraseOp(op);
  if (alloc->use_empty()) rewriter.eraseOp(alloc);
}

// Move dst's defining op before alloc if needed for dominance.
static void ensureDstDominatesAlloc(PatternRewriter &rewriter, Value dst,
                                    memref::AllocOp alloc) {
  if (auto *dstDef = dst.getDefiningOp()) {
    if (dstDef->getBlock() == alloc->getBlock() &&
        alloc->isBeforeInBlock(dstDef))
      rewriter.moveOpBefore(dstDef, alloc);
  }
}

// Trace backward through view-like ops to find the underlying memref.alloc.
static memref::AllocOp traceToAlloc(Value v) {
  auto memrefVal = dyn_cast<MemrefValue>(v);
  if (!memrefVal) return nullptr;
  return memref::skipViewLikeOps(memrefVal).getDefiningOp<memref::AllocOp>();
}

// If the target of the copy operation is a function argument and the only
// use of the argument is this operation, replace the uses of the source
// with the argument.
struct RemoveCopyToArg : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto target = copy.getTarget();
    if (!isa<BlockArgument>(target) || !target.hasOneUse()) {
      return failure();
    }

    auto src = copy.getSource().getDefiningOp();
    if (isa<memref::AllocOp>(src)) {
      rewriter.replaceAllUsesWith(src->getResult(0), target);
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(src);
               collapse &&
               isa<memref::AllocOp>(collapse.getSrc().getDefiningOp())) {
      auto fn = copy->getParentOfType<func::FuncOp>();
      rewriter.setInsertionPointToStart(&fn.front());
      auto expand = memref::ExpandShapeOp::create(
          rewriter, collapse.getLoc(), collapse.getSrc().getType(), target,
          collapse.getReassociationIndices());
      rewriter.replaceAllUsesWith(collapse.getSrc(), expand.getResult());
    } else {
      return failure();
    }

    rewriter.eraseOp(copy);
    return success();
  }
};

// Remove allocs that are only used by deallocs.
struct RemoveAllocDeallocPair : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (!alloc.getResult().hasOneUse()) {
      return failure();
    }

    auto dealloc =
        dyn_cast<memref::DeallocOp>(alloc.getResult().use_begin()->getOwner());
    if (!dealloc) {
      return failure();
    }

    rewriter.eraseOp(dealloc);
    rewriter.eraseOp(alloc);
    return success();
  }
};

// Collect the chain of view-like ops from `src` back to `alloc`.
// Returns the chain in alloc→src order (chain[0] is the op directly using
// alloc, chain.back() produces src).
static SmallVector<Operation *> collectViewChain(Value src,
                                                 memref::AllocOp alloc) {
  SmallVector<Operation *> chain;
  Value cur = src;
  while (cur.getDefiningOp() != alloc.getOperation()) {
    Operation *op = cur.getDefiningOp();
    if (!op || !isa<ViewLikeOpInterface>(op)) return {}; // invalid chain
    chain.push_back(op);
    cur = cast<ViewLikeOpInterface>(op).getViewSource();
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

// Try to build the inverse of reshape ops (after subviewIdx) applied to dst.
// Returns the inverted value, or nullptr on failure.
static Value invertChainOnDst(PatternRewriter &rewriter, Location loc,
                              ArrayRef<Operation *> chain, int subviewIdx,
                              Value dst) {
  Value cur = dst;
  for (int i = (int)chain.size() - 1; i > subviewIdx; --i) {
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(chain[i])) {
      cur = memref::CollapseShapeOp::create(rewriter, loc, cur,
                                            expand.getReassociationIndices());
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(chain[i])) {
      cur = memref::ExpandShapeOp::create(rewriter, loc, collapse.getSrcType(),
                                          cur,
                                          collapse.getReassociationIndices());
    } else {
      return nullptr;
    }
  }
  return cur;
}

// When a memref.alloc is only written (never read) and then copied to a
// destination through an arbitrary chain of view-like ops, eliminate the
// alloc by redirecting writes to dst (possibly reshaped).
//
// Supports chains of expand_shape / collapse_shape, optionally with a single
// subview (zero offsets, unit strides) indicating the alloc is padded.
struct FoldAllocCopyIntoDirectWrite final : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    Value src = copy.getSource();
    Value dst = copy.getTarget();

    auto alloc = traceToAlloc(src);
    if (!alloc) return failure();

    // Collect view-like chain: alloc → chain[0] → ... → chain[N-1] = src.
    auto chain = collectViewChain(src, alloc);
    if (src.getDefiningOp() != alloc.getOperation() && chain.empty())
      return failure();

    // All intermediate values in the chain must be single-use.
    for (Operation *op : chain)
      if (!op->getResult(0).hasOneUse()) return failure();

    if (!hasNoReads(alloc.getResult(), copy)) return failure();
    if (!hasNoReads(dst, copy)) return failure();

    // Find if there's a subview in the chain (at most one supported).
    int subviewIdx = -1;
    for (auto [i, op] : llvm::enumerate(chain)) {
      if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
        if (!sv.hasUnitStride()) return failure();
        for (auto off : sv.getMixedOffsets()) {
          auto cst = getConstantIntValue(off);
          if (!cst || *cst != 0) return failure();
        }
        if (subviewIdx != -1)
          return failure(); // multiple subviews not supported
        subviewIdx = i;
      }
    }

    ensureDstDominatesAlloc(rewriter, dst, alloc);
    rewriter.setInsertionPoint(alloc);

    if (subviewIdx == -1 && chain.empty()) {
      // Direct alloc → copy: types must match.
      if (src.getType() != dst.getType()) return failure();
      rewriter.replaceAllUsesWith(alloc.getResult(), dst);
    } else if (subviewIdx == -1) {
      // Pure reshape chain — invert all ops on dst, replaceAllUses.
      Value replacement = invertChainOnDst(rewriter, copy.getLoc(), chain,
                                           /*subviewIdx=*/-1, dst);
      if (!replacement) return failure();
      rewriter.replaceAllUsesWith(alloc.getResult(), replacement);
    } else {
      // Chain contains a subview — redirect writes with masking.
      Value writeTarget =
          invertChainOnDst(rewriter, copy.getLoc(), chain, subviewIdx, dst);
      if (!writeTarget) return failure();
      redirectTransferWrites(rewriter, alloc.getResult(), writeTarget);
    }

    cleanupAllocCopyChain(rewriter, alloc, copy,
                          SmallVector<Operation *>(chain.begin(), chain.end()));
    return success();
  }
};

struct RemoveAllocs final : gc::impl::RemoveAllocsBase<RemoveAllocs> {

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveCopyToArg, RemoveAllocDeallocPair,
                 FoldAllocCopyIntoDirectWrite>(&getContext());

    if (failed(applyPatternsGreedily(fn, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
