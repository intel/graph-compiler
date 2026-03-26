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
    if (user == excludeOp)
      continue;

    // View-like ops: recursively check their results.
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      if (!hasNoReads(viewLike->getResult(0), excludeOp))
        return false;
      continue;
    }

    // transfer_write with the memref as the base is a write-only use.
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (write.getBase() == memref)
        continue;
      return false;
    }

    // copy target is write-only.
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyOp.getTarget() == memref)
        continue;
      return false; // source of copy = read
    }

    // dealloc is not a read.
    if (isa<memref::DeallocOp>(user))
      continue;

    // Conservatively treat any other use as a read.
    return false;
  }
  return true;
}

// Trace backward through view-like ops to find the underlying memref.alloc.
static memref::AllocOp traceToAlloc(Value v) {
  auto memrefVal = dyn_cast<MemrefValue>(v);
  if (!memrefVal)
    return nullptr;
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

// When a memref.alloc is only written (never read) and then copied to a
// destination, replace uses of the alloc with the destination and remove the
// copy. Supports the copy source being the alloc directly, or separated by
// a single view-like op (expand_shape / collapse_shape).
//
// Example (with expand_shape):
//   %alloc = memref.alloc() : memref<128x80xf16>
//   vector.transfer_write %v, %alloc[%c0, %c0]
//   %exp = memref.expand_shape %alloc [[0,1],[2]] ...
//   memref.copy %exp, %subview
// =>
//   %col = memref.collapse_shape %subview [[0,1],[2]]
//   vector.transfer_write %v, %col[%c0, %c0]
//
struct FoldAllocCopyIntoDirectWrite final : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    Value src = copy.getSource();
    Value dst = copy.getTarget();

    // Find the root alloc, possibly through a single view-like op.
    auto alloc = traceToAlloc(src);
    if (!alloc)
      return failure();

    // Determine whether there's a view-like op between src and alloc.
    Operation *viewOp = nullptr;
    if (src.getDefiningOp() != alloc.getOperation()) {
      // Only support a single view-like op between alloc and copy.
      viewOp = src.getDefiningOp();
      if (!isa<ViewLikeOpInterface>(viewOp))
        return failure();
      auto viewLike = cast<ViewLikeOpInterface>(viewOp);
      if (viewLike.getViewSource().getDefiningOp() != alloc.getOperation())
        return failure();
    }

    // No reads from the alloc (only writes).
    if (!hasNoReads(alloc.getResult(), copy))
      return failure();

    // No reads from the destination either.
    if (!hasNoReads(dst, copy))
      return failure();

    Value replacement;

    if (!viewOp) {
      // Direct alloc -> copy: types must match for substitution.
      if (src.getType() != dst.getType())
        return failure();
      replacement = dst;
    } else {
      // Single view-like op between alloc and copy.  Create the inverse
      // op on dst so that the result type matches the alloc type.
      // The inverse op (and dst) must dominate all uses of alloc, so
      // move dst before alloc if it's defined later in the same block.
      if (auto *dstDef = dst.getDefiningOp()) {
        if (dstDef->getBlock() == alloc->getBlock() &&
            alloc->isBeforeInBlock(dstDef)) {
          rewriter.moveOpBefore(dstDef, alloc);
        }
      }
      rewriter.setInsertionPoint(alloc);
      if (auto expand = dyn_cast<memref::ExpandShapeOp>(viewOp)) {
        replacement = memref::CollapseShapeOp::create(
            rewriter, copy.getLoc(), dst, expand.getReassociationIndices());
      } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(viewOp)) {
        replacement = memref::ExpandShapeOp::create(
            rewriter, copy.getLoc(), collapse.getSrcType(), dst,
            collapse.getReassociationIndices());
      } else {
        return failure();
      }
    }

    // Collect deallocs of the alloc to erase after replacement.
    SmallVector<Operation *> deallocsToErase;
    for (OpOperand &use : alloc.getResult().getUses()) {
      if (isa<memref::DeallocOp>(use.getOwner()))
        deallocsToErase.push_back(use.getOwner());
    }

    // Replace alloc with the (possibly reshaped) destination.
    rewriter.replaceAllUsesWith(alloc.getResult(), replacement);

    rewriter.eraseOp(copy);
    for (Operation *op : deallocsToErase)
      rewriter.eraseOp(op);
    if (viewOp && viewOp->use_empty())
      rewriter.eraseOp(viewOp);
    rewriter.eraseOp(alloc);

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
