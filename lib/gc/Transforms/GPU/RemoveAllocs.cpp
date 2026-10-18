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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

// True if every use of `memref`, apart from `excludeOp` and deallocs, is a
// vector.transfer_write using it as the destination. Only then is it safe to
// replace the buffer by redirecting the writes.
static bool onlyWrittenByTransferWrites(Value memref, Operation *excludeOp) {
  for (OpOperand &use : memref.getUses()) {
    Operation *user = use.getOwner();
    if (user == excludeOp || isa<memref::DeallocOp>(user)) continue;
    auto write = dyn_cast<vector::TransferWriteOp>(user);
    if (!write || write.getBase() != memref) return false;
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

// Collect the ops that must be moved before `anchor` for `value` to dominate
// it, in dependency order (operands first). Returns false if any op on the way
// cannot be moved safely, in which case nothing must be moved at all.
static bool collectDefsToMove(Value value, Operation *anchor,
                              SmallVectorImpl<Operation *> &toMove,
                              SmallPtrSetImpl<Operation *> &visited) {
  auto *def = value.getDefiningOp();
  // Block arguments and values defined in an enclosing block already dominate,
  // values defined earlier in the same block need no move.
  if (!def || def->getBlock() != anchor->getBlock() ||
      def->isBeforeInBlock(anchor))
    return true;
  if (!visited.insert(def).second) return true;
  // Only pure, region-less ops (index arithmetic, view ops) may be reordered.
  if (def->getNumRegions() != 0 || !isMemoryEffectFree(def)) return false;
  for (Value operand : def->getOperands())
    if (!collectDefsToMove(operand, anchor, toMove, visited)) return false;
  toMove.push_back(def);
  return true;
}

// Move dst's defining op, together with the ops computing its operands, before
// alloc if needed for dominance. Returns false if that is not possible; the IR
// is left untouched in that case.
static bool ensureDstDominatesAlloc(PatternRewriter &rewriter, Value dst,
                                    memref::AllocOp alloc) {
  SmallVector<Operation *> toMove;
  SmallPtrSet<Operation *, 8> visited;
  if (!collectDefsToMove(dst, alloc, toMove, visited)) return false;
  // toMove is in dependency order, so moving each one directly before alloc
  // keeps the operands ahead of their users.
  for (Operation *op : toMove) rewriter.moveOpBefore(op, alloc);
  return true;
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
static bool canInvertChainOnDst(ArrayRef<Operation *> chain, int subviewIdx) {
  return llvm::all_of(chain.drop_front(subviewIdx + 1), [](Operation *op) {
    return isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(op);
  });
}

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

    // Direct alloc → copy. Identical types can simply be replaced; a
    // destination that only differs in layout (e.g. a subview of the function
    // output, as produced by concat elimination) needs the writes redirected,
    // because vector.transfer_write accepts any strided layout. All checks are
    // done before any mutation.
    bool redirectWrites = false;
    if (subviewIdx == -1 && chain.empty() && src.getType() != dst.getType()) {
      auto srcType = cast<MemRefType>(src.getType());
      auto dstType = cast<MemRefType>(dst.getType());
      if (srcType.getShape() != dstType.getShape() ||
          srcType.getElementType() != dstType.getElementType() ||
          srcType.getMemorySpace() != dstType.getMemorySpace() ||
          !onlyWrittenByTransferWrites(alloc.getResult(), copy))
        return failure();
      redirectWrites = true;
    }

    // Validate the inversion before moving destination definitions. A rewrite
    // pattern must not return failure after it has mutated the IR.
    if ((subviewIdx != -1 || !chain.empty()) &&
        !canInvertChainOnDst(chain, subviewIdx))
      return failure();

    if (!ensureDstDominatesAlloc(rewriter, dst, alloc)) return failure();
    rewriter.setInsertionPoint(alloc);

    if (subviewIdx == -1 && chain.empty()) {
      if (redirectWrites)
        redirectTransferWrites(rewriter, alloc.getResult(), dst);
      else rewriter.replaceAllUsesWith(alloc.getResult(), dst);
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

// Fold alloc → transfer_write → reshape → transfer_read into shape_cast:
//   %alloc = memref.alloc()
//   vector.transfer_write %v, %alloc[0,0,...]
//   %rs = memref.expand_shape / collapse_shape %alloc
//   %r  = vector.transfer_read %rs[0,0,...]
// → %r = vector.shape_cast %v  (+ erase alloc, write, reshape)
struct FoldAllocWriteReshapeRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rw) const override {
    auto isZero = [](Value v) {
      auto cst = getConstantIntValue(v);
      return cst && *cst == 0;
    };
    if (!llvm::all_of(read.getIndices(), isZero)) return failure();

    auto *reshape = read.getBase().getDefiningOp();
    if (!isa_and_present<memref::ExpandShapeOp, memref::CollapseShapeOp>(
            reshape))
      return failure();

    auto reshapeInput = cast<ViewLikeOpInterface>(reshape).getViewSource();
    auto alloc = reshapeInput.getDefiningOp<memref::AllocOp>();
    if (!alloc) return failure();

    // Find the single transfer_write into alloc.
    vector::TransferWriteOp write;
    for (auto &use : alloc.getResult().getUses()) {
      if (auto w = dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
        if (write || !llvm::all_of(w.getIndices(), isZero)) return failure();
        write = w;
      } else if (!isa<memref::DeallocOp, memref::ExpandShapeOp,
                      memref::CollapseShapeOp>(use.getOwner())) {
        return failure(); // unexpected use
      }
    }
    if (!write) return failure();

    auto srcType = cast<VectorType>(write.getVector().getType());
    auto dstType = cast<VectorType>(read.getResult().getType());
    if (srcType.getNumElements() != dstType.getNumElements()) return failure();

    rw.replaceOpWithNewOp<vector::ShapeCastOp>(read, dstType,
                                               write.getVector());
    if (reshape->use_empty()) rw.eraseOp(reshape);
    if (write->use_empty()) rw.eraseOp(write);
    if (alloc->use_empty()) rw.eraseOp(alloc);
    return success();
  }
};

struct PromoteRemainingAllocToGpuAlloc final
    : OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    // Ignore allocations inside a kernel
    if (getKernelLoop(alloc) != nullptr) return failure();
    bool usedByKernel = ::mlir::gc::isOpDependsOnResult(
        [](Operation *user) { return getKernelLoop(user) != nullptr; }, alloc);
    if (!usedByKernel) return failure();

    rewriter.setInsertionPoint(alloc);
    auto gpuAlloc = gpu::AllocOp::create(
        rewriter, alloc.getLoc(), alloc.getType(), Type{}, ValueRange{},
        alloc.getDynamicSizes(), alloc.getSymbolOperands(),
        /*hostShared=*/false);

    if (!hasDealloc(alloc.getResult())) {
      setInsertionPointAfterLastUse(rewriter, alloc.getResult());
      gpu::DeallocOp::create(rewriter, alloc.getLoc(), Type{}, ValueRange{},
                             gpuAlloc.getMemref());
    }

    rewriter.replaceOp(alloc, gpuAlloc.getMemref());
    return success();
  }

private:
  static bool hasDealloc(Value value) {
    for (Operation *user : value.getUsers()) {
      if (isa<memref::DeallocOp, gpu::DeallocOp>(user)) return true;
      auto view = dyn_cast<ViewLikeOpInterface>(user);
      if (view && hasDealloc(view->getResult(0))) return true;
    }
    return false;
  }

  static Operation *getAncestorInBlock(Operation *op, Block *block) {
    while (op && op->getBlock() != block) op = op->getParentOp();
    return op;
  }

  static void collectUseAncestors(Value value, Block *block,
                                  SmallVectorImpl<Operation *> &ops) {
    for (Operation *user : value.getUsers()) {
      if (Operation *ancestor = getAncestorInBlock(user, block))
        ops.push_back(ancestor);
      auto view = dyn_cast<ViewLikeOpInterface>(user);
      if (view) collectUseAncestors(view->getResult(0), block, ops);
    }
  }

  static void setInsertionPointAfterLastUse(PatternRewriter &rewriter,
                                            Value value) {
    auto *block = value.getParentBlock();
    Operation *last = value.getDefiningOp();
    SmallVector<Operation *> useAncestors;
    collectUseAncestors(value, block, useAncestors);
    for (Operation *op : useAncestors)
      if (last->isBeforeInBlock(op)) last = op;

    if (last->hasTrait<OpTrait::IsTerminator>())
      rewriter.setInsertionPoint(last);
    else rewriter.setInsertionPointAfter(last);
  }
};

struct PromoteRemainingDeallocToGpuDealloc final
    : OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp dealloc,
                                PatternRewriter &rewriter) const override {
    if (!getSourceGpuAlloc(dealloc.getMemref())) return failure();

    gpu::DeallocOp::create(rewriter, dealloc.getLoc(), Type{}, ValueRange{},
                           dealloc.getMemref());
    rewriter.eraseOp(dealloc);
    return success();
  }

private:
  static gpu::AllocOp getSourceGpuAlloc(Value value) {
    while (auto *op = value.getDefiningOp()) {
      if (auto alloc = dyn_cast<gpu::AllocOp>(op)) return alloc;
      auto view = dyn_cast<ViewLikeOpInterface>(op);
      if (!view) break;
      value = view.getViewSource();
    }
    return nullptr;
  }
};

struct RemoveAllocs final : ::mlir::gc::impl::RemoveAllocsBase<RemoveAllocs> {

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveCopyToArg, RemoveAllocDeallocPair,
                 FoldAllocCopyIntoDirectWrite, FoldAllocWriteReshapeRead>(
        &getContext());

    if (failed(applyPatternsGreedily(fn, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet fallbackPatterns(&getContext());
    fallbackPatterns.add<PromoteRemainingAllocToGpuAlloc,
                         PromoteRemainingDeallocToGpuDealloc>(&getContext());
    if (failed(applyPatternsGreedily(fn, std::move(fallbackPatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
