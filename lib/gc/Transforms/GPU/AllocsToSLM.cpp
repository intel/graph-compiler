//===- AllocsToSLM.cpp - A pass adding shared mem-space attr ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::gc;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_ALLOCSTOSLM
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {

bool isInGpuLaunch(Operation *op) {
  auto launchOp = op->getParentOfType<gpu::LaunchOp>();
  return launchOp != nullptr;
}

bool hasAssignedMemSpace(Value value) {
  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    if (memrefType.getMemorySpace()) {
      return true;
    }
  }
  return false;
}

// Converts `memref::AllocOp` within GPU regions to the GPU shared local
// memory. Adjusts the allocation shape based on GPU block dimensions and
// creates a `memref::SubViewOp` for thread-specific memory access.
struct ConvertAlloc : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  ConvertAlloc(MLIRContext *ctx) : OpRewritePattern<memref::AllocOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value memref = allocOp->getResult(0);

    if (hasAssignedMemSpace(memref)) {
      return rewriter.notifyMatchFailure(
          allocOp, "Memref already has some memory space attribute");
    }

    if (!isInGpuLaunch(allocOp)) {
      return rewriter.notifyMatchFailure(allocOp,
                                         "Only support allocs in GPU regions");
    }

    auto launchOp = allocOp->getParentOfType<gpu::LaunchOp>();

    auto xSz = dyn_cast<arith::ConstantIndexOp>(
        launchOp.getBlockSizeX().getDefiningOp());
    auto ySz = dyn_cast<arith::ConstantIndexOp>(
        launchOp.getBlockSizeY().getDefiningOp());
    auto zSz = dyn_cast<arith::ConstantIndexOp>(
        launchOp.getBlockSizeZ().getDefiningOp());

    if (!xSz || !ySz || !zSz)
      return rewriter.notifyMatchFailure(
          allocOp, "Only support constant block sizes for now");

    SmallVector<int64_t, 3> blockSizes = {xSz.value(), ySz.value(),
                                          zSz.value()};
    MemRefType originalMemRefType = cast<MemRefType>(memref.getType());
    auto originalShape = originalMemRefType.getShape();

    // Scale the allocation size (X dimension) by the number of threads in the
    // work-group
    int64_t newX =
        originalShape[0] * blockSizes[0] * blockSizes[1] * blockSizes[2];
    SmallVector<int64_t> newShape({newX});
    newShape.append(originalShape.begin() + 1, originalShape.end());

    IntegerAttr sharedAddressSpace =
        IntegerAttr::get(rewriter.getIntegerType(64),
                         static_cast<int64_t>(gpu::AddressSpace::Private));

    MemRefType newRootMemRefType =
        MemRefType::get(newShape, originalMemRefType.getElementType(),
                        originalMemRefType.getLayout(), sharedAddressSpace);

    Value newRootMemRef =
        rewriter
            .create<memref::AllocOp>(allocOp.getLoc(), newRootMemRefType,
                                     allocOp.getOperands())
            .getResult();

    // Compute the offsets in SLM chunk for the current thread:
    // X_off = (Xthr_i * Ybl_sz * Zbl_sz + Ythr_i * Zbl_sz + Zthr_i) * Xchunk_sz
    // Offsets for other dimensions = 0
    auto xI = getAffineDimExpr(0, rewriter.getContext());
    auto yI = getAffineDimExpr(1, rewriter.getContext());
    auto zI = getAffineDimExpr(2, rewriter.getContext());
    auto idxExpr =
        (xI * blockSizes[1] * blockSizes[2] + yI * blockSizes[2] + zI) *
        originalShape[0];
    auto idxMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, idxExpr);

    auto threadIds = launchOp.getThreadIds();
    auto offX = rewriter.create<affine::AffineApplyOp>(
        allocOp.getLoc(), idxMap,
        /*exprOperands=*/ValueRange({threadIds.x, threadIds.y, threadIds.z}));

    SmallVector<int64_t> staticOffsets({ShapedType::kDynamic});
    staticOffsets.insert(staticOffsets.end(), originalShape.size() - 1, 0);

    auto offsets = getMixedValues(staticOffsets, {offX}, rewriter);
    auto sizes = getMixedValues(originalShape, {}, rewriter);
    auto strides = getMixedValues(SmallVector<int64_t>(originalShape.size(), 1),
                                  {}, rewriter);

    auto newSlice =
        rewriter
            .create<memref::SubViewOp>(allocOp.getLoc(), newRootMemRef, offsets,
                                       sizes, strides)
            .getResult();
    memref.replaceAllUsesWith(newSlice);

    // Erase deallocs since we don't need them for SLM
    for (auto user : newSlice.getUsers())
      if (auto deallocOp = dyn_cast<memref::DeallocOp>(user))
        deallocOp->erase();

    return success();
  }
};

struct AllocsToSLM : public gc::impl::AllocsToSLMBase<AllocsToSLM> {
  void runOnOperation() override {
    const auto ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAlloc>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
