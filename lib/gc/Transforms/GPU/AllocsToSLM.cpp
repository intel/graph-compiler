//===- AllocsToSLM.cpp - A pass adding shared mem-space attr ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

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

    int64_t xI = xSz.value();
    int64_t yI = ySz.value();
    int64_t zI = zSz.value();

    if (zI != 1)
      return rewriter.notifyMatchFailure(
          allocOp, "Only support 2D shared memory for now");

    MemRefType originalMemRefType = cast<MemRefType>(memref.getType());
    auto originalShape = originalMemRefType.getShape();

    // Scale the allocation size by the number of threads in the work-group
    int64_t newX = originalShape[0] * xI;
    int64_t newY = originalShape[1] * yI;

    SmallVector<int64_t> newShape = {newX, newY};

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

    // Compute the offsets in SLM chunk for the current thread
    auto origXConst = rewriter.create<arith::ConstantIndexOp>(allocOp.getLoc(),
                                                              originalShape[0]);
    auto origYConst = rewriter.create<arith::ConstantIndexOp>(allocOp.getLoc(),
                                                              originalShape[1]);

    auto threadIds = launchOp.getThreadIds();

    auto offX =
        rewriter
            .create<arith::MulIOp>(allocOp.getLoc(), threadIds.x, origXConst)
            .getResult();
    auto offY =
        rewriter
            .create<arith::MulIOp>(allocOp.getLoc(), threadIds.y, origYConst)
            .getResult();

    auto offsets = getMixedValues({ShapedType::kDynamic, ShapedType::kDynamic},
                                  {offX, offY}, rewriter);
    auto sizes = getMixedValues(originalShape, {}, rewriter);
    auto strides = getMixedValues({1, 1}, {}, rewriter);

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
