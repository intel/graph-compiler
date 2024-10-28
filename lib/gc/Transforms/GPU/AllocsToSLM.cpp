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

struct ConvertAlloc : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  ConvertAlloc(MLIRContext *ctx) : OpRewritePattern<memref::AllocOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (hasAssignedMemSpace(allocOp->getResult(0))) {
      return rewriter.notifyMatchFailure(
          allocOp, "Memref already has some memory space attribute");
    }

    if (!isInGpuLaunch(allocOp)) {
      return rewriter.notifyMatchFailure(allocOp,
                                         "Only support allocs in GPU regions");
    }

    Value memref = allocOp->getResult(0);
    MemRefType originalMemRefType = cast<MemRefType>(memref.getType());

    IntegerAttr sharedAddressSpace =
        IntegerAttr::get(rewriter.getIntegerType(64),
                         static_cast<int64_t>(gpu::AddressSpace::Private));

    // Create a new MemRefType with the desired address space
    MemRefType newMemRefType = MemRefType::get(
        originalMemRefType.getShape(), originalMemRefType.getElementType(),
        originalMemRefType.getLayout(), sharedAddressSpace);

    Value newMemRef = rewriter.create<memref::AllocOp>(
        allocOp.getLoc(), newMemRefType, allocOp.getOperands());

    memref.replaceAllUsesWith(newMemRef);

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
