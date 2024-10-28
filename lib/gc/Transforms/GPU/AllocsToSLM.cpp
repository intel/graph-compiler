//===- LinalgToXeGPU.cpp - Linalg To XeGPU Lowering -------------*- C++ -*-===//
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

bool isInGpuLaunch(mlir::Operation *op) {
  // Traverse up through parent operations
  mlir::Operation *parentOp = op;
  while (parentOp) {
    // Check if the current parent is a gpu.launch operation
    if (llvm::isa<mlir::gpu::LaunchOp>(parentOp)) {
      return true;
    }
    // Move to the parent operation
    parentOp = parentOp->getParentOp();
  }
  // If we reached the top without finding a gpu.launch, return false
  return false;
}

bool hasAssignedMemSpace(mlir::Value value) {
  if (auto memrefType = value.getType().dyn_cast<mlir::MemRefType>()) {
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

    mlir::Value memref = allocOp->getResult(0);
    mlir::MemRefType originalMemRefType =
        memref.getType().cast<mlir::MemRefType>();

    IntegerAttr sharedAddressSpace =
        IntegerAttr::get(rewriter.getIntegerType(64),
                         static_cast<int64_t>(gpu::AddressSpace::Private));

    // Create a new MemRefType with the desired address space
    mlir::MemRefType newMemRefType = mlir::MemRefType::get(
        originalMemRefType.getShape(), originalMemRefType.getElementType(),
        originalMemRefType.getLayout(), sharedAddressSpace);

    mlir::Value newMemRef = rewriter.create<memref::AllocOp>(
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
