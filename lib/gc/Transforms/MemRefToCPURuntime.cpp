//===-- MemRefToCPURuntime.cpp - MemRef To CPURuntime Lowering --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <vector>

#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONVERTMEMREFTOCPURUNTIME
#include "gc/Transforms/Passes.h.inc"

namespace {

constexpr uint64_t STACK_ALLOC_THRESHOLD = 128;

bool hasParallelParent(Operation *op) {
  // Check if the parent contains a forall / parallel loop
  for (Operation *parentOp = op->getParentOp(); parentOp != nullptr;
       parentOp = parentOp->getParentOp()) {
    if (parentOp->hasTrait<OpTrait::HasParallelRegion>()) {
      return true;
    }
  }
  return false;
}

uint64_t getMemRefSizeInBytes(MemRefType memrefType) {
  if (ShapedType::isDynamicShape(memrefType.getShape()))
    return UINT64_MAX;
  ShapedType shapeType = cast<ShapedType>(memrefType);
  int elementSize = shapeType.getElementTypeBitWidth() / 8;
  AffineMap layout = memrefType.getLayout().getAffineMap();
  ArrayRef<int64_t> shape = memrefType.getShape();
  if (!layout.isIdentity()) {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(memrefType.getStridesAndOffset(strides, offset))) {
      return UINT64_MAX;
    }

    int totalSize = elementSize;
    for (size_t i = 0; i < shape.size(); ++i) {
      totalSize *= (i == shape.size() - 1) ? strides[i] : shape[i];
    }
    return totalSize;
  } else {
    int totalSize = elementSize;
    for (int64_t dim : shape) {
      totalSize *= dim;
    }
    return totalSize;
  }
}

struct AlignedAllocLowering : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    MemRefType type = op.getMemref().getType();
    ValueRange symbolOperands = op.getSymbolOperands();
    ValueRange dynamicSizes = op.getDynamicSizes();
    cpuruntime::AllocOp newAllocOp = rewriter.create<cpuruntime::AllocOp>(
        loc, type, dynamicSizes, symbolOperands);
    if (hasParallelParent(op))
      newAllocOp.setThreadLocal(true);
    rewriter.replaceOp(op, newAllocOp.getResult());
    return success();
  }
};
struct ConvertMemRefToCPURuntime
    : public impl::ConvertMemRefToCPURuntimeBase<ConvertMemRefToCPURuntime> {

  void runOnOperation() final {
    auto *ctx = &getContext();
    llvm::SmallSet<Operation *, 16> noTransformOps;

    // Create deallocOp corresponding to the alloca's location
    getOperation()->walk([&](func::FuncOp funcOp) {
      // Vector to store alloca operations
      SmallVector<memref::AllocaOp, 16> allocaOps;
      // Collect all alloca operations
      funcOp.walk([&](memref::AllocaOp allocaOp) {
        uint64_t allocSize =
            getMemRefSizeInBytes(allocaOp.getResult().getType());
        if (allocSize < STACK_ALLOC_THRESHOLD) {
          noTransformOps.insert(allocaOp);
          return;
        }
        allocaOps.push_back(allocaOp);
      });

      // Create dealloc operations in reverse order of alloca operations
      for (auto allocaOp = allocaOps.rbegin(); allocaOp != allocaOps.rend();
           ++allocaOp) {
        Operation *scopeOp =
            (*allocaOp)
                ->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
        OpBuilder builder(*allocaOp);
        Region &scopeRegion = scopeOp->getRegion(0);
        // Set the insertion point to the end of the region before the
        // terminator
        Block &lastBlock = scopeRegion.back();
        builder.setInsertionPointToEnd(&lastBlock);
        if (!lastBlock.empty() &&
            lastBlock.back().hasTrait<OpTrait::IsTerminator>()) {
          builder.setInsertionPoint(&lastBlock.back());
        }

        // Create the dealloc operation
        auto deallocOp = builder.create<cpuruntime::DeallocOp>(
            (*allocaOp).getLoc(), (*allocaOp).getResult());
        if (hasParallelParent(*allocaOp)) {
          deallocOp.setThreadLocal(true);
        }
      }
    });

    // add lowering target
    ConversionTarget target(getContext());
    // Make all operations legal by default.
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addDynamicallyLegalOp<memref::AllocaOp>([&](Operation *op) {
      // Return true if the operation is in the noTransformOps set, making
      // it dynamically legal.
      return noTransformOps.find(op) != noTransformOps.end();
    });
    // set pattern
    RewritePatternSet patterns(ctx);
    patterns.add<AlignedAllocLowering>(ctx);
    // perform conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir
