//===- GPUAttachAddrSpaces.cpp - Attach addr spaces to memrefs --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_GPUATTACHADDRSPACES
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

struct ConvertGpuSignatureMemrefs : public OpRewritePattern<gpu::GPUFuncOp> {
  using OpRewritePattern<gpu::GPUFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gpu::GPUFuncOp func,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
ConvertGpuSignatureMemrefs::matchAndRewrite(gpu::GPUFuncOp func,
                                            PatternRewriter &rewriter) const {

  SmallVector<Type, 4> argTypes;
  for (auto [index, argType] : enumerate(func.getArgumentTypes())) {
    if (auto memrefType = llvm::dyn_cast<MemRefType>(argType)) {
      // llvm::errs() << "Found a memref argument of type " << memrefType <<
      // "\n";
      auto attr = memrefType.getMemorySpace();
      if (attr) {
        // llvm::errs() << "Already has memory space attr attached (" << attr
        //              << "). Skipping function...\n";
        return failure();
      }
      // llvm::errs() << "Found a memref with no addr space attribute at
      // position "
      //              << index << "\n";
      auto newMemrefType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                          memrefType.getLayout(),
                          gpu::GPUMemorySpaceMappingAttr::get(
                              getContext(), gpu::AddressSpace::Global));
      argTypes.push_back(newMemrefType);
    } else {
      argTypes.push_back(argType);
    }
  }

  // Update the block args.
  Block &entryBlock = func.front();
  for (auto [bbArg, type] : llvm::zip(entryBlock.getArguments(), argTypes)) {
    if (bbArg.getType() == type)
      continue;

    // Collect all uses of the bbArg.
    SmallVector<OpOperand *> bbArgUses;
    for (OpOperand &use : bbArg.getUses())
      bbArgUses.push_back(&use);

    // Change the bbArg type to memref with the correct addr space.
    bbArg.setType(type);

    // Not sure this is needed, just in case for now.
    rewriter.setInsertionPointToStart(&entryBlock);
    if (!bbArgUses.empty()) {
      for (OpOperand *use : bbArgUses)
        use->set(bbArg);
    }
  }

  FunctionType funcType = FunctionType::get(getContext(), argTypes, {});
  func.setType(funcType);

  return success();
}

struct GpuAttachAddrSpaces
    : public gc::impl::GpuAttachAddrSpacesBase<GpuAttachAddrSpaces> {
  using GpuAttachAddrSpacesBase::GpuAttachAddrSpacesBase;

  void runOnOperation() override;
};

void GpuAttachAddrSpaces::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertGpuSignatureMemrefs>(&getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
