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
ConvertGpuSignatureMemrefs::matchAndRewrite(gpu::GPUFuncOp f,
                                            PatternRewriter &rewriter) const {
  gpu::GPUMemorySpaceMappingAttr global_addr_space_attr =
      gpu::GPUMemorySpaceMappingAttr::get(f->getContext(),
                                          gpu::AddressSpace::Global);

  // Aliases are not handled, assuming ssa-like semantics for memrefs.
  SmallVector<Value, 4> memrefs;
  // take args, check if memref
  // go through ops (getBody -> walk) - get additional producers
  // get uses -> replace (attach attr)

  for (auto [index, argument] :
       llvm::enumerate(f.getFunctionType().getInputs())) {
    if (auto memrefType = llvm::dyn_cast<MemRefType>(argument)) {
      f.setArgAttr(index, global_addr_space_attr.name, global_addr_space_attr);
    }
  }

  return failure();
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
