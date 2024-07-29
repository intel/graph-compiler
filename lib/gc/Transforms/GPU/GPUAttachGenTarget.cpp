//===- GPUAttachGenTarget.cpp - Attach Gen target to gpu module -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "gc/Dialect/LLVMIR/GENDialect.h"
#include "gc/Target/LLVM/GEN/Target.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_GPUGENATTACHTARGET
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

struct GpuGenAttachTarget
    : public gc::impl::GpuGenAttachTargetBase<GpuGenAttachTarget> {
  using GpuGenAttachTargetBase::GpuGenAttachTargetBase;

  void runOnOperation() override;
};

void GpuGenAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  auto target =
      builder.getAttr<gen::GenTargetAttr>(2, "spir64-unknown-unknown");
  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    SmallVector<Attribute> targets;
    if (std::optional<ArrayAttr> attrs = gpuModule.getTargets())
      targets.append(attrs->getValue().begin(), attrs->getValue().end());
    targets.push_back(target);
    // Remove any duplicate targets.
    targets.erase(llvm::unique(targets), targets.end());
    gpuModule.setTargetsAttr(builder.getArrayAttr(targets));
  });
}
