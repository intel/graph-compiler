//===- XeVMAttachTarget.cpp - Attach an XeVM target -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuXeVMAttachTarget` pass, attaching `#xevm.target`
// attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/LLVMIR/XeVMDialect.h"

#include "gc/Target/LLVM/XeVM/Target.h"
#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Regex.h"
#include <iostream>

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_GPUXEVMATTACHTARGET
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

using namespace mlir::xevm;
using namespace mlir;

namespace {
struct XeVMAttachTarget
    : public gc::impl::GpuXeVMAttachTargetBase<XeVMAttachTarget> {
  using Base::Base;

  DictionaryAttr getFlags(OpBuilder &builder) const;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xevm::XeVMDialect>();
  }
};
} // namespace

DictionaryAttr XeVMAttachTarget::getFlags(OpBuilder &builder) const {
  UnitAttr unitAttr = builder.getUnitAttr();
  SmallVector<NamedAttribute, 2> flags;
  auto addFlag = [&](StringRef flag) {
    flags.push_back(builder.getNamedAttr(flag, unitAttr));
  };
  if (!flags.empty())
    return builder.getDictionaryAttr(flags);
  return nullptr;
}

void XeVMAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  auto target = builder.getAttr<XeVMTargetAttr>(optLevel, triple, chip);
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (!moduleMatcher.empty() && !matcher.match(module.getName()))
          continue;
        // Create the target array.
        SmallVector<Attribute> targets;
        if (std::optional<ArrayAttr> attrs = module.getTargets())
          targets.append(attrs->getValue().begin(), attrs->getValue().end());
        targets.push_back(target);
        // Remove any duplicate targets.
        targets.erase(llvm::unique(targets), targets.end());
        // Update the target attribute array.
        module.setTargetsAttr(builder.getArrayAttr(targets));
      }
}
