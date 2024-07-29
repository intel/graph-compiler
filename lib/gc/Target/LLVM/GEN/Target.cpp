//===- Target.cpp - MLIR LLVM XE target compilation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Xe target related functions including registration
// calls for the `#xe.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "gc/Target/LLVM/GEN/Target.h"

#include "gc/Dialect/LLVMIR/GENDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ExtensibleDialect.h"

using namespace mlir;
using namespace mlir::gen;

namespace {

// Xe implementation of the gpu:TargetAttrInterface.
class GenTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<GenTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

void mlir::gen::registerGenTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gen::GENDialect *dialect) {
    GenTargetAttr::attachInterface<GenTargetAttrImpl>(*ctx);
  });
}

void mlir::gen::registerGenTargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  registerGenTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

std::optional<SmallVector<char, 0>>
GenTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                     const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("expected to be a gpu.module op");
    return std::nullopt;
  }

  // todo
}

Attribute
GenTargetAttrImpl::createObject(Attribute attribute,
                                const SmallVector<char, 0> &object,
                                const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps);
}
