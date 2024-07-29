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

#include "gc/Target/LLVM/XE/Target.h"

#include "gc/Dialect/LLVMIR/XeDefinitions.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ExtensibleDialect.h"

using namespace mlir;
using namespace mlir::xe;

namespace mlir::xe::detail {
XeTargetAttrStorage::KeyTy XeTargetAttrStorage::getAsKey() const {
  return KeyTy(O, triple);
}

bool XeTargetAttrStorage::operator==(
    const XeTargetAttrStorage::KeyTy &tblgenKey) const {
  return (O == std::get<0>(tblgenKey)) && (triple == std::get<1>(tblgenKey));
}

::llvm::hash_code
XeTargetAttrStorage::hashKey(const XeTargetAttrStorage::KeyTy &tblgenKey) {
  return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
}

XeTargetAttrStorage *
XeTargetAttrStorage::construct(::mlir::AttributeStorageAllocator &allocator,
                               XeTargetAttrStorage::KeyTy &&tblgenKey) {
  auto O = std::move(std::get<0>(tblgenKey));
  auto triple = std::move(std::get<1>(tblgenKey));
  triple = allocator.copyInto(triple);
  return new (allocator.allocate<XeTargetAttrStorage>())
      XeTargetAttrStorage(std::move(O), std::move(triple));
}
} // namespace mlir::xe::detail

namespace {

// Xe implementation of the gpu:TargetAttrInterface.
class XeTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<XeTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

namespace mlir::xe {
XeTargetAttr XeTargetAttr::get(::mlir::MLIRContext *context, int optLevel,
                               StringRef triple) {
  return Base::get(context, optLevel, triple);
}
} // namespace mlir::xe
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xe::XeTargetAttr)

void mlir::xe::registerXeTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    XeTargetAttr::attachInterface<XeTargetAttrImpl>(*ctx);
  });
}

void mlir::xe::registerXeTargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  registerXeTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

std::optional<SmallVector<char, 0>>
XeTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
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
XeTargetAttrImpl::createObject(Attribute attribute,
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
