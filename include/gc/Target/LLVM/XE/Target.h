//===- Target.h - MLIR Xe target registration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the Xe target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_XE_TARGET_H
#define MLIR_TARGET_XE_TARGET_H

#include <tuple>
#include <utility>

#include "mlir/IR/Dialect.h"

namespace mlir {
class DialectRegistry;
class MLIRContext;
// namespace gpu {
// class TargetOptions;
// }
namespace xe {
namespace detail {
struct XeTargetAttrStorage;
struct XeTargetAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int, ::llvm::StringRef>;
  XeTargetAttrStorage(int O, ::llvm::StringRef triple)
      : O(std::move(O)), triple(std::move(triple)) {}
  KeyTy getAsKey() const;
  bool operator==(const KeyTy &tblgenKey) const;
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey);
  static XeTargetAttrStorage *
  construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey);

  int O;
  ::llvm::StringRef triple;
};
} // namespace detail

struct XeTargetAttr
    : public ::mlir::Attribute::AttrBase<XeTargetAttr, ::mlir::Attribute,
                                         detail::XeTargetAttrStorage> {
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xe.target";
  static constexpr ::llvm::StringLiteral dialectName = "llvm";
  static constexpr ::llvm::StringLiteral getMnemonic() { return {"target"}; }
  static XeTargetAttr get(::mlir::MLIRContext *context, int optLevel = 2,
                          StringRef triple = "spir64-unknown-unknown");
  //   std::optional<::mlir::SmallVector<char, 0>>
  //   serializeToObject(::mlir::Operation *module,
  //                     const ::mlir::gpu::TargetOptions &options);
  //   ::mlir::Attribute createObject(const ::llvm::SmallVector<char, 0>
  //   &object,
  //                                  const ::mlir::gpu::TargetOptions
  //                                  &options);
};

/// Registers the `TargetAttrInterface` for the `#xe.target` attribute in
/// the given registry.
void registerXeTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#xe.target` attribute in
/// the registry associated with the given context.
void registerXeTargetInterfaceExternalModels(MLIRContext &context);
} // namespace xe
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xe::XeTargetAttr);

#endif // MLIR_TARGET_XE_TARGET_H
