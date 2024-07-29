//===- Target.h - MLIR Xe target registration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the Gen target interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_GEN_TARGET_H
#define MLIR_TARGET_GEN_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace gen {
/// Registers the `TargetAttrInterface` for the `#gen.target` attribute in
/// the given registry.
void registerGenTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#gen.target` attribute in
/// the registry associated with the given context.
void registerGenTargetInterfaceExternalModels(MLIRContext &context);
} // namespace gen
} // namespace mlir

#endif // MLIR_TARGET_GEN_TARGET_H
