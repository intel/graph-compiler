//===- Utils.h - MLIR GEN target utils --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares GEN target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_GEN_UTILS_H
#define MLIR_TARGET_LLVM_GEN_UTILS_H

#include "gc/Dialect/LLVMIR/GENDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Target/LLVM/ModuleToObject.h"

namespace mlir {
namespace gen {

StringRef getONEAPIToolkitPath();

/// Base class for all GEN serializations from GPU modules into binary strings.
/// By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public LLVM::ModuleToObject {
public:
  /// Initializes the `toolkitPath` with the path in `targetOptions` or if empty
  /// with the path in `getONEAPIToolkitPath`.
  SerializeGPUModuleBase(Operation &module, GenTargetAttr target,
                         const gpu::TargetOptions &targetOptions = {});

  // Initialize intermediate spirv target llvm backend
  static void init();

  /// Returns the target attribute.
  GenTargetAttr getTarget() const;

  /// Returns the ONEAPI toolkit path.
  StringRef getToolkitPath() const;

protected:
  /// GEN target attribute.
  GenTargetAttr target;

  /// ONEAPI toolkit path.
  std::string toolkitPath;
};
} // namespace gen
} // namespace mlir

#endif // MLIR_TARGET_LLVM_GEN_UTILS_H
