//===-- GENToLLVMIRTranslation.cpp - Translate GEN to LLVM IR ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR GEN dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "gc/Target/LLVMIR/Dialect/GEN/GENToLLVMIRTranslation.h"
#include "gc/Dialect/LLVMIR/GENDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the GEN dialect to LLVM IR.
class GENDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    // no operations, not supposed to be called
    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();
    // todo; note: migth not need it as we'll have storage classes translated
    // already
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    if (attribute.getName() == gen::GENDialect::getKernelFuncAttrName()) {
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    }
    return success();
  }
};
} // namespace

void mlir::registerGENDialectTranslation(DialectRegistry &registry) {
  registry.insert<gen::GENDialect>();
  registry.addExtension(+[](MLIRContext *ctx, gen::GENDialect *dialect) {
    dialect->addInterfaces<GENDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerGENDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGENDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
