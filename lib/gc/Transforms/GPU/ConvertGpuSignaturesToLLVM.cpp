//===- ConvertGpuSignaturesToLLVM.cpp - Legalize signatures -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// TODO: replace once upstream support signature conversion
#include "GPUOpsLowering.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONVERTGPUSIGNATURESTOLLVM
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

struct ConvertGpuSignaturesToLLVM
    : public gc::impl::ConvertGpuSignaturesToLLVMBase<
          ConvertGpuSignaturesToLLVM> {
  using ConvertGpuSignaturesToLLVMBase::ConvertGpuSignaturesToLLVMBase;

  void runOnOperation() override;
};

void ConvertGpuSignaturesToLLVM::runOnOperation() {
  gpu::GPUModuleOp gpuModule = getOperation();

  for (auto func : gpuModule.getOps<func::FuncOp>()) {
    func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(&getContext()));
  }

  LLVMTypeConverter converter(gpuModule.getContext());
  RewritePatternSet patterns(gpuModule.getContext());
  LLVMConversionTarget target(getContext());

  patterns.add<GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter, 0 /*local*/, 3 /*shared*/,
      StringAttr::get(&converter.getContext(), "xe.kernel"));

  if (failed(applyPartialConversion(gpuModule, target, std::move(patterns))))
    signalPassFailure();
}
