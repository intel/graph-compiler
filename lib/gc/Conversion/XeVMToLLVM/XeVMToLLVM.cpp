//===-- XeVMToLLVM.cpp - XeVM to LLVM dialect conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Conversion/XeVMToLLVM/XeVMToLLVM.h"

#include "gc/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "xevm-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEVMTOLLVMPASS
#include "gc/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace xevm;

namespace {
struct ConvertXeVMToLLVMPass
    : public impl::ConvertXeVMToLLVMPassBase<ConvertXeVMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, xevm::XeVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    RewritePatternSet pattern(&getContext());
    mlir::populateXeVMToLLVMConversionPatterns(pattern);
    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateXeVMToLLVMConversionPatterns(RewritePatternSet &patterns) {
  /*TODO*/
}

void mlir::registerConvertXeVMToLLVMInterface(DialectRegistry &registry) {
  /*TODO*/
}
