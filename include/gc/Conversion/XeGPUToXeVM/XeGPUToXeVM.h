//===-- XeGPUToXeVM.h - Convert XeVM to LLVM dialect -------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_XEGPUTOXEVM_XEGPUTOXEVMPASS_H_
#define MLIR_CONVERSION_XEGPUTOXEVM_XEGPUTOXEVMPASS_H_

#include <memory>

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTXEGPUTOXEVMPASS
#include "gc/Conversion/Passes.h.inc"

void populateXeGPUToXeVMConversionPatterns(RewritePatternSet &patterns,
                                           LLVMTypeConverter &typeConverter);

} // namespace mlir

#endif // MLIR_CONVERSION_XEGPUTOXEVM_XEGPUTOXEVMPASS_H_
