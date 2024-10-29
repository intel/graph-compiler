//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_CONVERSION_PASSES
#define GC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// XeVMToLLVM
//===----------------------------------------------------------------------===//

def ConvertXeVMToLLVMPass : Pass<"convert-xevm-to-llvm"> {
  let summary = "Convert XeVM to LLVM dialect";
  let dependentDialects = [
    "xevm::XeVMDialect",
  ];
}

#endif // GC_CONVERSION_PASSES
