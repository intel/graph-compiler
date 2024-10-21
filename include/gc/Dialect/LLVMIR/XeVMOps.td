//===-- XeVMOps.td - XeVM dialect definition ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef XEVMIR_OPS
#define XEVMIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XeVM_Dialect : Dialect {
  let name = "xevm";
  let cppNamespace = "::mlir::xevm";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let useDefaultAttributePrinterParser = 1;
}

class XeVM_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<XeVM_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def XeVM_TargettAttr : XeVM_Attr<"XeVMTarget", "target"> {}

#endif // XEVMIR_OPS
