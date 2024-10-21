//===-- XeVMDialect.cpp - XeVM dialect registration -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Dialect/LLVMIR/XeVMDialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xevm;

#include "gc/Dialect/LLVMIR/XeVMOpsDialect.cpp.inc"

void XeVMDialect::initialize() {
  // NOLINTBEGIN
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/LLVMIR/XeVMOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "gc/Dialect/LLVMIR/XeVMOpsAttributes.cpp.inc"
      >();
  // NOLINTEND
}

#define GET_OP_CLASSES
#include "gc/Dialect/LLVMIR/XeVMOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/LLVMIR/XeVMOpsAttributes.cpp.inc"
