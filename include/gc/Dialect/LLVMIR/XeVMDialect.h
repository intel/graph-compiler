//===-- XeVMDialect.h - MLIR XeVM target definitions ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "gc/Dialect/LLVMIR/XeVMOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/LLVMIR/XeVMOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "gc/Dialect/LLVMIR/XeVMOps.h.inc"

#include "gc/Dialect/LLVMIR/XeVMOpsDialect.h.inc"

namespace mlir::xevm {
/// XeVM memory space identifiers following SPIRV storage class convention
/// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
///
enum class XeVMMemorySpace : uint32_t {
  kFunction = 0,        // OpenCL workitem address space
  kCrossWorkgroup = 1,  // OpenCL Global memory
  kUniformConstant = 2, // OpenCL Constant memory
  kWorkgroup = 3,       // OpenCL Local memory
  kGeneric = 4          // OpenCL Generic memory
};

} // namespace mlir::xevm
#endif /* MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_ */
