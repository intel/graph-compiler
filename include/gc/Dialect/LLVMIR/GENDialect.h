//===-- GENDialect.h - MLIR GEN target definitions --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_GENDIALECT_H_
#define MLIR_DIALECT_LLVMIR_GENDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/LLVMIR/GenOpsAttributes.h.inc"

#include "gc/Dialect/LLVMIR/GenOpsDialect.h.inc"

#endif /* MLIR_DIALECT_LLVMIR_XEDEFS_H_ */
