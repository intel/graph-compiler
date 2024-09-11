//===- LinalgxOps.h - linalgx dialect ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECTS_LINALGXOPS_H
#define GC_DIALECTS_LINALGXOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOps.h.inc"

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxStructuredOps.h.inc"

#endif // GC_DIALECTS_LINALGXOPS_H
