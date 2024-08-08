//===-- LinalgxDialect.cpp - linalgx dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalgx;

#include "gc/Dialect/Linalgx/LinalgxOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOpsAttributes.cpp.inc"

void LinalgxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/Linalgx/LinalgxStructuredOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "gc/Dialect/Linalgx/LinalgxOpsAttributes.cpp.inc"
      >();
}
