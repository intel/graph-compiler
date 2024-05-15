//===- OneDNNGraphDialect.h - OneDNN input dialect --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECTS_ONEDNNGRAPHDIALECT_H
#define GC_DIALECTS_ONEDNNGRAPHDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOpsDialect.h.inc"

#endif // GC_DIALECTS_ONEDNNGRAPHDIALECT_H
