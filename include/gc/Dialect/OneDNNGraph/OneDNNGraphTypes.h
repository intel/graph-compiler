//===- OneDNNGraphTypes.h - OneDNN input dialect types ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONEDNNGRAPH_ONEDNNGRAPHTYPES_H
#define ONEDNNGRAPH_ONEDNNGRAPHTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#include "gc/Dialect/OneDNNGraph/OneDNNGraphOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOpsAttributes.h.inc"

#endif // ONEDNNGRAPH_ONEDNNGRAPHTYPES_H
