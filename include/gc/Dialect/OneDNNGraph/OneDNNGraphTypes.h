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

#define GET_TYPEDEF_CLASSES
#include "gc-dialects/OneDNNGraph/OneDNNGraphOpsTypes.h.inc"

#endif // ONEDNNGRAPH_ONEDNNGRAPHTYPES_H
