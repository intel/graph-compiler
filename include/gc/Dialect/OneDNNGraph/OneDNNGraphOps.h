//===- OneDNNGraphOps.h - OneDNN input dialect ops --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECTS_ONEDNNGRAPHOPS_H
#define GC_DIALECTS_ONEDNNGRAPHOPS_H

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Traits.h"

#define GET_OP_CLASSES
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.h.inc"

#endif // GC_DIALECTS_ONEDNNGRAPHOPS_H
