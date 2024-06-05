//===-- Utils.h - Utilities to support oneDnnGraph --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for the oneDnnGraph dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONEDNNGRAPH_UTILS_UTILS_H
#define ONEDNNGRAPH_UTILS_UTILS_H

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace onednn_graph {

// ---------------------- class LogicalTensorInfo ------------------------------
// provide interface to query PropertyType for each tensor args in func::FuncOp
// -----------------------------------------------------------------------------
class LogicalTensorInfo {
public:
  LogicalTensorInfo(mlir::func::FuncOp funcOp);
  onednn_graph::PropertyType queryPropertyType(mlir::Value val);

private:
  llvm::SmallDenseMap<mlir::Value, onednn_graph::PropertyType> propertyTypesMap;
};

} // namespace onednn_graph
} // namespace mlir

#endif // ONEDNNGRAPH_UTILS_UTILS_H
