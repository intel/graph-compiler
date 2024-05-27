//===-- OneDNNGraphDialect.cpp - OneDNN input dialect -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphTypes.h"

using namespace mlir;
using namespace mlir::onednn_graph;

#include "gc/Dialect/OneDNNGraph/OneDNNGraphOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// OneDNNGraph dialect.
//===----------------------------------------------------------------------===//

void OneDNNGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.cpp.inc"
      >();
}

LogicalResult
OneDNNGraphDialect::verifyOperationAttribute(Operation *op,
                                             NamedAttribute attr) {
  return success();
}
