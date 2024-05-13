//===- OneDNNGraphDialect.h - OneDNN input dialect --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc-dialects/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc-dialects/OneDNNGraph/OneDNNGraphOps.h"
#include "gc-dialects/OneDNNGraph/OneDNNGraphTypes.h"

#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::onednn_graph;

#include "gc-dialects/OneDNNGraph/OneDNNGraphOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// OneDNNGraph dialect.
//===----------------------------------------------------------------------===//

void OneDNNGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc-dialects/OneDNNGraph/OneDNNGraphOps.cpp.inc"
      >();
}
