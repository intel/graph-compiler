//===- Utils.cpp - Utilities to support the oneDnnGraph dialect -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the oneDnnGraph dialect.
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphTypes.h"
#include "gc/Dialect/OneDNNGraph/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::onednn_graph;

// -----------------------------------------------------------------------------
// LogicalTensorInfo
// -----------------------------------------------------------------------------

LogicalTensorInfo::LogicalTensorInfo(mlir::func::FuncOp funcOp) {
  // Get onednn_graph.property_types attr as array
  auto args = funcOp.getArguments();
  auto attrs = llvm::dyn_cast_or_null<ArrayAttr>(
      funcOp->getAttr(OneDNNGraphDialect::PropertyTypesAttrName));
  if (attrs && attrs.size() != args.size()) {
    funcOp.emitError("property_types array size must equal func args size.");
    return;
  }
  // Get each property_type as enum
  for (size_t i = 0; i < attrs.size(); i++) {
    auto attr = llvm::dyn_cast_or_null<PropertyTypeAttr>(attrs[i]);
    if (!attr) {
      funcOp.emitError(
          "property_types array must only contain PropertyType enum.");
      return;
    }
    propertyTypesMap.insert({args[i], attr.getValue()});
  }
}

PropertyType LogicalTensorInfo::queryPropertyType(mlir::Value val) {
  auto it = propertyTypesMap.find(val);
  if (it == propertyTypesMap.end()) {
    return PropertyType::undef;
  }
  return it->second;
}
