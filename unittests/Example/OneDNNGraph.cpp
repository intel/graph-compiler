//===-- OneDNNGraph.cpp - Tests for OneDNNGraph -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphTypes.h"
#include "gc/Dialect/OneDNNGraph/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;

TEST(onednn_graph, LogicalTensorInfo) {
  std::string moduleStr = R"mlir(
    func.func @foo(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2 : tensor<1xf32>) 
        attributes {onednn_graph.property_types = [#onednn_graph.property_type<undef>, 
                                                   #onednn_graph.property_type<variable>, 
                                                   #onednn_graph.property_type<constant>]} { 
      return 
    }
  )mlir";

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  auto funcOp = cast<func::FuncOp>(module->getBody()->getOperations().front());

  mlir::onednn_graph::LogicalTensorInfo info(funcOp);
  ASSERT_EQ(info.queryPropertyType(Value()),
            mlir::onednn_graph::PropertyType::undef);
  ASSERT_EQ(info.queryPropertyType(funcOp.getArguments()[0]),
            mlir::onednn_graph::PropertyType::undef);
  ASSERT_EQ(info.queryPropertyType(funcOp.getArguments()[1]),
            mlir::onednn_graph::PropertyType::variable);
  ASSERT_EQ(info.queryPropertyType(funcOp.getArguments()[2]),
            mlir::onednn_graph::PropertyType::constant);
}
