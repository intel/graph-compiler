//===-- TestJsonParser.cpp - JsonParser test --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "DnnlTestUtils.h"
#include "JsonParser.h"

#include "gc/Transforms/Passes.h"

static mlir::ModuleOp parse(const char *fileName,
                            llvm::SmallVector<size_t> &inputIds,
                            llvm::SmallVector<size_t> &outputIds,
                            std::unordered_map<std::size_t, Strides> &strides) {
  static auto registry = []() {
    mlir::DialectRegistry registry;
    mlir::gc::registerGraphCompilerPasses();
    registry.insert<mlir::BuiltinDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
    return registry;
  }();
  static mlir::MLIRContext context(registry);
  static bool printModule = []() {
    context.loadAllAvailableDialects();
#ifndef NDEBUG
    return true;
#else
    return false;
#endif
  }();

  auto json = read_str_resource(fileName);
  mlir::ModuleOp module =
      JsonParser::parse(context, json, inputIds, outputIds, strides);

  if (printModule) {
    auto &out = llvm::outs();
    out << "OneDNN JSON to MLIR:\n";
    module.print(out);
    out << "Input IDs: ";
    for (auto i : inputIds) {
      out << i << ' ';
    }
    out << "\nOutput IDs: ";
    for (auto i : outputIds) {
      out << i << ' ';
    }
    out << '\n';
  }

  return module;
}

TEST(TestJsonParser, AddRelu) {
  llvm::SmallVector<size_t> inputIds;
  llvm::SmallVector<size_t> outputIds;
  std::unordered_map<std::size_t, Strides> strides;
  mlir::ModuleOp module = parse("add_relu.json", inputIds, outputIds, strides);

  ASSERT_EQ(inputIds.size(), 2);
  ASSERT_EQ(outputIds.size(), 1);
  ASSERT_EQ(strides.size(), 1);
  ASSERT_EQ(inputIds[0], 0);
  ASSERT_EQ(inputIds[1], 1);
  ASSERT_EQ(outputIds[0], 3);
  ASSERT_EQ(strides[1], Strides({1, 2, 3, 4}));

  auto functions = module.getOps<mlir::func::FuncOp>();
  ASSERT_EQ(std::distance(functions.begin(), functions.end()), 1);
  auto &&func = *functions.begin();
  auto funcType = func.getFunctionType();
  ASSERT_EQ(funcType.getNumInputs(), 2);
  ASSERT_EQ(funcType.getNumResults(), 1);

  auto checkTensorType = [](mlir::Type type) {
    ASSERT_EQ(type.getTypeID(), mlir::RankedTensorType::getTypeID());
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    ASSERT_TRUE(tensorType.getElementType().isBF16());
    ASSERT_EQ(tensorType.getRank(), 4);
    ASSERT_EQ(tensorType.getDimSize(0), 1);
    ASSERT_EQ(tensorType.getDimSize(1), 32);
    ASSERT_EQ(tensorType.getDimSize(2), 28);
    ASSERT_EQ(tensorType.getDimSize(3), 28);
  };

  checkTensorType(funcType.getInput(0));
  checkTensorType(funcType.getInput(1));
  checkTensorType(funcType.getResult(0));

  auto &body = func.getBody();
  auto ops = body.getOps();
  ASSERT_EQ(std::distance(ops.begin(), ops.end()), 3);

  auto addOp = ops.begin();
  auto attrs = addOp->getAttrs();
  ASSERT_EQ(addOp->getName().getStringRef(), "onednn_graph.add");
  ASSERT_EQ(addOp->getNumOperands(), 2);
  ASSERT_EQ(addOp->getNumResults(), 1);
  ASSERT_EQ(attrs.size(), 1);
  ASSERT_EQ(attrs.begin()->getName(),
            mlir::StringAttr::get(addOp->getContext(), "auto_broadcast"));
  ASSERT_FALSE(
      mlir::cast<mlir::BoolAttr>(attrs.begin()->getValue()).getValue());
  checkTensorType(addOp->getOperandTypes()[0]);
  checkTensorType(addOp->getOperandTypes()[1]);
  checkTensorType(addOp->getResultTypes()[0]);

  auto reluOp = std::next(addOp);
  ASSERT_EQ(reluOp->getName().getStringRef(), "onednn_graph.relu");
  ASSERT_EQ(reluOp->getNumOperands(), 1);
  ASSERT_EQ(reluOp->getNumResults(), 1);
  checkTensorType(reluOp->getOperandTypes()[0]);
  checkTensorType(reluOp->getResultTypes()[0]);

  auto returnOp = std::next(reluOp);
  ASSERT_EQ(returnOp->getName().getStringRef(), "func.return");
  ASSERT_EQ(returnOp->getNumOperands(), 1);
  checkTensorType(returnOp->getOperandTypes()[0]);
}

TEST(TestJsonParser, Mpl) {
  llvm::SmallVector<size_t> inputIds;
  llvm::SmallVector<size_t> outputIds;
  std::unordered_map<std::size_t, Strides> strides;
  mlir::ModuleOp module = parse("mpl.json", inputIds, outputIds, strides);

  ASSERT_EQ(inputIds.size(), 5);
  ASSERT_EQ(outputIds.size(), 1);
  ASSERT_EQ(strides.size(), 0);
  ASSERT_EQ(inputIds[0], 0);
  ASSERT_EQ(inputIds[1], 1);
  ASSERT_EQ(inputIds[2], 2);
  ASSERT_EQ(inputIds[3], 5);
  ASSERT_EQ(inputIds[4], 7);
  ASSERT_EQ(outputIds[0], 9);

  auto functions = module.getOps<mlir::func::FuncOp>();
  ASSERT_EQ(std::distance(functions.begin(), functions.end()), 1);
  auto &&func = *functions.begin();
  auto funcType = func.getFunctionType();
  ASSERT_EQ(funcType.getNumInputs(), 5);
  ASSERT_EQ(funcType.getNumResults(), 1);

  auto checkTensorType = [](mlir::Type type, llvm::SmallVector<uint16_t> dims) {
    ASSERT_EQ(type.getTypeID(), mlir::RankedTensorType::getTypeID());
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    ASSERT_TRUE(tensorType.getElementType().isBF16());
    ASSERT_EQ(tensorType.getRank(), dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      ASSERT_EQ(tensorType.getDimSize(i), dims[i]);
    }
  };

  checkTensorType(funcType.getInput(0), {128, 512});
  checkTensorType(funcType.getInput(1), {512, 64});
  checkTensorType(funcType.getInput(2), {64});
  checkTensorType(funcType.getInput(3), {64, 256});
  checkTensorType(funcType.getInput(4), {256});
  checkTensorType(funcType.getResult(0), {128, 256});

  auto &body = func.getBody();
  auto ops = body.getOps();
  ASSERT_EQ(std::distance(ops.begin(), ops.end()), 6);

  auto matmulOp1 = ops.begin();
  ASSERT_EQ(matmulOp1->getName().getStringRef(), "onednn_graph.matmul");
  ASSERT_EQ(matmulOp1->getNumOperands(), 3);
  ASSERT_EQ(matmulOp1->getNumResults(), 1);
  checkTensorType(matmulOp1->getOperandTypes()[0], {128, 512});
  checkTensorType(matmulOp1->getOperandTypes()[1], {512, 64});
  checkTensorType(matmulOp1->getOperandTypes()[2], {64});
  checkTensorType(matmulOp1->getResultTypes()[0], {128, 64});

  auto reluOp1 = std::next(matmulOp1);
  ASSERT_EQ(reluOp1->getName().getStringRef(), "onednn_graph.relu");
  ASSERT_EQ(reluOp1->getNumOperands(), 1);
  ASSERT_EQ(reluOp1->getNumResults(), 1);
  checkTensorType(reluOp1->getOperandTypes()[0], {128, 64});
  checkTensorType(reluOp1->getResultTypes()[0], {128, 64});

  auto matmulOp2 = std::next(reluOp1);
  ASSERT_EQ(matmulOp2->getName().getStringRef(), "onednn_graph.matmul");
  ASSERT_EQ(matmulOp2->getNumOperands(), 2);
  ASSERT_EQ(matmulOp2->getNumResults(), 1);
  checkTensorType(matmulOp2->getOperandTypes()[0], {128, 64});
  checkTensorType(matmulOp2->getOperandTypes()[1], {64, 256});
  checkTensorType(matmulOp2->getResultTypes()[0], {128, 256});

  auto addOp = std::next(matmulOp2);
  ASSERT_EQ(addOp->getName().getStringRef(), "onednn_graph.add");
  ASSERT_EQ(addOp->getNumOperands(), 2);
  ASSERT_EQ(addOp->getNumResults(), 1);
  checkTensorType(addOp->getOperandTypes()[0], {128, 256});
  checkTensorType(addOp->getOperandTypes()[1], {256});
  checkTensorType(addOp->getResultTypes()[0], {128, 256});

  auto reluOp2 = std::next(addOp);
  ASSERT_EQ(reluOp2->getName().getStringRef(), "onednn_graph.relu");
  ASSERT_EQ(reluOp2->getNumOperands(), 1);
  ASSERT_EQ(reluOp2->getNumResults(), 1);
  checkTensorType(reluOp2->getOperandTypes()[0], {128, 256});
  checkTensorType(reluOp2->getResultTypes()[0], {128, 256});

  auto returnOp = std::next(reluOp2);
  ASSERT_EQ(returnOp->getName().getStringRef(), "func.return");
  ASSERT_EQ(returnOp->getNumOperands(), 1);
  checkTensorType(returnOp->getOperandTypes()[0], {128, 256});
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
