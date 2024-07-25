//===-- TestUtils.cpp - Tests for Linalgx Utils -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/Utils.h"

using namespace mlir;

// -----------------------------------------------------------------------------
// Test Helpers
// -----------------------------------------------------------------------------

OpBuilder getMLIRBuilder(MLIRContext *context) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<linalg::LinalgDialect>();

  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  return OpBuilder(context);
}

ModuleOp getTestModule(OpBuilder &builder,
                       std::function<void(OpBuilder &, ValueRange)> createBody,
                       TypeRange types) {
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(builder.getUnknownLoc());
  // Create new function at the end of module
  builder.setInsertionPointToEnd(module.getBody());

  auto function = builder.create<func::FuncOp>(
      loc, "test", builder.getFunctionType(types, {}));

  // Create entry block for function
  Block *body = function.addEntryBlock();
  builder.setInsertionPointToStart(body);

  createBody(builder, body->getArguments());
  builder.create<func::ReturnOp>(loc);

  return module;
}

FailureOr<linalg::GenericOp> createMatmulTestBody(OpBuilder &builder,
                                                  linalgx::VnniOpType opType,
                                                  Value tensorA, Value tensorB,
                                                  Value tensorC) {
  FailureOr<linalg::GenericOp> op = linalgx::makeGenericVnniMatmulOp(
      builder, builder.getUnknownLoc(), opType, {tensorA, tensorB}, {tensorC});
  return op;
}

bool compareModule(MLIRContext *context, ModuleOp *module,
                   const std::string &moduleStr) {
  OwningOpRef<ModuleOp> moduleCmp =
      parseSourceString<ModuleOp>(moduleStr, context);
  return OperationEquivalence::isEquivalentTo(
      *moduleCmp, *module, OperationEquivalence::ignoreValueEquivalence,
      /*markEquivalent=*/nullptr, OperationEquivalence::Flags::IgnoreLocations);
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST(TestUtils, VnniMatmul2D) {
  MLIRContext context;
  OpBuilder builder = getMLIRBuilder(&context);
  // Test params
  auto opType = linalgx::VnniOpType::MM2D;
  Type shapeA = RankedTensorType::get({256, 64}, //
                                      builder.getIntegerType(8));
  Type shapeB = RankedTensorType::get({16, 2, 8, 32, 4}, //
                                      builder.getIntegerType(8));
  Type shapeC = RankedTensorType::get({256, 512}, //
                                      builder.getIntegerType(32));
  // Expected IR
  std::string moduleStr = R"mlir(
  func.func @test(%arg0: tensor<256x64xi8>, %arg1: tensor<16x2x8x32x4xi8>, %arg2: tensor<256x512xi32>) {
    %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3 * 32 + d4 * 4 + d5)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d2, d5)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 32 + d2)>], 
          iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
          } 
          ins(%arg0, %arg1 : tensor<256x64xi8>, tensor<16x2x8x32x4xi8>) 
          outs(%arg2 : tensor<256x512xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %4 = arith.extsi %in : i8 to i32
      %5 = arith.extsi %in_0 : i8 to i32
      %6 = arith.muli %4, %5 : i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<256x512xi32>
    return
  }
  )mlir";
  // Make test module
  FailureOr<linalg::GenericOp> op;
  auto makeVnniMatmul = [&](OpBuilder &b, ValueRange vals) {
    op = createMatmulTestBody(b, opType, vals[0], vals[1], vals[2]);
  };
  ModuleOp module =
      getTestModule(builder, makeVnniMatmul, {shapeA, shapeB, shapeC});
  // Get result
  ASSERT_TRUE(succeeded(op));
  ASSERT_TRUE(isGenericVnniMatmulOp(*op, opType));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::NONE));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::MM4D));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::BRMM3D));
  ASSERT_TRUE(compareModule(&context, &module, moduleStr));
}

TEST(TestUtils, VnniMatmul4D) {
  MLIRContext context;
  OpBuilder builder = getMLIRBuilder(&context);
  // Test params
  auto opType = linalgx::VnniOpType::MM4D;
  Type shapeA = RankedTensorType::get({2, 8, 32, 32}, //
                                      builder.getBF16Type());
  Type shapeB = RankedTensorType::get({4, 8, 16, 32, 2}, //
                                      builder.getBF16Type());
  Type shapeC = RankedTensorType::get({2, 4, 32, 32}, //
                                      builder.getBF16Type());
  // Expected IR
  std::string moduleStr = R"mlir(
  func.func @test(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, %arg2: tensor<2x4x32x32xbf16>) {
    %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d1, d5 * 2 + d6)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d4, d5, d3, d6)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d1, d3)>], 
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
          } 
          ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
          outs(%arg2 : tensor<2x4x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_0 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    } -> tensor<2x4x32x32xbf16>
    return
  }
  )mlir";
  // Make test module
  FailureOr<linalg::GenericOp> op;
  auto makeVnniMatmul = [&](OpBuilder &b, ValueRange vals) {
    op = createMatmulTestBody(b, opType, vals[0], vals[1], vals[2]);
  };
  ModuleOp module =
      getTestModule(builder, makeVnniMatmul, {shapeA, shapeB, shapeC});
  // Get result
  ASSERT_TRUE(succeeded(op));
  ASSERT_TRUE(isGenericVnniMatmulOp(*op, opType));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::NONE));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::MM2D));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::BRMM3D));
  ASSERT_TRUE(compareModule(&context, &module, moduleStr));
}

TEST(TestUtils, VnniBatchReduceMatmul3D) {
  MLIRContext context;
  OpBuilder builder = getMLIRBuilder(&context);
  // Test params
  auto opType = linalgx::VnniOpType::BRMM3D;
  Type shapeA = RankedTensorType::get({512, 32, 64}, //
                                      builder.getBF16Type());
  Type shapeB = RankedTensorType::get({512, 32, 128, 2}, //
                                      builder.getBF16Type());
  Type shapeC = RankedTensorType::get({32, 128}, //
                                      builder.getF32Type());
  // Expected IR
  std::string moduleStr = R"mlir(
  func.func @test(%arg0: tensor<512x32x64xbf16>, %arg1: tensor<512x32x128x2xbf16>, %arg2: tensor<32x128xf32>) {
    %0 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3 * 2 + d4)>, 
                          affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>, 
                          affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>], 
          iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]} 
          ins(%arg0, %arg1 : tensor<512x32x64xbf16>, tensor<512x32x128x2xbf16>) 
          outs(%arg2 : tensor<32x128xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<32x128xf32>
    return
  }
  )mlir";
  // Make test module
  FailureOr<linalg::GenericOp> op;
  auto makeVnniMatmul = [&](OpBuilder &b, ValueRange vals) {
    op = createMatmulTestBody(b, opType, vals[0], vals[1], vals[2]);
  };
  ModuleOp module =
      getTestModule(builder, makeVnniMatmul, {shapeA, shapeB, shapeC});
  // Get result
  ASSERT_TRUE(succeeded(op));
  ASSERT_TRUE(isGenericVnniMatmulOp(*op, opType));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::NONE));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::MM2D));
  ASSERT_FALSE(isGenericVnniMatmulOp(*op, linalgx::VnniOpType::MM4D));
  ASSERT_TRUE(compareModule(&context, &module, moduleStr));
}
