//===- JitWrapper.cpp - Wrapper of JIT ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/JitWrapper/Module.hpp"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>

using namespace mlir;

static const char code1[] = R"mlir(
module {
llvm.mlir.global constant @__num_orig_num_args(3 : i32) : i32
func.func @compute(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> attributes { llvm.emit_c_interface } {
    %out = tensor.empty() : tensor<128xf32>
    %2 = linalg.add ins(%a, %b : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
    return %2 : tensor<128xf32>
}
}
)mlir";

extern "C" {
extern int gc_runtime_keep_alive;
}

TEST(ExecutionEngine, JitWrapper) {
  gc_runtime_keep_alive = 0;
  MLIRContext ctx{gc::initAndGetDialects()};
  std::unique_ptr<llvm::MemoryBuffer> ir_buffer =
      llvm::MemoryBuffer::getMemBuffer(code1);
  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ir_buffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  ASSERT_TRUE(module);
  auto jited = gc::JitModule::create(module.get());
  bool jit_success = static_cast<bool>(jited);
  if (!jit_success) {
    auto err = jited.takeError();
    llvm::errs() << err;
    llvm::consumeError(std::move(err));
  }
  ASSERT_TRUE(jit_success);
  OwningMemRef<float, 1> bufA{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t>) { ptr = 1.0f; }};
  OwningMemRef<float, 1> bufB{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t> idx) { ptr = idx[0]; }};
  OwningMemRef<float, 1> bufC{{128}, {128}};
  void *args[] = {&*bufA, &*bufB, &*bufC};
  jited.get()->call(args);
  for (int i = 0; i < 128; i++) {
    ASSERT_EQ(bufC[{i}], 1.0f + i);
  }
}

// compute d = (a+a) + (b+b) + c, where a,b is marked constant
// bufIdx: a=0, b=1, c=2, d=3, foldedA=4, foldedB=5
static const char code2[] = R"mlir(
module {
llvm.mlir.global constant @__num_orig_num_args(4 : i32) : i32
llvm.mlir.global constant @__fold_buffer_ids(dense<[2, 114514, 1919810]> : tensor<3 x i64>) : !llvm.array<3 x i64>
// a,b, foldedA,foldedB
llvm.mlir.global constant @__fold_args(dense<[4, 0, 1, 4, 5]> : tensor<5xi32>) : !llvm.array<5 x i32>
// foldedA, foldedB, c, d
llvm.mlir.global constant @__compute_args(dense<[4, 4, 5, 2, 3]> : tensor<5xi32>) : !llvm.array<5 x i32>

func.func @fold(%a: tensor<128xf32>, %b: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    cpuruntime.printf "HI%zu\n" %c0 : index
    %out = tensor.empty() : tensor<128xf32>
    %2 = linalg.add ins(%a, %a : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
    %out2 = tensor.empty() : tensor<128xf32>
    %3 = linalg.add ins(%b, %b : tensor<128xf32>,tensor<128xf32>) outs(%out2 : tensor<128xf32>) -> tensor<128xf32>
    return %2, %3 : tensor<128xf32>, tensor<128xf32>
}

func.func @compute(%ax2: tensor<128xf32>, %bx2: tensor<128xf32>, %c: tensor<128xf32>) -> tensor<128xf32> attributes { llvm.emit_c_interface } {
    %out = tensor.empty() : tensor<128xf32>
    %2 = linalg.add ins(%ax2, %bx2 : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
    %d = linalg.add ins(%2, %c : tensor<128xf32>,tensor<128xf32>) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    return %d : tensor<128xf32>
}
}
)mlir";

TEST(ExecutionEngine, JitWrapperCached) {
  MLIRContext ctx{gc::initAndGetDialects()};
  std::unique_ptr<llvm::MemoryBuffer> ir_buffer =
      llvm::MemoryBuffer::getMemBuffer(code2);
  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ir_buffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);

  // foldedA and foldedB uses this buffer
  auto ret = std::shared_ptr<float[]>(new float[128 * 2]);
  auto proxy = std::make_shared<gc::const_cache_proxy>(
      ret, ret.get(), 128 * 2 * sizeof(float), true);

  ASSERT_TRUE(gc::reg_cached_tensor(114514, proxy, 0));
  ASSERT_TRUE(gc::reg_cached_tensor(1919810, proxy, 128 * sizeof(float)));

  ASSERT_TRUE(module);
  auto jited = gc::JitModule::create(module.get());
  bool jit_success = static_cast<bool>(jited);
  if (!jit_success) {
    auto err = jited.takeError();
    llvm::errs() << err;
    llvm::consumeError(std::move(err));
  }
  ASSERT_TRUE(jit_success);
  OwningMemRef<float, 1> bufA{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t>) { ptr = 1.0f; }};
  OwningMemRef<float, 1> bufB{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t> idx) { ptr = idx[0]; }};
  OwningMemRef<float, 1> bufC{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t> idx) {
        ptr = -idx[0] * 3;
      }};
  OwningMemRef<float, 1> bufD{{128}, {128}};
  void *args[] = {&*bufA, &*bufB, &*bufC, &*bufD};

  // first call, should run fold()
  {
    testing::internal::CaptureStdout();
    // first call, should run fold()
    jited.get()->call(args);
    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(bufD[{i}], 2 * 1.0f + 2 * i - 3 * i);
    }
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output, "HI0\n");
  }

  {
    testing::internal::CaptureStdout();
    // second call, should not run fold()
    jited.get()->call(args);
    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(bufD[{i}], 2 * 1.0f + 2 * i - 3 * i);
    }
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_TRUE(output.empty());
  }

  // the cache is evicted
  proxy->deref();
  {
    testing::internal::CaptureStdout();
    // third call, should run fold()
    jited.get()->call(args);
    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(bufD[{i}], 2 * 1.0f + 2 * i - 3 * i);
    }
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output, "HI0\n");
  }
}
