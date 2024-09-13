//===-- JitWrapper.cpp - Wrapper for JIT ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/Driver/Driver.h"
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
func.func @entry(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> attributes { llvm.emit_c_interface } {
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
  MLIRContext ctx{gc::initCompilerAndGetDialects()};
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
  jited.get()->call(args, 3);
  for (int i = 0; i < 128; i++) {
    ASSERT_EQ(bufC[{i}], 1.0f + i);
  }
}

// compute d = (a+a) + (b+b) + c, where a,b is marked constant
// bufIdx: a=0, b=1, c=2, d=3, foldedA=4, foldedB=5
static const char code2[] = R"mlir(
module {
func.func @entry(%a: tensor<128xf32>, %b: tensor<128xf32>, %c: tensor<128xf32>) -> tensor<128xf32> attributes { llvm.emit_c_interface, runtime_const_args_index = [0 : i32, 1 : i32] } {
    %out = tensor.empty() : tensor<128xf32>
    %ax2 = linalg.add ins(%a, %a : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
    %out2 = tensor.empty() : tensor<128xf32>
    %bx2 = linalg.add ins(%b, %b : tensor<128xf32>,tensor<128xf32>) outs(%out2 : tensor<128xf32>) -> tensor<128xf32>
    %out3 = tensor.empty() : tensor<128xf32>
    %ax2pbx2 = linalg.add ins(%ax2, %bx2 : tensor<128xf32>,tensor<128xf32>) outs(%out3 : tensor<128xf32>) -> tensor<128xf32>
    %out4 = tensor.empty() : tensor<128xf32>
    %d = linalg.add ins(%ax2pbx2, %c : tensor<128xf32>,tensor<128xf32>) outs(%out4 : tensor<128xf32>) -> tensor<128xf32>
    return %d : tensor<128xf32>
}
}
)mlir";

TEST(ExecutionEngine, JitWrapperCached) {
  MLIRContext ctx{gc::initCompilerAndGetDialects()};
  std::unique_ptr<llvm::MemoryBuffer> ir_buffer =
      llvm::MemoryBuffer::getMemBuffer(code2);
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

  auto ret = std::shared_ptr<float[]>(new float[128]);
  auto proxy = std::make_shared<gc::ConstCacheProxy>(ret, ret.get(),
                                                     128 * sizeof(float), true);
  // Can not register with already existing key.
  ASSERT_FALSE(gc::regCachedTensor(0, proxy, 0));

  proxy = gc::queryCacheTensor(0)->base;
  auto data = (float *)proxy->getBufferUnsafe();

  OwningMemRef<float, 1> bufA{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t>) { ptr = 1.0f; }};
  OwningMemRef<float, 1> bufB{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t> idx) { ptr = idx[0]; }};
  OwningMemRef<float, 1> bufC{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t> idx) {
        ptr = -idx[0] * 3;
      }};
  OwningMemRef<float, 1> bufD{
      {128}, {128}, [](float &ptr, ArrayRef<int64_t>) { ptr = 100.0f; }};
  void *args[] = {&*bufA, &*bufB, &*bufC, &*bufD};

  {
    // first call, should run fold()
    jited.get()->call(args, 4);

    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(*(data + i), 2 * 1.0f + 2 * i);
    }
    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(bufD[{i}], 2 * 1.0f + 2 * i - 3 * i);
    }
  }

  {
    // second call, should not run fold()
    jited.get()->call(args, 4);
    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(bufD[{i}], 2 * 1.0f + 2 * i - 3 * i);
    }
  }

  // the cache is evicted
  proxy->deref();
  {
    // third call, should run fold()
    jited.get()->call(args, 4);
    for (int i = 0; i < 128; i++) {
      ASSERT_EQ(bufD[{i}], 2 * 1.0f + 2 * i - 3 * i);
    }
  }
}
