//===-- TargetDescriptionAnalysisTest.cpp -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Analysis/TargetDescriptionAnalysis.h"
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
module attributes {
dlti.target_system_spec = #dlti.target_system_spec<
"CPU": #dlti.target_device_spec<
    #dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : ui32>,
    #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : ui64>,
    #dlti.dl_entry<"L3_cache_size_in_bytes", "110100480">,
    #dlti.dl_entry<"num_threads", 56 : i32>,
    #dlti.dl_entry<"max_vector_width", 512 : i64>>
>} {}
)mlir";

TEST(TargetDescriptionAnalysis, CPUNormal) {
  MLIRContext ctx{gc::initCompilerAndGetDialects()};
  std::unique_ptr<llvm::MemoryBuffer> ir_buffer =
      llvm::MemoryBuffer::getMemBuffer(code1);
  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ir_buffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  ASSERT_TRUE(module);
  auto CPUTagetDesc = gc::CPUTargetDescriptionAnalysis(module.get());
  ASSERT_EQ(CPUTagetDesc.getNumThreads(), 56);
  ASSERT_EQ(CPUTagetDesc.getCacheSize(1), 49152);
  ASSERT_EQ(CPUTagetDesc.getCacheSize(2), 2097152);
  ASSERT_EQ(CPUTagetDesc.getCacheSize(3), 110100480);
  ASSERT_EQ(CPUTagetDesc.getMaxVectorWidth(), 512);
}

static const char code2[] = R"mlir(
module attributes {
dlti.target_system_spec = #dlti.target_system_spec<
"CPU": #dlti.target_device_spec<
    #dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : ui32>,
    #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : ui32>>
>} {}
)mlir";

TEST(TargetDescriptionAnalysis, CPUMissingValue) {
  MLIRContext ctx{gc::initCompilerAndGetDialects()};
  std::unique_ptr<llvm::MemoryBuffer> ir_buffer =
      llvm::MemoryBuffer::getMemBuffer(code2);
  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ir_buffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  ASSERT_TRUE(module);
  auto CPUTagetDesc = gc::CPUTargetDescriptionAnalysis(module.get());
  ASSERT_EQ(CPUTagetDesc.getNumThreads(), 1);
  ASSERT_EQ(CPUTagetDesc.getCacheSize(1), 49152);
  ASSERT_EQ(CPUTagetDesc.getCacheSize(2), 2097152);
  ASSERT_EQ(CPUTagetDesc.getCacheSize(3), 1048576);
  ASSERT_EQ(CPUTagetDesc.getMaxVectorWidth(), 512);
}