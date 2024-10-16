//===-- GpuOclRuntime.cpp - -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl_ext.h>
#include <memory>

#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Utils/Error.h"

#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace gc::gpu;

constexpr char addStatic[] = R"mlir(
module @test {
  func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<64x64xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<64x64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.add ins(%1, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
    return
  }
}
)mlir";

constexpr char addDynamic[] = R"mlir(
module @test {
  func.func @entry(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg0 restrict : memref<?x?xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<?x?xf32>
    %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
    %3 = linalg.add ins(%1, %0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}
)mlir";

constexpr char matmulAddStatic[] = R"mlir(
module @fragment_name attributes {"#dlti.sys_spec" = #dlti.target_system_spec<"GPU" : #dlti.target_device_spec<#dlti.dl_entry<"max_work_group_size", 16 : i64>>>} {
  func.func @entry(%arg0: memref<128x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<128x256xf16>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<128x256xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<256x256xf16>
    %2 = tensor.empty() : tensor<128x256xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<128x256xf16>) -> tensor<128x256xf16>
    %4 = linalg.matmul ins(%0, %1 : tensor<128x256xf16>, tensor<256x256xf16>) outs(%3 : tensor<128x256xf16>) -> tensor<128x256xf16>
    %5 = tensor.empty() : tensor<128x256xf16>
    %6 = linalg.add ins(%4, %0 : tensor<128x256xf16>, tensor<128x256xf16>) outs(%5 : tensor<128x256xf16>) -> tensor<128x256xf16>
    bufferization.materialize_in_destination %6 in restrict writable %arg2 : (tensor<128x256xf16>, memref<128x256xf16>) -> ()
    return
  }
}
)mlir";

struct TestBase {
  OclRuntime runtime = gcGetOrReport(OclRuntime::get());
  cl_command_queue queue = gcGetOrReport(runtime.createQueue());
  OclContext ctx{runtime, queue};
  MLIRContext mlirCtx{gc::initCompilerAndGetDialects()};

  virtual void exec(std::shared_ptr<const OclModule> &mod) = 0;

  virtual ~TestBase() { gcGetOrReport(runtime.releaseQueue(queue)); }

  OwningOpRef<ModuleOp> parse(const char *code) {
    std::unique_ptr<llvm::MemoryBuffer> memBuf =
        llvm::MemoryBuffer::getMemBuffer(code);
    llvm::SourceMgr srcMgr;
    srcMgr.AddNewSourceBuffer(std::move(memBuf), SMLoc());
    return parseSourceFile<ModuleOp>(srcMgr, &mlirCtx);
  }
};

template <unsigned N, unsigned M = N> struct TestAdd : TestBase {
  static constexpr unsigned size = N * M;
  float *buf0 = gcGetOrReport(runtime.usmNewDev<float>(size));
  float *buf1 = gcGetOrReport(runtime.usmNewDev<float>(size));
  float *buf2 = gcGetOrReport(runtime.usmNewShared<float>(size));

  explicit TestAdd() {
    float cpuBuf[size];
    std::fill(cpuBuf, cpuBuf + size, 2.0f);
    assert(runtime.usmCpy(ctx, cpuBuf, buf0, size));
    assert(runtime.usmCpy(ctx, cpuBuf, buf1, size));
    gcGetOrReport(ctx.finish());
  }

  ~TestAdd() override {
    assert(runtime.usmFree(buf0));
    assert(runtime.usmFree(buf1));
    assert(runtime.usmFree(buf2));
  }

  void test(const char *code) {
    OclModuleBuilder builder(parse(code));
    auto mod = gcGetOrReport(builder.build(runtime));
    exec(mod);

    float cpuBuf[size];
    assert(runtime.usmCpy(ctx, buf2, cpuBuf, size));
    gcGetOrReport(ctx.finish());

    for (unsigned i = 0; i < size; i++) {
      // std::cout << buf2[i] << " ";
      assert(buf2[i] == 4.0f);
    }
    // std::cout << "\n";

    for (float i : cpuBuf) {
      // std::cout << i << " ";
      assert(i == 4.0f);
    }
  }
};

template <unsigned N, unsigned M = N> struct TestMatmulAdd : TestBase {
  static constexpr unsigned size1 = N * M;
  static constexpr unsigned size2 = M * M;
  cl_half *buf0 = gcGetOrReport(runtime.usmNewDev<cl_half>(size1));
  cl_half *buf1 = gcGetOrReport(runtime.usmNewDev<cl_half>(size2));
  cl_half *buf2 = gcGetOrReport(runtime.usmNewShared<cl_half>(size1));

  explicit TestMatmulAdd() {
    cl_half cpuBuf[size2];
    std::fill(cpuBuf, cpuBuf + size2, 14336);
    assert(runtime.usmCpy(ctx, cpuBuf, buf0, size1));
    assert(runtime.usmCpy(ctx, cpuBuf, buf1, size2));
    gcGetOrReport(ctx.finish());
  }

  ~TestMatmulAdd() override {
    assert(runtime.usmFree(buf0));
    assert(runtime.usmFree(buf1));
    assert(runtime.usmFree(buf2));
  }

  void test(const char *code) {
    OclModuleBuilder builder(parse(code));
    auto mod = gcGetOrReport(builder.build(runtime));
    exec(mod);

    gcGetOrReport(ctx.finish());
    for (unsigned i = 0; i < size1; i++) {
      // std::cout << buf2[i] << " ";
      assert(buf2[i] == 21512);
    }
    // std::cout << "\n";
  }
};

TEST(GpuOclRuntime, TestAddStatic) {
  struct TestAddStatic1 : TestAdd<64> {
    void exec(std::shared_ptr<const OclModule> &mod) override {
      assert(mod->isStatic);
      StaticExecutor<3> exec(mod);
      exec(ctx, buf0, buf1, buf2);
      // Check if the executor is allocated on the stack
      assert(exec.isSmall());
    }
  } test1;
  test1.test(addStatic);

  struct TestAddStatic2 : TestAdd<64> {
    void exec(std::shared_ptr<const OclModule> &mod) override {
      assert(mod->isStatic);
      StaticExecutor<3> exec(mod);
      exec.arg(buf0);
      exec.arg(buf1);
      exec.arg(buf2);
      // Check if the executor is allocated on the stack
      assert(exec.isSmall());
      exec(ctx);
    }
  } test2;
  test2.test(addStatic);
}

TEST(GpuOclRuntime, TestAddDynamic) {
  GTEST_SKIP() << "Dynamic shapes are not yet supported";
  struct TestAddDynamic : TestAdd<32, 64> {
    void exec(std::shared_ptr<const OclModule> &mod) override {
      assert(!mod->isStatic);
      int64_t shape[] = {32, 64};
      int64_t strides[] = {64, 1};
      DynamicExecutor<24> exec(mod);
      exec.arg(buf0, 2, shape, strides);
      exec.arg(buf1, 2, shape, strides);
      exec.arg(buf2, 2, shape, strides);
      exec(ctx);
      // Check if the executor is allocated on the stack
      assert(exec.isSmall());
    }
  } test;
  test.test(addDynamic);
}

TEST(GpuOclRuntime, TestMatmulAddStatic) {
  struct Test : TestMatmulAdd<128, 256> {
    void exec(std::shared_ptr<const OclModule> &mod) override {
      assert(mod->isStatic);
      StaticExecutor<3> exec(mod);
      exec(ctx, buf0, buf1, buf2);
      assert(exec.isSmall());
    }
  } test;
  test.test(matmulAddStatic);
}
