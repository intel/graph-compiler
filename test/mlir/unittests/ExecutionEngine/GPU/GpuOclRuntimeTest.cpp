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

#include "mlir/ExecutionEngine/CRunnerUtils.h"
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

constexpr char rope1[] = R"mlir(
!dtype=i16
!input_memref_type=memref<2x7x32x128x!dtype>
!input_tensor_type=tensor<2x7x32x128x!dtype>
!output_memref_type=memref<2x32x7x128x!dtype>
!output_tensor_type=tensor<2x32x7x128x!dtype>
!cos_sin_cache_tensor_shrink_type=tensor<1x1x7x128x!dtype>
!cos_sin_cache_memref_type=memref<1x1x7x128x!dtype>
!cos_sin_cache_tensor_type=tensor<1x1x7x128x!dtype>
!pos_ids_memref_type=memref<1x7xindex>

module @fragment_name {

  func.func @rope1(%iinput: !input_memref_type, %ipos_ids: !pos_ids_memref_type, %out: !output_memref_type,
                  %cos_cache : !cos_sin_cache_memref_type, %sin_cache : !cos_sin_cache_memref_type) {
      %input = bufferization.to_tensor %iinput restrict : !input_memref_type
      %cos_cache_tensor = bufferization.to_tensor %cos_cache restrict : !cos_sin_cache_memref_type
      %sin_cache_tensor = bufferization.to_tensor %sin_cache restrict : !cos_sin_cache_memref_type
      %pos_ids = bufferization.to_tensor %ipos_ids restrict : !pos_ids_memref_type
      %3 = tensor.empty(): !output_tensor_type
      %transpose_in =  linalg.transpose ins(%input: !input_tensor_type) outs(%3:!output_tensor_type)  permutation = [0, 2, 1, 3]

      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %cos_cache_slice = tensor.extract_slice %cos_cache_tensor[0, 0, 0, 0] [1, 1, 7, 128] [1, 1, 1, 1] : !cos_sin_cache_tensor_type to !cos_sin_cache_tensor_shrink_type
      %cos_cache_slice2 = tensor.collapse_shape %cos_cache_slice [[0, 1], [2],[3]] : tensor<1x1x7x128x!dtype> into tensor<1x7x128x!dtype>
      %cos_cache_slice3 = tensor.collapse_shape %cos_cache_slice2 [[0, 1], [2]] : tensor<1x7x128x!dtype> into tensor<7x128x!dtype>
      %pos_ids_index=tensor.expand_shape %pos_ids [[0],[1,2]] output_shape [1, 7, 1] : tensor<1x7xindex> into tensor<1x7x1xindex>

      %cos_cache_slice4 = tensor.gather %cos_cache_slice3[%pos_ids_index] gather_dims([0]) : (tensor<7x128x!dtype>, tensor<1x7x1xindex>) -> tensor<1x7x128x!dtype>

      %cos_cache_slice5 = tensor.expand_shape %cos_cache_slice4 [[0,1],[2],[3]] output_shape [1,1,7,128] : tensor<1x7x128x!dtype> into tensor<1x1x7x128x!dtype>
      %cos_cache_slice6 = tensor.collapse_shape %cos_cache_slice5 [[0,1,2],[3]] : tensor<1x1x7x128x!dtype> into tensor<7x128x!dtype>


      %cos_cache_slice7 = linalg.broadcast ins(%cos_cache_slice6: tensor<7x128x!dtype>) outs(%3: !output_tensor_type) dimensions = [0, 1]
      %input_apply_cos_cache = linalg.mul ins(%transpose_in, %cos_cache_slice7:  !output_tensor_type, !output_tensor_type) outs(%3: !output_tensor_type) -> !output_tensor_type

      %head_dim = tensor.dim  %transpose_in, %c3 : !output_tensor_type
      %c2 = arith.constant 2 : index
      %half_head_dim = arith.floordivsi %head_dim, %c2 : index
      %transpose_input_first_half = tensor.extract_slice %transpose_in[0, 0, 0, 0][2, 32, 7, 64][1,1,1,1] : !output_tensor_type to tensor<2x32x7x64x!dtype>
      %transpose_input_second_half = tensor.extract_slice %transpose_in[0, 0, 0, %half_head_dim][2, 32, 7, 64][1,1,1,1] : !output_tensor_type to tensor<2x32x7x64x!dtype>
      %cnegative1 = arith.constant dense<-1> : tensor<2x32x7x64x!dtype>
      %empty_tensor = tensor.empty() : tensor<2x32x7x64x!dtype>
      %transpose_input_second_half_opposite = linalg.mul ins(%transpose_input_second_half, %cnegative1:  tensor<2x32x7x64x!dtype>, tensor<2x32x7x64x!dtype>) outs(%empty_tensor: tensor<2x32x7x64x!dtype>) -> tensor<2x32x7x64x!dtype>

      %transformed_input = tensor.concat dim(3) %transpose_input_second_half_opposite, %transpose_input_first_half : (tensor<2x32x7x64x!dtype>, tensor<2x32x7x64x!dtype>) -> !output_tensor_type

      %sin_cache_slice = tensor.extract_slice %sin_cache_tensor[0, 0, 0, 0] [1, 1, 7, 128] [1, 1, 1, 1] : !cos_sin_cache_tensor_type to !cos_sin_cache_tensor_shrink_type
      %sin_cache_slice2 = tensor.collapse_shape %sin_cache_slice [[0, 1], [2],[3]] : tensor<1x1x7x128x!dtype> into tensor<1x7x128x!dtype>
      %sin_cache_slice3 = tensor.collapse_shape %sin_cache_slice2 [[0, 1], [2]] : tensor<1x7x128x!dtype> into tensor<7x128x!dtype>
      %sin_cache_slice4 = tensor.gather %sin_cache_slice3[%pos_ids_index] gather_dims([0]) : (tensor<7x128x!dtype>, tensor<1x7x1xindex>) -> tensor<1x7x128x!dtype>

      %sin_cache_slice5 = tensor.expand_shape %sin_cache_slice4 [[0,1],[2],[3]] output_shape [1,1,7,128] : tensor<1x7x128x!dtype> into tensor<1x1x7x128x!dtype>
      %sin_cache_slice6 = tensor.collapse_shape %sin_cache_slice5 [[0,1,2],[3]] : tensor<1x1x7x128x!dtype> into tensor<7x128x!dtype>
      %sin_cache_slice7 = linalg.broadcast ins(%sin_cache_slice6: tensor<7x128x!dtype>) outs(%3: !output_tensor_type) dimensions = [0, 1]
      %input_apply_sin_cache = linalg.mul ins(%transformed_input, %sin_cache_slice7:  !output_tensor_type, !output_tensor_type) outs(%3: !output_tensor_type) -> !output_tensor_type

      %result = linalg.add ins(%input_apply_cos_cache, %input_apply_sin_cache:  !output_tensor_type, !output_tensor_type) outs(%3: !output_tensor_type) -> !output_tensor_type
      bufferization.materialize_in_destination %result in restrict writable %out : (!output_tensor_type, !output_memref_type) -> ()
      return
  }
}
)mlir";
constexpr char rope2[] = R"mlir(
!dtype=i16
!input_memref_type=memref<2x7x32x128x!dtype>
!input_tensor_type=tensor<2x7x32x128x!dtype>
!output_memref_type=memref<2x32x7x128x!dtype>
!output_tensor_type=tensor<2x32x7x128x!dtype>
!cos_sin_cache_memref_type=memref<1x1x7x128x!dtype>
!cos_sin_cache_tensor_type=tensor<1x1x7x128x!dtype>
!pos_ids_memref_type=memref<1x7xindex>

module @fragment_name {

  func.func @rope2(%iinput: !input_memref_type, %ipos_ids: !pos_ids_memref_type, %out: !output_memref_type,
                  %cos_cache: !cos_sin_cache_memref_type, %sin_cache: !cos_sin_cache_memref_type) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %cm1 = arith.constant -1 : !dtype

      %input = bufferization.to_tensor %iinput restrict : !input_memref_type
      %cos_cache_tensor = bufferization.to_tensor %cos_cache restrict : !cos_sin_cache_memref_type
      %sin_cache_tensor = bufferization.to_tensor %sin_cache restrict : !cos_sin_cache_memref_type
      %pos_ids = bufferization.to_tensor %ipos_ids restrict : !pos_ids_memref_type
      %tmp = tensor.empty(): !output_tensor_type

      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%tmp : !output_tensor_type) {
        ^bb0(%ignore: !dtype):
          %i0 = linalg.index 0 : index
          %i1 = linalg.index 1 : index
          %i2 = linalg.index 2 : index
          %i3 = linalg.index 3 : index
          %pos = tensor.extract %pos_ids[%c0, %i2] : tensor<1x7xindex>
          %cos = tensor.extract %cos_cache_tensor[%c0, %c0, %pos, %i3] : !cos_sin_cache_tensor_type
          %sin = tensor.extract %sin_cache_tensor[%c0, %c0, %pos, %i3] : !cos_sin_cache_tensor_type
          %in = tensor.extract %input[%i0, %i2, %i1, %i3] : !input_tensor_type
          %cos_val = arith.muli %cos, %in : !dtype

          %cond = arith.cmpi slt, %i3, %c64 : index
          %sin_val = scf.if %cond -> (!dtype) {
            %i3_plus_64 = arith.addi %i3, %c64 : index
            %v = tensor.extract %input[%i0, %i2, %i1, %i3_plus_64] : !input_tensor_type
            %minusv = arith.muli %cm1, %v : !dtype
            %mul = arith.muli %sin, %minusv : !dtype
            scf.yield %mul : !dtype
          } else {
            %i3_minus_64 = arith.addi %i3, %c64 : index
            %v = tensor.extract %input[%i0, %i2, %i1, %i3_minus_64] : !input_tensor_type
            %mul = arith.muli %sin, %v : !dtype
            scf.yield %mul : !dtype
          }

          %sum = arith.addi %cos_val, %sin_val : !dtype
          linalg.yield %sum : !dtype
      } -> !output_tensor_type

      bufferization.materialize_in_destination %result in restrict writable %out : (!output_tensor_type, !output_memref_type) -> ()
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

TEST(GpuOclRuntime, TestRope) {
  struct Test : TestBase {
    int16_t *inputs;
    int16_t *ipos;
    int16_t *outputs;
    int16_t *cosCache;
    int16_t *sinCache;

    explicit Test(bool sharedMem) {
      auto memAlloc = [&](size_t size) {
        return gcGetOrReport(sharedMem ? runtime.usmNewShared<int16_t>(size)
                                       : runtime.usmNewDev<int16_t>(size));
      };

      {
        int16_t inputsCpu[2][7][32][128];
        size_t bufLen = sizeof(inputsCpu) / sizeof(int16_t);
        inputs = memAlloc(bufLen);
        outputs = memAlloc(bufLen);
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 32; k++) {
              for (int l = 0; l < 128; l++) {
                inputsCpu[i][j][k][l] = static_cast<int16_t>(i + j + l + 1);
              }
            }
          }
        }
        assert(runtime.usmCpy(ctx, inputsCpu, inputs, bufLen));
      }

      {
        int16_t cosCacheCpu[1][1][7][128];
        int16_t sinCacheCpu[1][1][7][128];
        size_t bufLen = sizeof(cosCacheCpu) / sizeof(int16_t);
        cosCache = memAlloc(bufLen);
        sinCache = memAlloc(bufLen);
        for (int i = 0; i < 1; i++) {
          for (int j = 0; j < 1; j++) {
            for (int k = 0; k < 7; k++) {
              for (int l = 0; l < 128; l++) {
                cosCacheCpu[i][j][k][l] = static_cast<int16_t>(i + j + l + 3);
                sinCacheCpu[i][j][k][l] = static_cast<int16_t>(i + j + l + 2);
              }
            }
          }
        }
        assert(runtime.usmCpy(ctx, cosCacheCpu, cosCache, bufLen));
        assert(runtime.usmCpy(ctx, sinCacheCpu, sinCache, bufLen));
      }

      int16_t iposCpu[]{6, 5, 4, 3, 2, 1, 0};
      ipos = memAlloc(7);
      assert(runtime.usmCpy(ctx, iposCpu, ipos,
                            sizeof(iposCpu) / sizeof(int16_t)));
    }

    ~Test() override {
      assert(runtime.usmFree(inputs));
      assert(runtime.usmFree(ipos));
      assert(runtime.usmFree(outputs));
      assert(runtime.usmFree(cosCache));
      assert(runtime.usmFree(sinCache));
    }

    void exec(std::shared_ptr<const OclModule> &mod) override {
      StaticExecutor<3> exec(mod);
      exec.arg(inputs);
      exec.arg(ipos);
      exec.arg(outputs);
      exec.arg(cosCache);
      exec.arg(sinCache);

      auto start =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
      exec(ctx);
      assert(ctx.finish());
      auto end =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
      gcLogD("Execution time: ", end - start, " ns");
    }

    void test(const char *code, int16_t (&outputsCpu)[2][32][7][128]) {
      OclModuleBuilderOpts opts;
      opts.symbolMaper = [](llvm::orc::SymbolMap &map,
                            llvm::orc::MangleAndInterner &interner) {
        map.try_emplace(interner("memrefCopy"),
                        llvm::orc::ExecutorAddr::fromPtr(&memrefCopy),
                        llvm::JITSymbolFlags::Exported);
      };
      OclModuleBuilder builder(parse(code), opts);
      auto mod = gcGetOrReport(builder.build(runtime));
      exec(mod);

      assert(runtime.usmCpy(ctx, outputs, outputsCpu,
                            sizeof(outputsCpu) / sizeof(int16_t)));
      assert(ctx.finish());
    }
  };

  int16_t outputs1[2][32][7][128];
  int16_t outputs2[2][32][7][128];

  {
    Test test(true);
    test.test(rope1, outputs1);
  }
  {
    Test test(false);
    test.test(rope2, outputs2);
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 32; j++) {
      for (int k = 0; k < 7; k++) {
        for (int l = 0; l < 128; l++) {
          // std::cout << outputs1[i][j][k][l] << " ";
          assert(outputs1[i][j][k][l] == outputs2[i][j][k][l]);
        }
        // std::cout << "\n";
      }
    }
  }
}
