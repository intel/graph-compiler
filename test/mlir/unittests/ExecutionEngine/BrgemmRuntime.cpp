//===-- BrgemmRuntime.cpp - Brgemm Runtime ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include <cstdlib>
#include <cstring>

#include "gc/ExecutionEngine/CPURuntime/Microkernel/BrgemmInterface.h"

extern "C" {
extern int gc_runtime_keep_alive;
}

template <typename T> inline T randomFP(int range = 1, float delta = 0.0f) {
  float fraction = (float)(rand()) / (float)(RAND_MAX);
  return T(fraction * range - delta);
}

template <typename T>
inline void randomInitWithFP(T *buffer, size_t size, int range = 1,
                             float delta = 0.0f) {
  for (size_t index = 0; index < size; index++)
    buffer[index] = randomFP<T>(range, delta);
}

template <typename T>
inline bool compareDataFP(T *ref, T *dst, size_t size, float rtol = 1e-4f,
                          float atol = 1e-6f) {
  for (size_t index = 0; index < size; index++) {
    const float ref_f32 = static_cast<float>(ref[index]);
    const float dst_f32 = static_cast<float>(dst[index]);
    const double diff_f32 = dst_f32 - ref_f32;
    const double gap = double(rtol) * (std::abs(ref_f32) > std::abs(dst_f32)
                                           ? std::abs(ref_f32)
                                           : std::abs(dst_f32)) +
                       atol;
    bool good = std::abs(diff_f32) <= gap;
    EXPECT_TRUE(good) << "Index: " << index << ", ref_f32=" << ref_f32
                      << ", dst_f32=" << dst_f32;
    if (!good)
      return false;
  }
  return true;
}

template <typename T>
inline void testBrgemmRuntimeFP(int batch, int M, int N, int K, int LDA,
                                int LDB, int LDC, int strideA, int strideB,
                                float beta) {
  using dnnl_f32_enum_val_t = std::integral_constant<int, 3>;
  using dnnl_bf16_enum_val_t = std::integral_constant<int, 2>;
  constexpr int dtypeA =
      std::conditional<std::is_same<T, float>::value, dnnl_f32_enum_val_t,
                       dnnl_bf16_enum_val_t>::type::value;
  constexpr int dtypeB = dtypeA;

  T A[batch * M * K];
  T B[batch * K * N];
  float refC[M * N];

  randomInitWithFP<T>(A, batch * M * K, 10, 10.0f);
  randomInitWithFP<T>(B, batch * K * N, 10, 10.0f);
  randomInitWithFP<float>(refC, M * N, 10, 10.0f);

  float dstC[M * N];
  memcpy(dstC, refC, sizeof(float) * M * N);

  // Calculate reference
  auto refHandle = dnnl_brgemm_dispatch_naive(M, N, K, LDA, LDB, LDC, strideA,
                                              strideB, beta, dtypeA, dtypeB);
  dnnl_brgemm_execute_naive(refHandle, A, 0, B, 0, refC, 0, batch);

  // Calculate destination
  auto dstHandle = dnnl_brgemm_dispatch(M, N, K, LDA, LDB, LDC, strideA,
                                        strideB, beta, dtypeA, dtypeB);
  dnnl_brgemm_tileconfig(dstHandle);
  dnnl_brgemm_execute(dstHandle, A, 0, B, 0, dstC, 0, batch);
  dnnl_brgemm_tilerelease();

  ASSERT_TRUE(compareDataFP<float>(refC, dstC, M * N));
}

template <typename T>
inline void randomInitWithInt(T *buffer, size_t size, int range,
                              int delta = 0) {
  for (size_t index = 0; index < size; index++)
    buffer[index] = rand() % range - delta;
}

template <typename T> inline bool compareDataInt(T *ref, T *dst, size_t size) {
  for (size_t index = 0; index < size; index++) {
    bool good = ref[index] == dst[index];
    EXPECT_TRUE(good) << "Index: " << index << ", ref=" << ref[index]
                      << ", dst=" << dst[index];
    if (!good)
      return false;
  }
  return true;
}

inline void testBrgemmRuntimeInt(int batch, int M, int N, int K, int LDA,
                                 int LDB, int LDC, int strideA, int strideB,
                                 float beta) {
  constexpr int dtypeA = 6; // dnnl_u8 enum val
  constexpr int dtypeB = 5; // dnnl_s8 enum val

  uint8_t A[batch * M * K];
  int8_t B[batch * K * N];
  int32_t refC[M * N];

  randomInitWithInt(A, batch * M * K, 100, 100);
  randomInitWithInt(B, batch * K * N, 100, 100);
  randomInitWithInt(refC, M * N, 500, 500);

  int32_t dstC[M * N];
  memcpy(dstC, refC, sizeof(int32_t) * M * N);

  // Calculate reference
  auto refHandle = dnnl_brgemm_dispatch_naive(M, N, K, LDA, LDB, LDC, strideA,
                                              strideB, beta, dtypeA, dtypeB);
  dnnl_brgemm_execute_naive(refHandle, A, 0, B, 0, refC, 0, batch);

  // Calculate destination
  auto dstHandle = dnnl_brgemm_dispatch(M, N, K, LDA, LDB, LDC, strideA,
                                        strideB, beta, dtypeA, dtypeB);
  dnnl_brgemm_tileconfig(dstHandle);
  dnnl_brgemm_execute(dstHandle, A, 0, B, 0, dstC, 0, batch);
  dnnl_brgemm_tilerelease();

  ASSERT_TRUE(compareDataInt(refC, dstC, M * N));
}

TEST(ExecutionEngine, TestBrgemmRuntimeF32) {
  gc_runtime_keep_alive = 0;

  srand(static_cast<unsigned>(time(nullptr)));

  constexpr int batch = 4;
  constexpr int M = 32, N = 32, K = 32;
  constexpr int LDA = 32, LDB = 32, LDC = 32;
  constexpr int strideA = 1024, strideB = 1024;

  testBrgemmRuntimeFP<float>(batch, M, N, K, LDA, LDB, LDC, strideA, strideB,
                             0.0f);
  testBrgemmRuntimeFP<float>(batch, M, N, K, LDA, LDB, LDC, strideA, strideB,
                             1.0f);
}

TEST(ExecutionEngine, TestBrgemmRuntimeBF16) {
  gc_runtime_keep_alive = 0;

  srand(static_cast<unsigned>(time(nullptr)));

  constexpr int batch = 4;
  constexpr int M = 32, N = 32, K = 32;
  constexpr int LDA = 32, LDB = 32, LDC = 32;
  constexpr int strideA = 1024, strideB = 1024;

  testBrgemmRuntimeFP<bf16_t>(batch, M, N, K, LDA, LDB, LDC, strideA, strideB,
                              0.0f);
  testBrgemmRuntimeFP<bf16_t>(batch, M, N, K, LDA, LDB, LDC, strideA, strideB,
                              1.0f);
}

TEST(ExecutionEngine, TestBrgemmRuntimeU8S8) {
  gc_runtime_keep_alive = 0;

  srand(static_cast<unsigned>(time(nullptr)));

  constexpr int batch = 4;
  constexpr int M = 32, N = 32, K = 32;
  constexpr int LDA = 32, LDB = 32, LDC = 32;
  constexpr int strideA = 1024, strideB = 1024;

  testBrgemmRuntimeInt(batch, M, N, K, LDA, LDB, LDC, strideA, strideB, 0.0f);
  testBrgemmRuntimeInt(batch, M, N, K, LDA, LDB, LDC, strideA, strideB, 1.0f);
}
