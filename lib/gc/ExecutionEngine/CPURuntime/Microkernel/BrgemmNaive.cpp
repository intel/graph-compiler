//===-- BrgemmNaive.cpp - BRGEMM Naive Implementation -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "oneapi/dnnl/dnnl_types.h"

namespace {

struct bf16_t {
  uint16_t storage_;
  union caster_t {
    uint32_t vl;
    float vf;
  };
  operator float() const {
    caster_t val;
    val.vl = uint32_t(storage_) << 16;
    return val.vf;
  }
  bool operator==(const bf16_t &compare_to) const {
    return storage_ == compare_to.storage_;
  }
  bool operator!=(const bf16_t &compare_to) const {
    return storage_ != compare_to.storage_;
  }
  bf16_t(float v) {
    if (std::isnan(v)) {
      storage_ = UINT32_C(0x7FC0);
    } else {
      caster_t caster;
      caster.vf = v;
      uint32_t rounding_bias = ((caster.vl >> 16) & 1) + UINT32_C(0x7FFF);
      storage_ = static_cast<uint16_t>((caster.vl + rounding_bias) >> 16);
    }
  }
  bf16_t() : storage_(0) {}
  inline static bf16_t from_storage(uint16_t v) {
    bf16_t ret;
    ret.storage_ = v;
    return ret;
  }
};

struct brgemm_params_t {
  int64_t M, N, K;
  int64_t LDA, LDB, LDC;
  int64_t stride_a, stride_b;
  float beta;
  int64_t dtypeA, dtypeB;
  brgemm_params_t(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
                  int64_t ldc, int64_t sa, int64_t sb, float b, int64_t da,
                  int64_t db)
      : M(m), N(n), K(k), LDA(lda), LDB(ldb), LDC(ldc), stride_a(sa),
        stride_b(sb), beta(b), dtypeA(da), dtypeB(db) {}
};

}; // namespace

static int naive_brgemm_execute_fp32(brgemm_params_t params, void *A,
                                     uint64_t A_offset, void *B,
                                     uint64_t B_offset, void *C,
                                     uint64_t C_offset, int num) {
  float *Abuf = (float *)A;
  float *Bbuf = (float *)B;
  float *Cbuf = (float *)C;
  Abuf += A_offset;
  Bbuf += B_offset;
  Cbuf += C_offset;
  for (int i = 0; i < num; i++) {
    // a is MxK
    for (int m = 0; m < params.M; m++) {
      for (int n = 0; n < params.N; n++) {
        for (int k = 0; k < params.K; k++) {
          Cbuf[m * params.LDC + n] +=
              Abuf[m * params.LDA + k] * Bbuf[k * params.LDB + n];
        }
      }
    }
    Abuf += params.stride_a;
    Bbuf += params.stride_b;
  }
  return 0;
}

static void naive_brgemm_execute_bf16(brgemm_params_t params, void *A,
                                      uint64_t A_offset, void *B,
                                      uint64_t B_offset, void *C,
                                      uint64_t C_offset, int num) {
  bf16_t *Abuf = (bf16_t *)A;
  bf16_t *Bbuf = (bf16_t *)B;
  float *Cbuf = (float *)C;
  Abuf += A_offset;
  Bbuf += B_offset;
  Cbuf += C_offset;
  for (int i = 0; i < num; i++) {
    // a is MxK
    // b is KxNx2k (vnni format)
    for (int m = 0; m < params.M; m++) {
      for (int n = 0; n < params.N; n++) {
        for (int k = 0; k < params.K; k += 2) {
          Cbuf[m * params.LDC + n] +=
              Abuf[m * params.LDA + k] * Bbuf[k * params.LDB + 2 * n];
          if (k + 1 < params.K) {
            // simulate vnni padding
            Cbuf[m * params.LDC + n] +=
                Abuf[m * params.LDA + k + 1] * Bbuf[k * params.LDB + 2 * n + 1];
          }
        }
      }
    }
    Abuf += params.stride_a;
    Bbuf += params.stride_b;
  }
}

template <typename TA, typename TB>
static void naive_brgemm_execute_int8(brgemm_params_t params, void *A,
                                      uint64_t A_offset, void *B,
                                      uint64_t B_offset, void *C,
                                      uint64_t C_offset, int num) {
  TA *Abuf = (TA *)A;
  TB *Bbuf = (TB *)B;
  int32_t *Cbuf = (int32_t *)C;
  Abuf += A_offset;
  Bbuf += B_offset;
  Cbuf += C_offset;
  for (int i = 0; i < num; i++) {
    // a is MxK
    // b is KxNx4k (vnni format)
    for (int m = 0; m < params.M; m++) {
      for (int n = 0; n < params.N; n++) {
        for (int k = 0; k < params.K; k += 4) {
          Cbuf[m * params.LDC + n] +=
              Abuf[m * params.LDA + k] * Bbuf[k * params.LDB + 4 * n];
          if (k + 1 < params.K) {
            // simulate vnni padding
            Cbuf[m * params.LDC + n] +=
                Abuf[m * params.LDA + k + 1] * Bbuf[k * params.LDB + 4 * n + 1];
          }
          if (k + 2 < params.K) {
            Cbuf[m * params.LDC + n] +=
                Abuf[m * params.LDA + k + 2] * Bbuf[k * params.LDB + 4 * n + 2];
          }
          if (k + 3 < params.K) {
            Cbuf[m * params.LDC + n] +=
                Abuf[m * params.LDA + k + 3] * Bbuf[k * params.LDB + 4 * n + 3];
          }
        }
      }
    }
    Abuf += params.stride_a;
    Bbuf += params.stride_b;
  }
}

static std::vector<brgemm_params_t> brgemm_list;

extern "C" {

int64_t dnnl_brgemm_dispatch(int64_t M, int64_t N, int64_t K, int64_t LDA,
                             int64_t LDB, int64_t LDC, int64_t stride_a,
                             int64_t stride_b, float beta, int64_t dtypeA,
                             int64_t dtypeB) {
  // simply store the given parameters for naive BRGEMM
  brgemm_list.emplace_back(brgemm_params_t(M, N, K, LDA, LDB, LDC, stride_a,
                                           stride_b, beta, dtypeA, dtypeB));
  // std::cout << ">>>>> dnnl_brgemm_dispatch: " << brgemm_list.size() - 1 <<
  // std::endl;
  return brgemm_list.size() - 1;
}

void dnnl_brgemm_tileconfig(int64_t kernel) { return; }

void dnnl_brgemm_tilerelease() { return; }

void dnnl_brgemm_execute(int64_t kernel, void *A, uint64_t A_offset, void *B,
                         uint64_t B_offset, void *C, uint64_t C_offset,
                         int num) {
  assert(kernel >= 0 && kernel < (int64_t)brgemm_list.size() &&
         "Invalid kernel handler");
  brgemm_params_t &params = brgemm_list[kernel];
  if (params.dtypeA == static_cast<int64_t>(dnnl_f32) &&
      params.dtypeB == static_cast<int64_t>(dnnl_f32)) {
    // std::cout << ">>>>> dnnl_brgemm_execute_f32: " << kernel << std::endl;
    naive_brgemm_execute_fp32(params, A, A_offset, B, B_offset, C, C_offset,
                              num);
  } else if (params.dtypeA == static_cast<int64_t>(dnnl_bf16) &&
             params.dtypeB == static_cast<int64_t>(dnnl_bf16)) {
    // std::cout << ">>>>> dnnl_brgemm_execute_bf16: " << kernel << std::endl;
    naive_brgemm_execute_bf16(params, A, A_offset, B, B_offset, C, C_offset,
                              num);
  } else if (params.dtypeA == static_cast<int64_t>(dnnl_s8) &&
             params.dtypeB == static_cast<int64_t>(dnnl_s8)) {
    // std::cout << ">>>>> dnnl_brgemm_execute_s8s8: " << kernel << std::endl;
    naive_brgemm_execute_int8<int8_t, int8_t>(params, A, A_offset, B, B_offset,
                                              C, C_offset, num);
  } else if (params.dtypeA == static_cast<int64_t>(dnnl_s8) &&
             params.dtypeB == static_cast<int64_t>(dnnl_u8)) {
    // std::cout << ">>>>> dnnl_brgemm_execute_s8u8: " << kernel << std::endl;
    naive_brgemm_execute_int8<int8_t, uint8_t>(params, A, A_offset, B, B_offset,
                                               C, C_offset, num);
  } else if (params.dtypeA == static_cast<int64_t>(dnnl_u8) &&
             params.dtypeB == static_cast<int64_t>(dnnl_u8)) {
    // std::cout << ">>>>> dnnl_brgemm_execute_u8u8: " << kernel << std::endl;
    naive_brgemm_execute_int8<uint8_t, uint8_t>(params, A, A_offset, B,
                                                B_offset, C, C_offset, num);
  } else if (params.dtypeA == static_cast<int64_t>(dnnl_u8) &&
             params.dtypeB == static_cast<int64_t>(dnnl_s8)) {
    // std::cout << ">>>>> dnnl_brgemm_execute_u8s8: " << kernel << std::endl;
    naive_brgemm_execute_int8<uint8_t, int8_t>(params, A, A_offset, B, B_offset,
                                               C, C_offset, num);
  } else {
    assert(false && "unsupported input dtypes");
  }
}
}
