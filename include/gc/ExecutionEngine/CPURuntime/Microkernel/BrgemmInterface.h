//===-- BrgemmInterface.h - The interfaces of runtime Brgemm ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_CPURUNTIME_MICROKERNEL_BRGEMMINTERFACE_H
#define GC_EXECUTIONENGINE_CPURUNTIME_MICROKERNEL_BRGEMMINTERFACE_H

#include <cmath>

extern "C" {
// Runtime interfaces

/**
 * Dispatch (JIT) the Brgemm kernel based on given parameters using DNNL
 * Inputs:
 * 	M, N, K: The size of Brgemm dims, given in element size;
 * 	LDA, LDB, LDC: The stride of leading dim of
 * 	               each Brgemm matrix, given in element size;
 * 	stride_a, stride_b: The stride of batch of Brgemm
 * 	                    input A & B, given in element size;
 * 	dtypeA, dtypeB: The dtype of Brgemm input A and B,
 * 	                given in dnnl type value.
 * Output: A handle of dispatched kernel.
 */
int64_t dnnl_brgemm_dispatch(int64_t M, int64_t N, int64_t K, int64_t LDA,
                             int64_t LDB, int64_t LDC, int64_t stride_a,
                             int64_t stride_b, float beta, int64_t dtypeA,
                             int64_t dtypeB);

/**
 * Config the AMX tile context for given kernel.
 * Inputs: A handle of dispatched kernel.
 * Output: None.
 */
void dnnl_brgemm_tileconfig(int64_t kernel);

/**
 * Release the current AMX tile context.
 * Inputs: None.
 * Output: None.
 */
void dnnl_brgemm_tilerelease();

/**
 * Execute the given kernel with given parameters.
 * Inputs:
 * 	kernel: A handle of dispatched kernel;
 * 	A, A_offset, B, B_offset, C, C_offset:
 * 	      Pointers and starting offset of each Brgemm matrix;
 * 	num: Batch size of Brgemm.
 * Output: None.
 */
void dnnl_brgemm_execute(int64_t kernel, void *A, uint64_t A_offset, void *B,
                         uint64_t B_offset, void *C, uint64_t C_offset,
                         int num);
}

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

// Naive implementation of `dnnl_brgemm_dispatch`
int64_t dnnl_brgemm_dispatch_naive(int64_t M, int64_t N, int64_t K, int64_t LDA,
                                   int64_t LDB, int64_t LDC, int64_t stride_a,
                                   int64_t stride_b, float beta, int64_t dtypeA,
                                   int64_t dtypeB);

// Naive implementation of `dnnl_brgemm_execute`
void dnnl_brgemm_execute_naive(int64_t kernel, void *A, uint64_t A_offset,
                               void *B, uint64_t B_offset, void *C,
                               uint64_t C_offset, int num);

#endif
