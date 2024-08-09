//===-- BrgemmInterface.h - The interfaces of runtime Brgemm ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_CPURUNTIME_MICROKERNEL_BRGEMMINTERFACE_H
#define GC_EXECUTIONENGINE_CPURUNTIME_MICROKERNEL_BRGEMMINTERFACE_H

extern "C" {

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

#endif
