//===-- BrgemmNaive.cpp - BRGEMM Naive Implementation -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// manually include xbyak header here to avoid no-exception compile issue
#define XBYAK_NO_EXCEPTION
#include <cpu/x64/xbyak/xbyak.h> // NOLINT
#undef XBYAK_NO_EXCEPTION

#include <cpu/x64/amx_tile_configure.hpp>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>

using namespace dnnl::impl::cpu::x64;

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
// dummy definition for DNNL lite linkage
__attribute__((weak)) void print_verbose_header() {}
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

static constexpr int PALETTE_SIZE = 64;
static std::vector<brgemm_t> brgemm_desc_list;
static std::vector<brgemm_kernel_t *> brgemm_kernel_list;

extern "C" {

int64_t dnnl_brgemm_dispatch(int64_t M, int64_t N, int64_t K, int64_t LDA,
                             int64_t LDB, int64_t LDC, int64_t stride_a,
                             int64_t stride_b, float beta, int64_t dtypeA,
                             int64_t dtypeB) {
  std::cout << ">>> Brgemm dispatch: " << std::endl;
  brgemm_desc_list.emplace_back(brgemm_t());
  brgemm_kernel_list.emplace_back(nullptr);

  brgemm_t &desc = brgemm_desc_list.back();
  auto &kernel = brgemm_kernel_list.back();
  brgemm_strides_t stride_info{stride_a, stride_b};

  dnnl::impl::status_t status = brgemm_desc_init(
      &desc, cpu_isa_t::isa_undef, brgemm_batch_kind_t::brgemm_strd,
      static_cast<dnnl_data_type_t>(dtypeA),
      static_cast<dnnl_data_type_t>(dtypeB), false, false,
      brgemm_layout_t::brgemm_row_major, 1.0f, beta, LDA, LDB, LDC, M, N, K,
      &stride_info);
  assert(status == dnnl::impl::status::success &&
         "Failed to initialize BRGEMM descriptor");

  status = brgemm_kernel_create(&kernel, desc);
  assert(status == dnnl::impl::status::success &&
         "Failed to JIT BRGEMM kernel");

  return brgemm_desc_list.size() - 1;
}

void dnnl_brgemm_tileconfig(int64_t kernel_idx) {
  assert(kernel_idx >= 0 && kernel_idx < (int64_t)brgemm_desc_list.size() &&
         "Invalid kernel handler");
  std::cout << ">>> Brgemm tileconfig: " << kernel_idx << std::endl;

  brgemm_t &desc = brgemm_desc_list[kernel_idx];
  if (!desc.is_tmm) {
    return;
  }

  char palette_buffer[PALETTE_SIZE];
  dnnl::impl::status_t status = brgemm_init_tiles(desc, palette_buffer);
  assert(status == dnnl::impl::status::success &&
         "Failed to initialize palette for BRGEMM");

  amx_tile_configure(palette_buffer);
}

void dnnl_brgemm_tilerelease() {
  if (!mayiuse(avx512_core_amx)) {
    return;
  }
  std::cout << ">>> Brgemm tilerelease" << std::endl;

  amx_tile_release();
}

void dnnl_brgemm_execute(int64_t kernel_idx, void *A, uint64_t A_offset,
                         void *B, uint64_t B_offset, void *C, uint64_t C_offset,
                         int num) {
  assert(kernel_idx >= 0 && kernel_idx < (int64_t)brgemm_desc_list.size() &&
         "Invalid kernel handler");

  std::cout << ">>> Brgemm Execute: " << kernel_idx << std::endl;
  brgemm_t &desc = brgemm_desc_list[kernel_idx];
  brgemm_kernel_t *kernel = brgemm_kernel_list[kernel_idx];

  size_t A_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc.dt_a) * A_offset;
  size_t B_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc.dt_b) * A_offset;
  size_t C_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc.dt_c) * A_offset;

#ifdef _WIN32
  // fix-me: (win32) impl
  static size_t scratch_size = 2 * 4096;
#else
  static size_t scratch_size = 2 * getpagesize();
#endif
  // TODO(haixin): use thread local buffer for scratch
  char *scratch = new char[scratch_size];
  brgemm_kernel_execute(kernel, num, A + A_offset_in_bytes,
                        B + B_offset_in_bytes, nullptr, C + C_offset_in_bytes,
                        (void *)scratch);
  delete scratch;
}
}
