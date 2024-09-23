//===-- BrgemmOnednn.cpp - BRGEMM Onednn Implementation ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <math.h>
#include <memory>
#include <shared_mutex>
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

#include "gc/ExecutionEngine/CPURuntime/Microkernel/BrgemmInterface.h"

#if !defined(GC_ENABLE_RUNTIME_NAIVE_BRGEMM)

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

using read_lock_guard_t = std::shared_lock<std::shared_mutex>;
using write_lock_guard_t = std::unique_lock<std::shared_mutex>;
static std::shared_mutex g_brgemm_lock;

struct brgemm_cache_info_t {
  brgemm_desc_t desc;
  brgemm_kernel_t *kernel;
  std::shared_ptr<char[]> palette;
};

static std::vector<brgemm_cache_info_t> g_cache;

// TODO(haixin): use syscall to determine page size?
static constexpr size_t SCRATCH_SIZE = 2 * 4096;
// TODO(haixin): need to use custom thread management for scratch in the future?
static thread_local char scratch[SCRATCH_SIZE] = {0};

static std::unordered_map<int64_t, brgemm_cache_info_t> &get_tl_cache() {
  thread_local std::unordered_map<int64_t, brgemm_cache_info_t> tl_cache;
  return tl_cache;
}

extern "C" {

int64_t dnnl_brgemm_dispatch(int64_t M, int64_t N, int64_t K, int64_t LDA,
                             int64_t LDB, int64_t LDC, int64_t stride_a,
                             int64_t stride_b, float beta, int64_t dtypeA,
                             int64_t dtypeB) {
  brgemm_desc_t desc;
  brgemm_kernel_t *kernel;

  auto dnnl_dtypeA = static_cast<dnnl_data_type_t>(dtypeA);
  auto dnnl_dtypeB = static_cast<dnnl_data_type_t>(dtypeB);
  int64_t dtypeA_size = dnnl::impl::types::data_type_size(dnnl_dtypeA);
  int64_t dtypeB_size = dnnl::impl::types::data_type_size(dnnl_dtypeB);
  brgemm_strides_t stride_info{stride_a * dtypeA_size, stride_b * dtypeB_size};

  dnnl::impl::status_t status = brgemm_desc_init(
      &desc, cpu_isa_t::isa_undef, brgemm_batch_kind_t::brgemm_strd,
      dnnl_dtypeA, dnnl_dtypeB, /*transA=*/false, /*transB=*/false,
      brgemm_layout_t::brgemm_row_major, 1.0f, beta, LDA, LDB, LDC, M, N, K,
      &stride_info);
  assert(status == dnnl::impl::status::success &&
         "Failed to initialize BRGEMM descriptor");

  status = brgemm_kernel_create(&kernel, desc);
  assert(status == dnnl::impl::status::success &&
         "Failed to JIT BRGEMM kernel");

  brgemm_attr_t dnnl_attrs;
  brgemm_desc_set_attr(&desc, dnnl_attrs);

  // TODO(haixin): Reuse identical palettes across kernels
  std::shared_ptr<char[]> palette_buffer(new char[PALETTE_SIZE],
                                         std::default_delete<char[]>());
  if (desc.is_tmm) {
    dnnl::impl::status_t status = brgemm_init_tiles(desc, palette_buffer.get());
    assert(status == dnnl::impl::status::success &&
           "Failed to initialize palette for BRGEMM");
  }

  write_lock_guard_t g(g_brgemm_lock);
  g_cache.push_back(brgemm_cache_info_t{desc, kernel, palette_buffer});
  return g_cache.size() - 1;
}

void dnnl_brgemm_tileconfig(int64_t kernel_idx) {
  assert(kernel_idx >= 0 && "Invalid kernel handler");
  auto &tl_cache = get_tl_cache();
  auto it = tl_cache.find(kernel_idx);
  if (it == tl_cache.end()) {
    read_lock_guard_t g(g_brgemm_lock);
    assert(kernel_idx < (int64_t)g_cache.size() && "Invalid kernel handler");
    it = tl_cache.insert({kernel_idx, g_cache[kernel_idx]}).first;
  }
  brgemm_desc_t &desc = it->second.desc;
  char *palette_buffer = it->second.palette.get();

  if (!desc.is_tmm) {
    return;
  }

  assert(palette_buffer != nullptr && "Invalid palette for BRGEMM kernel");
  amx_tile_configure(palette_buffer);
}

void dnnl_brgemm_tilerelease() {
  if (!mayiuse(avx512_core_amx)) {
    return;
  }

  amx_tile_release();
}

void dnnl_brgemm_execute(int64_t kernel_idx, void *A, uint64_t A_offset,
                         void *B, uint64_t B_offset, void *C, uint64_t C_offset,
                         int num) {
  auto &tl_cache = get_tl_cache();
  if (tl_cache.find(kernel_idx) == tl_cache.end()) {
    read_lock_guard_t g(g_brgemm_lock);
    assert(kernel_idx >= 0 && kernel_idx < (int64_t)g_cache.size() &&
           "Invalid kernel handler");
    auto updated_cache =
        tl_cache.insert(std::make_pair(kernel_idx, g_cache[kernel_idx]));
    assert(updated_cache.second && "insert into thread local cache");
  }
  auto it = tl_cache.find(kernel_idx);
  brgemm_kernel_t *kernel = it->second.kernel;
  brgemm_desc_t *desc_ptr = &it->second.desc;

  assert(kernel && "Invalid brgemm kernel pointer");
  assert(desc_ptr && "Invalid brgemm descriptor pointer");

  size_t A_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc_ptr->dt_a) * A_offset;
  size_t B_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc_ptr->dt_b) * B_offset;
  size_t C_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc_ptr->dt_c) * C_offset;

  char *A_arith = (char *)A;
  char *B_arith = (char *)B;
  char *C_arith = (char *)C;
  brgemm_kernel_execute(kernel, num, (void *)(A_arith + A_offset_in_bytes),
                        (void *)(B_arith + B_offset_in_bytes), nullptr,
                        (void *)(C_arith + C_offset_in_bytes), (void *)scratch);
}
}

#endif
