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

static std::vector<brgemm_desc_t> g_brgemm_desc_list;
static std::vector<brgemm_kernel_t *> g_brgemm_kernel_list;
static std::vector<std::unique_ptr<char[]>> g_brgemm_palette;

struct brgemm_cache_info_t {
  std::shared_ptr<brgemm_desc_t> desc;
  std::shared_ptr<brgemm_kernel_t> kernel;
  std::shared_ptr<char> palette;

  brgemm_cache_info_t() = default;
  brgemm_cache_info_t(brgemm_desc_t *d, brgemm_kernel_t *k, char *p)
      : desc(d), kernel(k), palette(p) {}
  brgemm_cache_info_t &operator=(const brgemm_cache_info_t &other) {
    if (this != &other) {
      desc = other.desc;
      kernel = other.kernel;
      palette = other.palette;
    }
    return *this;
  }
};

class brgemm_cache_manager {
public:
  static brgemm_cache_manager &getInstance() {
    static thread_local brgemm_cache_manager instance;
    return instance;
  }

  void insertOrUpdate(int64_t key, const brgemm_cache_info_t &info) {
    cache_[key] = info;
  }

  bool tryGet(int64_t key, brgemm_cache_info_t &info) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      info = it->second;
      return true;
    }
    return false;
  }

private:
  brgemm_cache_manager() {}

  brgemm_cache_manager(const brgemm_cache_manager &) = delete;
  brgemm_cache_manager &operator=(const brgemm_cache_manager &) = delete;

  std::unordered_map<int64_t, brgemm_cache_info_t> cache_;
};

// TODO(haixin): use syscall to determine page size?
static constexpr size_t SCRATCH_SIZE = 2 * 4096;
// TODO(haixin): need to use custom thread management for scratch in the future?
static thread_local char scratch[SCRATCH_SIZE] = {0};

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
  char *palette_buffer = nullptr;
  if (desc.is_tmm) {
    palette_buffer = new char[PALETTE_SIZE];
    dnnl::impl::status_t status = brgemm_init_tiles(desc, palette_buffer);
    assert(status == dnnl::impl::status::success &&
           "Failed to initialize palette for BRGEMM");
  }

  write_lock_guard_t g(g_brgemm_lock);
  g_brgemm_desc_list.push_back(desc);
  g_brgemm_kernel_list.push_back(kernel);
  g_brgemm_palette.emplace_back(palette_buffer);
  return g_brgemm_desc_list.size() - 1;
}

void dnnl_brgemm_tileconfig(int64_t kernel_idx) {
  assert(kernel_idx >= 0 && "Invalid kernel handler");
  auto &cache_manager = brgemm_cache_manager::getInstance();
  brgemm_cache_info_t info;
  if (!cache_manager.tryGet(kernel_idx, info)) {
    read_lock_guard_t g(g_brgemm_lock);
    assert(kernel_idx < (int64_t)g_brgemm_desc_list.size() &&
           "Invalid kernel handler");
    info = {&g_brgemm_desc_list[kernel_idx], g_brgemm_kernel_list[kernel_idx],
            g_brgemm_palette[kernel_idx].get()};
    cache_manager.insertOrUpdate(kernel_idx, info);
  }
  brgemm_desc_t *desc = info.desc.get();
  std::shared_ptr<char> palette_buffer = info.palette;

  if (!desc->is_tmm) {
    return;
  }

  assert(palette_buffer && "Invalid palette for BRGEMM kernel");
  amx_tile_configure(palette_buffer.get());
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
  auto &cache_manager = brgemm_cache_manager::getInstance();
  brgemm_cache_info_t info;
  if (!cache_manager.tryGet(kernel_idx, info)) {
    read_lock_guard_t g(g_brgemm_lock);
    assert(kernel_idx >= 0 && kernel_idx < (int64_t)g_brgemm_desc_list.size() &&
           "Invalid kernel handler");
    info = {&g_brgemm_desc_list[kernel_idx], g_brgemm_kernel_list[kernel_idx],
            g_brgemm_palette[kernel_idx].get()};
    cache_manager.insertOrUpdate(kernel_idx, info);
  }

  assert(info.kernel && "Invalid brgemm kernel pointer");

  size_t A_offset_in_bytes =
      dnnl::impl::types::data_type_size(info.desc->dt_a) * A_offset;
  size_t B_offset_in_bytes =
      dnnl::impl::types::data_type_size(info.desc->dt_b) * B_offset;
  size_t C_offset_in_bytes =
      dnnl::impl::types::data_type_size(info.desc->dt_c) * C_offset;

  char *A_arith = (char *)A;
  char *B_arith = (char *)B;
  char *C_arith = (char *)C;
  brgemm_kernel_execute(info.kernel.get(), num,
                        (void *)(A_arith + A_offset_in_bytes),
                        (void *)(B_arith + B_offset_in_bytes), nullptr,
                        (void *)(C_arith + C_offset_in_bytes), (void *)scratch);
}
}

#endif
