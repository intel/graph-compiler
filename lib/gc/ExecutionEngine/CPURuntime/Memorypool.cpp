//===-- Memorypool.cpp - memorypool ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/MemoryPool/MemoryPool.h"
#include "gc/ExecutionEngine/MemoryPool/ThreadLocals.h"
#include <cassert>
#include <iostream>
#include <memory.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
#define __builtin_expect(EXP_, C) (EXP_)
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace mlir {
namespace gc {

namespace memory_pool {
static constexpr size_t divide_and_ceil(size_t x, size_t y) {
  return (x + y - 1) / y;
}
static constexpr size_t default_alignment = 64;

size_t get_os_page_size() {
#ifdef _WIN32
  // fix-me: (win32) impl
  return 4096;
#else
  static size_t v = getpagesize();
  return v;
#endif
}

memory_chunk_t *memory_chunk_t::init(intptr_t pdata, size_t sz) {
  memory_chunk_t *ths =
      reinterpret_cast<memory_chunk_t *>(pdata - sizeof(memory_chunk_t));
  ths->canary_ = magic_check_num_;
  ths->size_ = sz;
  return ths;
}

intptr_t memory_block_t::calc_alloc_ptr() {
  intptr_t start_addr =
      reinterpret_cast<intptr_t>(this) + allocated_ + sizeof(memory_chunk_t);
  return divide_and_ceil(start_addr, default_alignment) * default_alignment;
}

void *alloc_by_mmap(size_t sz) {
#ifdef _MSC_VER
  auto ret =
      VirtualAlloc(nullptr, sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
  auto ret = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#endif
  assert(ret);
  return ret;
}

memory_block_t *memory_block_t::make(size_t sz, memory_block_t *prev,
                                     memory_block_t *next) {
  auto ret = alloc_by_mmap(sz);
  if (!ret) {
    std::cerr << "Out of Memory." << std::endl;
  }
  memory_block_t *blk = reinterpret_cast<memory_block_t *>(ret);
  blk->size_ = sz;
  blk->allocated_ = sizeof(memory_block_t);
  static_assert(sizeof(memory_block_t) == offsetof(memory_block_t, buffer_),
                "sizeof(memory_block_t) == offsetof(memory_block_t, buffer_)");
  blk->prev_ = prev;
  blk->next_ = next;
  return blk;
}

void dealloc_by_mmap(void *b) {
#ifdef _MSC_VER
  auto ret = VirtualFree(b, 0, MEM_RELEASE);
  SC_UNUSED(ret);
  assert(ret);
#else
  munmap(b, reinterpret_cast<memory_block_t *>(b)->size_);
#endif
}

static void free_memory_block_list(memory_block_t *b) {
  while (b) {
    memory_block_t *next = b->next_;
    dealloc_by_mmap(b);
    b = next;
  }
}

size_t filo_memory_pool_t::get_block_size(size_t sz) const {
  // calculate the aligned size of management blocks in the header
  constexpr size_t header_size =
      divide_and_ceil(sizeof(memory_block_t) + sizeof(memory_chunk_t),
                      default_alignment) *
      default_alignment;
  // the allocated size should include the aligned header size
  sz = sz + header_size;
  if (sz > block_size_) {
    return divide_and_ceil(sz, get_os_page_size()) * get_os_page_size();
  } else {
    return block_size_;
  }
}

void *filo_memory_pool_t::alloc(size_t sz) {
  if (unlikely(!buffers_)) {
    buffers_ = memory_block_t::make(get_block_size(sz), nullptr, nullptr);
    current_ = buffers_;
  }
  do {
    intptr_t newptr = current_->calc_alloc_ptr();
    size_t newallocated = newptr + sz - reinterpret_cast<intptr_t>(current_);
    if (likely(newallocated <= current_->size_)) {
      // if the current block is not full
      size_t alloc_size = newallocated - current_->allocated_;
      current_->allocated_ = newallocated;
      memory_chunk_t *chunk = memory_chunk_t::init(newptr, alloc_size);
      return reinterpret_cast<void *>(newptr);
    }
    // if the block is full, check the next block
    // if there is no next block left, allocate a new one
    if (!current_->next_) {
      current_->next_ =
          memory_block_t::make(get_block_size(sz), current_, nullptr);
    }
    current_ = current_->next_;
  } while (true);
}

void filo_memory_pool_t::dealloc(void *ptr) {
  auto intptr = reinterpret_cast<intptr_t>(ptr);
  auto intcur = reinterpret_cast<intptr_t>(current_);
  // Optional: check if the pointer is valid in the current block

  assert(intptr > intcur &&
         intptr - intcur < static_cast<ptrdiff_t>(current_->size_));
  auto chunk =
      reinterpret_cast<memory_chunk_t *>(intptr - sizeof(memory_chunk_t));
  // Optional: check if the stack is ok
  assert(chunk->canary_ == memory_chunk_t::magic_check_num_ &&
         "Corrupt stack detected");
  assert(current_->allocated_ > chunk->size_);
  current_->allocated_ -= chunk->size_;

  // skip the empty blocks
  while (unlikely(current_->allocated_ == sizeof(memory_block_t))) {
    if (current_->prev_) {
      current_ = current_->prev_;
    } else {
      break;
    }
  }
}

void filo_memory_pool_t::release() {
  free_memory_block_list(buffers_);
  buffers_ = nullptr;
  current_ = nullptr;
}

void filo_memory_pool_t::clear() {
  for (auto cur = current_; cur; cur = cur->prev_) {
    cur->allocated_ = sizeof(memory_block_t);
  }
  current_ = buffers_;
}

filo_memory_pool_t::~filo_memory_pool_t() { release(); }

} // namespace memory_pool
} // namespace gc
} // namespace mlir

extern "C" void *gcAlignedMalloc(size_t sz) noexcept {
  if (sz == 0) {
    return nullptr;
  }
  return mlir::gc::thread_local_buffer_t::tls_buffer().main_memory_pool_.alloc(
      sz);
}

extern "C" void gcAlignedFree(void *p) noexcept {
  mlir::gc::thread_local_buffer_t::tls_buffer().main_memory_pool_.dealloc(p);
}

extern "C" void *gcThreadAlignedMalloc(size_t sz) noexcept {
  return mlir::gc::thread_local_buffer_t::tls_buffer()
      .thread_memory_pool_.alloc(sz);
}

extern "C" void gcThreadAlignedFree(void *p) noexcept {
  mlir::gc::thread_local_buffer_t::tls_buffer().thread_memory_pool_.dealloc(p);
}