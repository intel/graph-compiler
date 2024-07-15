//===-- Memorypool.cpp - memorypool ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

namespace {
// 4MB
constexpr size_t threadlocal_chunk_size = 4 * 1024 * 1024;
// 16MB
constexpr size_t main_chunk_size = 16 * 1024 * 1024;

static constexpr size_t default_alignment = 64;

static constexpr size_t divide_and_ceil(size_t x, size_t y) {
  return (x + y - 1) / y;
}

size_t get_os_page_size() {
#ifdef _WIN32
  // fix-me: (win32) impl
  return 4096;
#else
  static size_t v = getpagesize();
  return v;
#endif
}

// The chunk of memory that is allocated to the user
struct memory_chunk_t {
  static constexpr uint64_t magic_check_num_ = 0xc0ffeebeef0102ff;
  // `canary` should be set as a magic value to check the existence of
  // overflow
  uint64_t canary_;
  // the size of the memory_chunk_t allocated, incluing this `memory_chunk_t`
  size_t size_;
  // the memory for the user
  char buffer_[0];
  // initalizes the memory chunk, given the address of data
  static memory_chunk_t *init(intptr_t pdata, size_t sz);
};

memory_chunk_t *memory_chunk_t::init(intptr_t pdata, size_t sz) {
  memory_chunk_t *ths =
      reinterpret_cast<memory_chunk_t *>(pdata - sizeof(memory_chunk_t));
  ths->canary_ = magic_check_num_;
  ths->size_ = sz;
  return ths;
}

// the control block for pre-allocated memory block - created by page-wise
// allocation system calls (mmap). We can divide the memory block into memory
// chunks for user memory allocation in memory starting from `buffer_`
struct memory_block_t {
  // size of the memory block, starting from `this`
  size_t size_;
  // size of allocated bytes, including this struct
  size_t allocated_;
  memory_block_t *prev_;
  memory_block_t *next_;
  // here starts the allocatable memory
  char buffer_[0];

  /**
   * Calculates the next pointer to allocate with alignment = 512-bits (64
   * bytes). The  (returned pointer - sizeof(memory_chunk_t)) should be the
   * address of memory_chunk_t
   * */
  intptr_t calc_alloc_ptr();

  static memory_block_t *make(size_t sz, memory_block_t *prev,
                              memory_block_t *next);
};

void dealloc_by_mmap(void *b) {
#ifdef _MSC_VER
  auto ret = VirtualFree(b, 0, MEM_RELEASE);
  SC_UNUSED(ret);
  assert(ret);
#else
  munmap(b, reinterpret_cast<memory_block_t *>(b)->size_);
#endif
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

intptr_t memory_block_t::calc_alloc_ptr() {
  intptr_t start_addr =
      reinterpret_cast<intptr_t>(this) + allocated_ + sizeof(memory_chunk_t);
  return divide_and_ceil(start_addr, default_alignment) * default_alignment;
}

// The FILO memory pool. The memory allocation and deallocation should be in
// first-in-last-out fashion
struct filo_memory_pool_t {
  size_t block_size_;
  // the linked list of all allocated memory blocks
  memory_block_t *buffers_ = nullptr;
  memory_block_t *current_ = nullptr;
  size_t get_block_size(size_t sz) const;
  void *alloc(size_t sz);
  void dealloc(void *ptr);
  filo_memory_pool_t(size_t block_size) : block_size_(block_size) {}
  ~filo_memory_pool_t();
  // release the memory to os/underlying memory allocator
  void release();
  // reset the memory pool, but keep the allocated memory in the pool
  void clear();
};

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

} // namespace

extern "C" void *gcAlignedMalloc(size_t sz) noexcept {
  if (sz == 0) {
    return nullptr;
  }
  filo_memory_pool_t main_memory_pool_{main_chunk_size};
  return main_memory_pool_.alloc(sz);
}

extern "C" void gcAlignedFree(void *p) noexcept {
  filo_memory_pool_t main_memory_pool_{main_chunk_size};
  main_memory_pool_.dealloc(p);
}

extern "C" void *gcThreadAlignedMalloc(size_t sz) noexcept {
  filo_memory_pool_t thread_memory_pool_{threadlocal_chunk_size};
  return thread_memory_pool_.alloc(sz);
}

extern "C" void gcThreadAlignedFree(void *p) noexcept {
  filo_memory_pool_t thread_memory_pool_{threadlocal_chunk_size};
  thread_memory_pool_.dealloc(p);
}