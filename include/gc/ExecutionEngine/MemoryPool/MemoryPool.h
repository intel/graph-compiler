//===-- MemoryPool.h - Runtime allocator memory pool ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_MEMORYPOOL_MEMORYPOOL_H
#define GC_EXECUTIONENGINE_MEMORYPOOL_MEMORYPOOL_H

#include <memory>
#include <stddef.h>
#include <stdint.h>

namespace mlir {
namespace gc {

namespace memory_pool {

// 4MB
constexpr size_t threadlocalChunkSize = 4 * 1024 * 1024;
// 16MB
constexpr size_t mainChunkSize = 16 * 1024 * 1024;

// The chunk of memory that is allocated to the user
struct memoryChunk {
  static constexpr uint64_t magicCheckNum_ = 0xc0ffeebeef0102ff;
  // `canary` should be set as a magic value to check the existence of
  // overflow
  uint64_t canary_;
  // the size of the memoryChunk allocated, incluing this `memoryChunk`
  size_t size_;
  // the memory for the user
  char buffer_[0];
  // initalizes the memory chunk, given the address of data
  static memoryChunk *init(intptr_t pdata, size_t sz);
};

// the control block for pre-allocated memory block - created by page-wise
// allocation system calls (mmap). We can divide the memory block into memory
// chunks for user memory allocation in memory starting from `buffer_`
struct memoryBlock {
  // size of the memory block, starting from `this`
  size_t size_;
  // size of allocated bytes, including this struct
  size_t allocated_;
  memoryBlock *prev_;
  memoryBlock *next_;
  // here starts the allocatable memory
  char buffer_[0];

  /**
   * Calculates the next pointer to allocate with alignment = 512-bits (64
   * bytes). The  (returned pointer - sizeof(memoryChunk)) should be the
   * address of memoryChunk
   * */
  intptr_t callAllocPtr();

  static memoryBlock *make(size_t sz, memoryBlock *prev, memoryBlock *next);
};

// The FILO memory pool. The memory allocation and deallocation should be in
// first-in-last-out fashion
struct filoMemoryPool {
  size_t blockSize_;
  // the linked list of all allocated memory blocks
  memoryBlock *buffers_ = nullptr;
  memoryBlock *current_ = nullptr;
  size_t getBlockSize(size_t sz) const;
  void *alloc(size_t sz);
  void dealloc(void *ptr);
  filoMemoryPool(size_t blockSize) : blockSize_(blockSize) {}
  ~filoMemoryPool();
  // release the memory to os/underlying memory allocator
  void release();
  // reset the memory pool, but keep the allocated memory in the pool
  void clear();
};
void deallocByMmap(void *b);
void *allocByMmap(size_t sz);
} // namespace memory_pool

} // namespace gc
} // namespace mlir

extern "C" void *gcAlignedMalloc(size_t sz) noexcept;

extern "C" void gcAlignedFree(void *p) noexcept;

extern "C" void *gcThreadAlignedMalloc(size_t sz) noexcept;

extern "C" void gcThreadAlignedFree(void *p) noexcept;

#endif
