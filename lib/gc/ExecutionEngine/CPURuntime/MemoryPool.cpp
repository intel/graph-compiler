//===-- MemoryPool.cpp - Runtime allocator memory pool ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
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

#include "gc_utils.h"

#ifdef _MSC_VER
#define __builtin_expect(EXP_, C) (EXP_)
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
namespace {
// 4MB
constexpr size_t threadlocalChunkSize = 4 * 1024 * 1024;
// 16MB
constexpr size_t mainChunkSize = 16 * 1024 * 1024;

static constexpr size_t defaultAlignment = 64;

size_t getOsPageSize() {
#ifdef _WIN32
  // fix-me: (win32) impl
  return 4096;
#else
  static size_t v = getpagesize();
  return v;
#endif
}

static constexpr size_t alignTo(size_t x, size_t y) {
  return (x + y - 1) / y * y;
}

// The chunk of memory that is allocated to the user
struct MemoryChunk {
  static constexpr uint64_t magicCheckNum = 0xc0ffeebeef0102ff;
  // `canary` should be set as a magic value to check the existence of
  // overflow
  uint64_t canary;
  // the size of the MemoryChunk allocated, incluing this `MemoryChunk`
  size_t size;
  // the memory for the user
  char buffer[0];
  // initalizes the memory chunk, given the address of data
  static MemoryChunk *init(intptr_t pdata, size_t sz);
};

MemoryChunk *MemoryChunk::init(intptr_t pdata, size_t sz) {
  MemoryChunk *ths =
      reinterpret_cast<MemoryChunk *>(pdata - sizeof(MemoryChunk));
  ths->canary = magicCheckNum;
  ths->size = sz;
  return ths;
}

// the control block for pre-allocated memory block - created by page-wise
// allocation system calls (mmap). We can divide the memory block into memory
// chunks for user memory allocation in memory starting from `buffer`
struct MemoryBlock {
  // size of the memory block, starting from `this`
  size_t size;
  // size of allocated bytes, including this struct
  size_t allocated;
  MemoryBlock *prev;
  MemoryBlock *next;
  // here starts the allocatable memory
  char buffer[0];

  /**
   * Calculates the next pointer to allocate with alignment = 512-bits (64
   * bytes). The  (returned pointer - sizeof(MemoryChunk)) should be the
   * address of MemoryChunk
   * */
  intptr_t callAllocPtr();

  static MemoryBlock *make(size_t sz, MemoryBlock *prev, MemoryBlock *next);
};

void *allocByMmap(size_t sz) {
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

void deallocByMmap(void *b) {
#ifdef _MSC_VER
  auto ret = VirtualFree(b, 0, MEM_RELEASE);
  (void)(ret);
  assert(ret);
#else
  munmap(b, reinterpret_cast<MemoryBlock *>(b)->size);
#endif
}

intptr_t MemoryBlock::callAllocPtr() {
  intptr_t startAddr =
      reinterpret_cast<intptr_t>(this) + allocated + sizeof(MemoryChunk);
  return alignTo(startAddr, defaultAlignment);
}

MemoryBlock *MemoryBlock::make(size_t sz, MemoryBlock *prev,
                               MemoryBlock *next) {
  auto ret = allocByMmap(sz);
  if (!ret) {
    std::cerr << "Out of Memory." << std::endl;
  }
  MemoryBlock *blk = reinterpret_cast<MemoryBlock *>(ret);
  blk->size = sz;
  blk->allocated = sizeof(MemoryBlock);
  static_assert(sizeof(MemoryBlock) == offsetof(MemoryBlock, buffer),
                "sizeof(MemoryBlock) == offsetof(MemoryBlock, buffer)");
  blk->prev = prev;
  blk->next = next;
  return blk;
}

// The FILO memory pool. The memory allocation and deallocation should be in
// first-in-last-out fashion
struct FILOMemoryPool {
  size_t blockSize;
  // the linked list of all allocated memory blocks
  MemoryBlock *buffers = nullptr;
  MemoryBlock *current = nullptr;
  size_t getBlockSize(size_t sz) const;
  void *alloc(size_t sz);
  void dealloc(void *ptr);
  FILOMemoryPool(size_t bs) : blockSize(bs) {}
  ~FILOMemoryPool();
  // release the memory to os/underlying memory allocator
  void release();
  // reset the memory pool, but keep the allocated memory in the pool
  void clear();
};

static void freeMemoryBlockList(MemoryBlock *b) {
  while (b) {
    MemoryBlock *next = b->next;
    deallocByMmap(b);
    b = next;
  }
}

size_t FILOMemoryPool::getBlockSize(size_t sz) const {
  // calculate the aligned size of management blocks in the header
  constexpr size_t header_size =
      alignTo(sizeof(MemoryBlock) + sizeof(MemoryChunk), defaultAlignment);
  // the allocated size should include the aligned header size
  sz = sz + header_size;
  if (sz > blockSize) {
    return alignTo(sz, getOsPageSize());
  } else {
    return blockSize;
  }
}

void *FILOMemoryPool::alloc(size_t sz) {
  if (unlikely(!buffers)) {
    buffers = MemoryBlock::make(getBlockSize(sz), nullptr, nullptr);
    current = buffers;
  }
  do {
    intptr_t newptr = current->callAllocPtr();
    size_t newallocated = newptr + sz - reinterpret_cast<intptr_t>(current);
    if (likely(newallocated <= current->size)) {
      // if the current block is not full
      size_t alloc_size = newallocated - current->allocated;
      current->allocated = newallocated;
      MemoryChunk::init(newptr, alloc_size);
      return reinterpret_cast<void *>(newptr);
    }
    // if the block is full, check the next block
    // if there is no next block left, allocate a new one
    if (!current->next) {
      current->next = MemoryBlock::make(getBlockSize(sz), current, nullptr);
    }
    current = current->next;
  } while (true);
}

void FILOMemoryPool::dealloc(void *ptr) {
  auto intptr = reinterpret_cast<intptr_t>(ptr);
  auto intcur = reinterpret_cast<intptr_t>(current);
  // Optional: check if the pointer is valid in the current block

  assert(intptr > intcur &&
         intptr - intcur < static_cast<ptrdiff_t>(current->size));
  auto chunk = reinterpret_cast<MemoryChunk *>(intptr - sizeof(MemoryChunk));
  // Optional: check if the stack is ok
  assert(chunk->canary == MemoryChunk::magicCheckNum &&
         "Corrupt stack detected");
  assert(current->allocated > chunk->size);
  current->allocated -= chunk->size;

  // skip the empty blocks
  while (unlikely(current->allocated == sizeof(MemoryBlock))) {
    if (current->prev) {
      current = current->prev;
    } else {
      break;
    }
  }
}

void FILOMemoryPool::release() {
  freeMemoryBlockList(buffers);
  buffers = nullptr;
  current = nullptr;
}

FILOMemoryPool::~FILOMemoryPool() { release(); }

} // namespace

static thread_local FILOMemoryPool mainMemoryPool_{mainChunkSize};
// if the current thread is a worker thread, use this pool
static thread_local FILOMemoryPool threadMemoryPool_{threadlocalChunkSize};

extern "C" GC_DLL_EXPORT void *gcAlignedMalloc(size_t sz) noexcept {
  return mainMemoryPool_.alloc(sz);
}

extern "C" GC_DLL_EXPORT void gcAlignedFree(void *p) noexcept {
  mainMemoryPool_.dealloc(p);
}

extern "C" GC_DLL_EXPORT void *gcThreadAlignedMalloc(size_t sz) noexcept {
  return threadMemoryPool_.alloc(sz);
}

extern "C" GC_DLL_EXPORT void gcThreadAlignedFree(void *p) noexcept {
  threadMemoryPool_.dealloc(p);
}
