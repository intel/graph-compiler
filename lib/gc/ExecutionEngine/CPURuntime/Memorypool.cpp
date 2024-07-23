//===-- Memorypool.cpp - memorypool ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/MemoryPool/MemoryPool.h"
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
static constexpr size_t divideAndCeil(size_t x, size_t y) {
  return (x + y - 1) / y;
}
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

memoryChunk *memoryChunk::init(intptr_t pdata, size_t sz) {
  memoryChunk *ths =
      reinterpret_cast<memoryChunk *>(pdata - sizeof(memoryChunk));
  ths->canary_ = magicCheckNum_;
  ths->size_ = sz;
  return ths;
}

intptr_t memoryBlock::callAllocPtr() {
  intptr_t startAddr =
      reinterpret_cast<intptr_t>(this) + allocated_ + sizeof(memoryChunk);
  return divideAndCeil(startAddr, defaultAlignment) * defaultAlignment;
}

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

memoryBlock *memoryBlock::make(size_t sz, memoryBlock *prev,
                               memoryBlock *next) {
  auto ret = allocByMmap(sz);
  if (!ret) {
    std::cerr << "Out of Memory." << std::endl;
  }
  memoryBlock *blk = reinterpret_cast<memoryBlock *>(ret);
  blk->size_ = sz;
  blk->allocated_ = sizeof(memoryBlock);
  static_assert(sizeof(memoryBlock) == offsetof(memoryBlock, buffer_),
                "sizeof(memoryBlock) == offsetof(memoryBlock, buffer_)");
  blk->prev_ = prev;
  blk->next_ = next;
  return blk;
}

void deallocByMmap(void *b) {
#ifdef _MSC_VER
  auto ret = VirtualFree(b, 0, MEM_RELEASE);
  SC_UNUSED(ret);
  assert(ret);
#else
  munmap(b, reinterpret_cast<memoryBlock *>(b)->size_);
#endif
}

static void freeMemoryBlockList(memoryBlock *b) {
  while (b) {
    memoryBlock *next = b->next_;
    deallocByMmap(b);
    b = next;
  }
}

size_t filoMemoryPool::getBlockSize(size_t sz) const {
  // calculate the aligned size of management blocks in the header
  constexpr size_t header_size =
      divideAndCeil(sizeof(memoryBlock) + sizeof(memoryChunk),
                    defaultAlignment) *
      defaultAlignment;
  // the allocated size should include the aligned header size
  sz = sz + header_size;
  if (sz > blockSize_) {
    return divideAndCeil(sz, getOsPageSize()) * getOsPageSize();
  } else {
    return blockSize_;
  }
}

void *filoMemoryPool::alloc(size_t sz) {
  if (unlikely(!buffers_)) {
    buffers_ = memoryBlock::make(getBlockSize(sz), nullptr, nullptr);
    current_ = buffers_;
  }
  do {
    intptr_t newptr = current_->callAllocPtr();
    size_t newallocated = newptr + sz - reinterpret_cast<intptr_t>(current_);
    if (likely(newallocated <= current_->size_)) {
      // if the current block is not full
      size_t alloc_size = newallocated - current_->allocated_;
      current_->allocated_ = newallocated;
      memoryChunk *chunk = memoryChunk::init(newptr, alloc_size);
      return reinterpret_cast<void *>(newptr);
    }
    // if the block is full, check the next block
    // if there is no next block left, allocate a new one
    if (!current_->next_) {
      current_->next_ = memoryBlock::make(getBlockSize(sz), current_, nullptr);
    }
    current_ = current_->next_;
  } while (true);
}

void filoMemoryPool::dealloc(void *ptr) {
  auto intptr = reinterpret_cast<intptr_t>(ptr);
  auto intcur = reinterpret_cast<intptr_t>(current_);
  // Optional: check if the pointer is valid in the current block

  assert(intptr > intcur &&
         intptr - intcur < static_cast<ptrdiff_t>(current_->size_));
  auto chunk = reinterpret_cast<memoryChunk *>(intptr - sizeof(memoryChunk));
  // Optional: check if the stack is ok
  assert(chunk->canary_ == memoryChunk::magicCheckNum_ &&
         "Corrupt stack detected");
  assert(current_->allocated_ > chunk->size_);
  current_->allocated_ -= chunk->size_;

  // skip the empty blocks
  while (unlikely(current_->allocated_ == sizeof(memoryBlock))) {
    if (current_->prev_) {
      current_ = current_->prev_;
    } else {
      break;
    }
  }
}

void filoMemoryPool::release() {
  freeMemoryBlockList(buffers_);
  buffers_ = nullptr;
  current_ = nullptr;
}

void filoMemoryPool::clear() {
  for (auto cur = current_; cur; cur = cur->prev_) {
    cur->allocated_ = sizeof(memoryBlock);
  }
  current_ = buffers_;
}

filoMemoryPool::~filoMemoryPool() { release(); }

static thread_local filoMemoryPool mainMemoryPool_{memory_pool::mainChunkSize};
// if the current thread is a worker thread, use this pool
static thread_local filoMemoryPool threadMemoryPool_{threadlocalChunkSize};

} // namespace memory_pool
} // namespace gc
} // namespace mlir

extern "C" void *gcAlignedMalloc(size_t sz) noexcept {
  if (sz == 0) {
    return nullptr;
  }
  return mlir::gc::memory_pool::mainMemoryPool_.alloc(sz);
}

extern "C" void gcAlignedFree(void *p) noexcept {
  mlir::gc::memory_pool::mainMemoryPool_.dealloc(p);
}

extern "C" void *gcThreadAlignedMalloc(size_t sz) noexcept {
  return mlir::gc::memory_pool::threadMemoryPool_.alloc(sz);
}

extern "C" void gcThreadAlignedFree(void *p) noexcept {
  mlir::gc::memory_pool::threadMemoryPool_.dealloc(p);
}
