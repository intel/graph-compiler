//===-- ConstantCache.hpp - Constant cache interfaces -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GC_EXECUTIONENGINE_CPURUNTIME_CONSTANT_CACHE_H
#define GC_EXECUTIONENGINE_CPURUNTIME_CONSTANT_CACHE_H
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <atomic>
#include <memory>
#include <stdint.h>

namespace mlir {
namespace gc {
/**
 * The helper class to manage ref count manually for an object allocated with
 * shared ptr. It holds an additional shared ptr reference to the object and
 * contains an additional self-managed refcount. The refcount will be set to 1
 * when the object is initialized (see init()). When the refcount counts down to
 * 0, the additional shared ptr is reset.
 */
struct RefCountManaged {
  RefCountManaged() = default;
  RefCountManaged(const std::shared_ptr<void> &keep_alive) { init(keep_alive); }
  void init(const std::shared_ptr<void> &keep_alive) {
    keepAlive = keep_alive;
    refCount.store(1);
  }

  void ref() { ++refCount; }
  void deref() {
    auto newv = --refCount;
    if (newv == 0) {
      keepAlive = nullptr;
    }
  }

  // atomically check if refCount > 0. if so, ref() the object and return
  // true. Otherwise (if refCount==0), return false
  bool checkAliveAndRef() {
    auto oldv = refCount.load();
    for (;;) {
      if (oldv <= 0) {
        return false;
      }
      if (refCount.compare_exchange_strong(oldv, oldv + 1)) {
        return true;
      }
      // CAS failed, oldv has now the newest known value of refCount
    }
  }

  bool isAlive() const { return refCount > 0; }
  void *getPtrUnsafe() const { return keepAlive.get(); }

private:
  std::shared_ptr<void> keepAlive;
  std::atomic<int> refCount{0};
};

/**
 * The proxy for the constant cache of Graph API. It holds a shared ptr pointing
 * to the cache item in the cache manager (keep_alive) to extend the lifetime by
 * refcount, @see RefCountManaged. To access the memory buffer of the const
 * cache, use acauire/release functions. They will ref/deref the ConstCacheProxy
 * to make sure the cache is alive after calling acauire and before release. The
 * cache manager of Graph API may evict the cache item by dereferenceing this
 * RefCountManaged object. {acquire,release} functions will find out that the
 * cache has been invalidated. Usually we expect JIT modules to hold shared ptr
 * to ConstCacheProxy via  CachedGraphTensor. If is_lazy_ == true, the cache
 * item's lifetime will be managed by the cache manager of Graph API and it is
 * filled with data after the first execution of the computation. Otherwise, the
 * cache item is always alive as long as the jit_module of the kernel is alive.
 */
struct ConstCacheProxy : RefCountManaged {
  ConstCacheProxy(const std::shared_ptr<void> &keep_alive, void *buffer,
                  size_t size, bool is_lazy)
      : RefCountManaged(keep_alive), size_(size), is_lazy_(is_lazy),
        buffer_(buffer) {}
  ~ConstCacheProxy();

  // get the buffer and increment the refcount. If the buffer is evicted,
  // returns null
  void *acquire(int32_t *inited) {
    if (checkAliveAndRef()) {
      *inited = *inited && initialized_;
      return buffer_;
    }
    return nullptr;
  }
  // decrement the refcount
  bool release() {
    if (isAlive()) {
      deref();
      initialized_ = 1;
      return true;
    }
    return false;
  }

  // return the buffer. Do not directly use the buffer because it may be already
  // release! To access the buffer, always acquire() before using it.
  void *getBufferUnsafe() const { return buffer_; }

  size_t size_;
  // if the buffer is lazy-initialized. If false, it should be filled before
  // computation
  bool is_lazy_;

private:
  // raw pointer to the buffer
  void *buffer_;
  // if the buffer has been initialized. calling release() will set this to 1
  int32_t initialized_ = 0;
};

struct CachedGraphTensor {
  std::shared_ptr<ConstCacheProxy> base;
  size_t offset;
  CachedGraphTensor(const std::shared_ptr<ConstCacheProxy> &base,
                    size_t offset);
  friend class JitModule;

private:
  StridedMemRefType<char, 8> ref;
};

std::shared_ptr<CachedGraphTensor> queryCacheTensor(uint64_t key);
bool regCachedTensor(uint64_t key, const std::shared_ptr<ConstCacheProxy> &base,
                     size_t offset);

} // namespace gc
} // namespace mlir

#endif