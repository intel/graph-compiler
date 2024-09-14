//===-- ConstantCache.h - Constant cache interfaces -------------*- C++ -*-===//
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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdint.h>
#include <unordered_map>
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
  RefCountManaged(const std::shared_ptr<void> &vkeepAlive) { init(vkeepAlive); }
  void init(const std::shared_ptr<void> &vkeepAlive) {
    keepAlive = vkeepAlive;
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
 * to the cache item in the cache manager (keepAlive) to extend the lifetime by
 * refcount, @see RefCountManaged. To access the memory buffer of the const
 * cache, use acauire/release functions. They will ref/deref the ConstCacheProxy
 * to make sure the cache is alive after calling acauire and before release. The
 * cache manager of Graph API may evict the cache item by dereferenceing this
 * RefCountManaged object. {acquire,release} functions will find out that the
 * cache has been invalidated. Usually we expect JIT modules to hold shared ptr
 * to ConstCacheProxy via  CachedGraphTensor. If isLazy == true, the cache
 * item's lifetime will be managed by the cache manager of Graph API and it is
 * filled with data after the first execution of the computation. Otherwise, the
 * cache item is always alive as long as the jit_module of the kernel is alive.
 */
struct ConstCacheProxy : RefCountManaged {
  ConstCacheProxy(const std::shared_ptr<void> &vkeepAlive, void *buffer,
                  size_t size, bool is_lazy)
      : RefCountManaged(vkeepAlive), size(size), isLazy(is_lazy),
        buffer(buffer) {}
  ~ConstCacheProxy() = default;

  // get the buffer and increment the refcount. If the buffer is evicted,
  // returns null
  void *acquire(int32_t *inited) {
    if (checkAliveAndRef()) {
      *inited = *inited && initialized;
      return buffer;
    }
    return nullptr;
  }
  // decrement the refcount
  bool release() {
    if (isAlive()) {
      deref();
      initialized = 1;
      return true;
    }
    return false;
  }

  // return the buffer. Do not directly use the buffer because it may be already
  // release! To access the buffer, always acquire() before using it.
  void *getBufferUnsafe() const { return buffer; }

  size_t size;
  // if the buffer is lazy-initialized. If false, it should be filled before
  // computation
  bool isLazy;

private:
  // raw pointer to the buffer
  void *buffer;
  // if the buffer has been initialized. calling release() will set this to 1
  int32_t initialized = 0;
};

struct CachedGraphTensor {
  // Multiple tensors can reside in one common ConstCacheProxy `base`, with
  // different offsets.
  std::shared_ptr<ConstCacheProxy> base;
  size_t offset;
  CachedGraphTensor(const std::shared_ptr<ConstCacheProxy> &base, size_t offset)
      : base{base}, offset{offset} {
    // todo: fill in real values
    ref.basePtr = (char *)base->getBufferUnsafe() + offset;
    ref.data = ref.basePtr;
    ref.offset = 0;
    memset(ref.sizes, 0, sizeof(ref.sizes));
    memset(ref.strides, 0, sizeof(ref.strides));
  }
  friend class JitModule;

private:
  StridedMemRefType<char, 8> ref;
};

inline std::shared_ptr<ConstCacheProxy> createConstCacheProxy(size_t size) {
  // simply allocate buffer and return
  std::shared_ptr<void> base = std::shared_ptr<void>{
      std::aligned_alloc(64, size), [](void *p) { std::free(p); }};
  return std::make_shared<ConstCacheProxy>(base, base.get(), size, true);
}

inline static size_t divideAndCeil(size_t x, size_t y) {
  return (x + y - 1) / y;
}

// Manager
struct ConstGraphTensorCacheManager {
  int64_t cachedTensorGlobalId = 0;

  std::unordered_map<int64_t, std::shared_ptr<CachedGraphTensor>> cache;

  // singleton
  static std::shared_ptr<ConstGraphTensorCacheManager> get() {
    static std::shared_ptr<ConstGraphTensorCacheManager> c =
        std::make_shared<ConstGraphTensorCacheManager>();
    return c;
  }

  std::shared_ptr<CachedGraphTensor> queryCacheTensor(int64_t key) {
    auto itr = cache.find(key);
    if (itr != cache.end()) {
      return itr->second;
    }
    return nullptr;
  }

  bool regCachedTensor(int64_t key,
                       const std::shared_ptr<ConstCacheProxy> &base,
                       size_t offset) {
    if (queryCacheTensor(key)) {
      return false;
    }

    cache[key] = std::make_shared<CachedGraphTensor>(base, offset);
    return true;
  }

  // alloc and set the buf_base_ and offset_ attributes of cache
  std::vector<int64_t> alloc(std::vector<size_t> buffersSize) {
    size_t totalSize = 0;
    for (size_t size : buffersSize) {
      totalSize += divideAndCeil(size, 64) * 64;
    }
    auto base = createConstCacheProxy(totalSize);
    std::vector<int64_t> globalIds(buffersSize.size());
    size_t offset = 0;
    for (size_t i = 0; i < buffersSize.size(); i++) {
      bool regRes = regCachedTensor(cachedTensorGlobalId, base, offset);
      assert(regRes && "Register constant tensor failed");
      globalIds[i] = cachedTensorGlobalId;
      ++cachedTensorGlobalId;
      offset += divideAndCeil(buffersSize[i], 64) * 64;
    }
    return globalIds;
  }
};

} // namespace gc
} // namespace mlir

#endif