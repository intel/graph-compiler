//===-- ThreadLocals.h - The MLIR compiler runtime allocator helper
//-----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_MEMORYPOOL_THREADLOCALS_H
#define GC_EXECUTIONENGINE_MEMORYPOOL_THREADLOCALS_H

#include "MemoryPool.h"
#include <assert.h>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
namespace mlir {
namespace gc {

struct thread_local_registry_t;

// a container for thread local resources. Users can call
// release_runtime_memory to manually release all thread local memory
// managed by this struct
struct thread_local_buffer_t {
  // the additional thread local data. Referenced via a pointer in
  // thread_local_buffer_t to reduce TLS size and improve performance
  struct additional_t {
    int linear_thread_id_ = 0;
    int instance_id_ = 0;
    bool is_main_thread_ = false;
    memory_pool::filo_memory_pool_t dyn_threadpool_mem_pool_{4096 * 16};
    // the pointer to keep registry alive
    std::shared_ptr<thread_local_registry_t> registry_;
    additional_t();
  };
  bool in_managed_thread_pool_ = false;

  // if the current thread is the "main" thread, use this pool
  memory_pool::filo_memory_pool_t main_memory_pool_{
      memory_pool::main_chunk_size};
  // if the current thread is a worker thread, use this pool
  memory_pool::filo_memory_pool_t thread_memory_pool_{
      memory_pool::threadlocal_chunk_size};

  std::unique_ptr<additional_t> additional_;

  ~thread_local_buffer_t();
  using list_type = std::list<thread_local_buffer_t *>;

  static inline thread_local_buffer_t &tls_buffer() {
    static thread_local thread_local_buffer_t tls_buffer_;
    return tls_buffer_;
  }

  // disable move and copy
  thread_local_buffer_t(const thread_local_buffer_t &) = delete;
  thread_local_buffer_t(thread_local_buffer_t &&) = delete;

  thread_local_buffer_t &operator=(const thread_local_buffer_t &) = delete;
  thread_local_buffer_t &operator=(thread_local_buffer_t &&) = delete;

private:
  friend struct thread_local_registry_t;
  // private ctor makes sure this struct can only be used in TLS
  thread_local_buffer_t();
  // the current position in thread_local_registry
  list_type::iterator cur_pos_;
};

// gets the Thread Local Storage associated with the stream. Note that we assume
// that a thread will be attached to one stream when the thread runs a kernel at
// the first time and it will not switch between streams at the run time. We
// also have the same assumption on the "main" thread which invokes the main
// entry of the kernel
inline thread_local_buffer_t &get_tls() {
  auto &ret = thread_local_buffer_t::tls_buffer();
  return ret;
}

const std::shared_ptr<thread_local_registry_t> &get_thread_locals_registry();

// the registry of all TLS resources.
struct thread_local_registry_t {
  std::mutex lock_;
  std::list<thread_local_buffer_t *> tls_buffers_;
  std::vector<std::unique_ptr<thread_local_buffer_t::additional_t>>
      dead_threads_;
  void release();
  void for_each_tls_additional(
      const std::function<void(thread_local_buffer_t::additional_t *)> &f);

  thread_local_registry_t();
  ~thread_local_registry_t();
};

} // namespace gc
} // namespace mlir

#endif
