//===-- Parallel.cpp - parallel ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <stdarg.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define WEAK_SYMBOL __attribute__((weak))

namespace {
struct barrier_t {
  alignas(64) std::atomic<int32_t> pending_;
  std::atomic<int32_t> rounds_;
  uint64_t total_;
  // pad barrier to size of cacheline to avoid false sharing
  char padding_[64 - 4 * sizeof(int32_t)];
};

using barrier_idle_func = uint64_t (*)(std::atomic<int32_t> *remaining,
                                       int32_t expected_remain, int32_t tid,
                                       void *args);
} // namespace

extern "C" {
int gc_runtime_keep_alive = 0;
void gc_arrive_at_barrier(barrier_t *b, barrier_idle_func idle_func,
                          void *idle_args) {
  auto cur_round = b->rounds_.load(std::memory_order_acquire);
  auto cnt = --b->pending_;
  assert(cnt >= 0);
  if (cnt == 0) {
    b->pending_.store(b->total_);
    b->rounds_.store(cur_round + 1);
  } else {
    if (idle_func) {
      if (cur_round != b->rounds_.load()) {
        return;
      }
      idle_func(&b->rounds_, cur_round + 1, -1, idle_args);
    }
    while (cur_round == b->rounds_.load()) {
      _mm_pause();
    }
  }
}

static_assert(sizeof(barrier_t) == 64, "size of barrier_t should be 64-byte");

void gc_init_barrier(barrier_t *b, int num_barriers, uint64_t thread_count) {
  for (int i = 0; i < num_barriers; i++) {
    b[i].total_ = thread_count;
    b[i].pending_.store(thread_count);
    b[i].rounds_.store(0);
  }
}

#if GC_NEEDS_OMP_WRAPPER
void WEAK_SYMBOL __kmpc_barrier(void *loc, int32_t global_tid) {
#pragma omp barrier
}

int WEAK_SYMBOL __kmpc_global_thread_num(void *loc) {
  return omp_get_thread_num();
}

void WEAK_SYMBOL __kmpc_for_static_init_8u(void *loc, int32_t gtid,
                                           int32_t schedtype,
                                           int32_t *plastiter, uint64_t *plower,
                                           uint64_t *pupper, int64_t *pstride,
                                           int64_t incr, int64_t chunk) {
  if (unlikely(schedtype != 34)) {
    std::abort();
  }
  const int32_t FALSE = 0;
  const int32_t TRUE = 1;
  using UT = uint64_t;
  //   using ST = int64_t;
  /*  this all has to be changed back to TID and such.. */
  uint32_t tid = gtid;
  uint32_t nth = omp_get_num_threads();
  UT trip_count;

  /* special handling for zero-trip loops */
  if (incr > 0 ? (*pupper < *plower) : (*plower < *pupper)) {
    if (plastiter != nullptr)
      *plastiter = FALSE;
    /* leave pupper and plower set to entire iteration space */
    *pstride = incr; /* value should never be used */
    return;
  }

  if (nth == 1) {
    if (plastiter != nullptr)
      *plastiter = TRUE;
    *pstride =
        (incr > 0) ? (*pupper - *plower + 1) : (-(*plower - *pupper + 1));
    return;
  }

  /* compute trip count */
  if (incr == 1) {
    trip_count = *pupper - *plower + 1;
  } else if (incr == -1) {
    trip_count = *plower - *pupper + 1;
  } else if (incr > 0) {
    // upper-lower can exceed the limit of signed type
    trip_count = (UT)(*pupper - *plower) / incr + 1;
  } else {
    trip_count = (UT)(*plower - *pupper) / (-incr) + 1;
  }
  if (trip_count < nth) {
    if (tid < trip_count) {
      *pupper = *plower = *plower + tid * incr;
    } else {
      // set bounds so non-active threads execute no iterations
      *plower = *pupper + (incr > 0 ? 1 : -1);
    }
    if (plastiter != nullptr)
      *plastiter = (tid == trip_count - 1);
  } else {
    UT small_chunk = trip_count / nth;
    UT extras = trip_count % nth;
    *plower += incr * (tid * small_chunk + (tid < extras ? tid : extras));
    *pupper = *plower + small_chunk * incr - (tid < extras ? 0 : incr);
    if (plastiter != nullptr)
      *plastiter = (tid == nth - 1);
  }
  *pstride = trip_count;
}

void WEAK_SYMBOL __kmpc_for_static_fini(void *ptr, int32_t v) {}

static thread_local int next_num_threads = 0;

/*!
@ingroup PARALLEL
The type for a microtask which gets passed to @ref __kmpc_fork_call().
The arguments to the outlined function are
@param global_tid the global thread identity of the thread executing the
function.
@param bound_tid  the local identity of the thread executing the function
@param ... pointers to shared variables accessed by the function.
*/
using kmpc_micro = void (*)(int32_t *global_tid, int32_t *bound_tid, ...);
void WEAK_SYMBOL __kmpc_fork_call(void *loc, int32_t argc, void *pfunc, ...) {
  if (unlikely(argc != 1 && argc != 0)) {
    std::abort();
  }
  va_list ap;
  va_start(ap, pfunc);
  void *c = va_arg(ap, void *);
  int32_t global_tid = 0;
  if (unlikely(next_num_threads)) {
#pragma omp parallel num_threads(next_num_threads)
    {
      kmpc_micro func = (kmpc_micro)(pfunc);
      func(&global_tid, nullptr, c);
    }
    next_num_threads = 0;
  } else {
#pragma omp parallel
    {
      kmpc_micro func = (kmpc_micro)(pfunc);
      func(&global_tid, nullptr, c);
    }
  }
  va_end(ap);
}

void WEAK_SYMBOL __kmpc_push_num_threads(void *loc, int32_t global_tid,
                                         int32_t num_threads) {
  next_num_threads = num_threads;
}
#endif
}
