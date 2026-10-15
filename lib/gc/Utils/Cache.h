//===-- Cache.h - Inmemory/file cache helpers ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <array>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/Threading.h>

#include "gc/Utils/Log.h"

namespace gc::cache {
struct Key {
  Key(llvm::ArrayRef<llvm::StringRef> inputs);
  llvm::SmallString<40> hex() const;

  std::array<uint8_t, 20> hash;
};
} // namespace gc::cache

namespace llvm {
template <> struct DenseMapInfo<gc::cache::Key> {
  static unsigned getHashValue(const gc::cache::Key &key) {
    return support::endian::read32le(key.hash.data());
  }
  static bool isEqual(const gc::cache::Key &lhs, const gc::cache::Key &rhs) {
    return lhs.hash == rhs.hash;
  }
};
} // namespace llvm

namespace gc::cache {
// In-memory cache of asynchronously produced entries. Pending entries are
// shared by all requesters, completed ones are held weakly.
template <typename E> struct AsyncCache {
  using FutureEntry = std::shared_future<std::shared_ptr<E>>;

  // 0 means all hardware threads. 1 means synchronous: tasks run in the
  // caller's thread.
  AsyncCache(unsigned concurrency = 0) {
    if (concurrency != 1) pool.emplace(llvm::hardware_concurrency(concurrency));
  }

  bool async() const { return pool.has_value(); }

  // Passed to the task, that must call put() once. If put() is never called,
  // waiters get nullptr. A nullptr entry is not cached, so the next get()
  // retries.
  struct Accessor {
    Accessor(AsyncCache &cache, const Key &key,
             std::promise<std::shared_ptr<E>> promise)
        : key(key), cache(cache), promise(std::move(promise)) {}

    ~Accessor() {
      if (done) return;
      {
        std::lock_guard<std::mutex> lock(cache.mux);
        cache.cache.erase(key);
      }
      // After erase, so that a woken waiter's get() retries.
      promise.set_value(nullptr);
    }

    void put(std::shared_ptr<E> entry) {
      std::weak_ptr<E> weak = entry;
      promise.set_value(std::move(entry));
      done = true;
      std::lock_guard<std::mutex> lock(cache.mux);
      cache.cache.find(key)->second = std::move(weak);
    }

    const Key &key;

  private:
    AsyncCache &cache;
    std::promise<std::shared_ptr<E>> promise;
    bool done = false;
  };

  // make() is invoked on a miss only and returns a task, that is run as
  // task(Acc& acc) on the pool or, if synchronous, in the caller's thread.
  template <typename Acc = Accessor, typename MakeTask>
  FutureEntry get(Key key, MakeTask &&make) {
    std::promise<std::shared_ptr<E>> promise;
    FutureEntry future;
    {
      std::lock_guard<std::mutex> lock(mux);
      if (auto now = std::chrono::steady_clock::now();
          now - lastCleanup >= std::chrono::minutes(1)) {
        lastCleanup = now;
        cache.remove_if([](auto &kv) {
          auto *weak = std::get_if<std::weak_ptr<E>>(&kv.second);
          return weak && weak->expired();
        });
      }
      auto [it, miss] = cache.try_emplace(key);
      if (!miss) {
        if (auto *f = std::get_if<FutureEntry>(&it->second)) {
          future = *f;
        } else if (auto e = std::get<std::weak_ptr<E>>(it->second).lock()) {
          promise.set_value(std::move(e));
          future = promise.get_future().share();
        }
        if (future.valid()) {
          gcLogD("Returning entry from in-memory cache: ", key.hex());
          return future;
        }
      }
      it->second = future = promise.get_future().share();
    }
    auto run = [this, key, task = make(),
                promise = std::move(promise)]() mutable {
      Acc acc(*this, key, std::move(promise));
      task(acc);
    };
    if (pool) pool->async(std::move(run));
    else run();
    return future;
  }

private:
  std::mutex mux;
  // Pending entries are futures, completed ones are weak pointers.
  llvm::DenseMap<Key, std::variant<FutureEntry, std::weak_ptr<E>>> cache;
  std::chrono::steady_clock::time_point lastCleanup;
  // Must be last: destroyed first, draining the tasks using the members above.
  std::optional<llvm::DefaultThreadPool> pool;
};

// Reads/writes the files of the key's directory in the file cache.
struct FileAccessor {
  explicit FileAccessor(const Key &key);

  // False if the file cache is disabled or unavailable.
  explicit operator bool() const { return !path.empty(); }

  // If checksum is true, the trailing checksum is verified and stripped.
  std::unique_ptr<llvm::MemoryBuffer> read(llvm::StringRef name,
                                           bool checksum = false);
  // If checksum is true, the checksum of the written content is appended.
  bool write(llvm::StringRef name,
             llvm::function_ref<bool(llvm::StringRef tmp)> writer,
             bool checksum = false);
  bool write(llvm::StringRef name, llvm::StringRef content);

private:
  llvm::StringRef file(llvm::StringRef name);

  llvm::SmallString<128> path;
  size_t dirLen = 0;
  bool dirCreated = false;
};

// AsyncCache backed by per-key directories in the file cache. The task gets an
// Accessor to read/write the key's files and must call Accessor::put().
template <typename E> struct FileCache final : private AsyncCache<E> {
  using typename AsyncCache<E>::FutureEntry;
  using AsyncCache<E>::AsyncCache;
  using AsyncCache<E>::async;

  struct Accessor : AsyncCache<E>::Accessor, FileAccessor {
    Accessor(AsyncCache<E> &cache, const Key &key,
             std::promise<std::shared_ptr<E>> promise)
        : AsyncCache<E>::Accessor(cache, key, std::move(promise)),
          FileAccessor(key) {}
  };

  template <typename MakeTask> FutureEntry get(Key key, MakeTask &&make) {
    return AsyncCache<E>::template get<Accessor>(key,
                                                 std::forward<MakeTask>(make));
  }
};
} // namespace gc::cache
