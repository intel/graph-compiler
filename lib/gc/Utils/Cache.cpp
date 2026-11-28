//===-- Cache.cpp - Inmemory/file cache helpers -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _WIN32
#include <unistd.h>
#endif

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/BLAKE3.h>
#include <llvm/Support/Chrono.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include "gc/Utils/Cache.h"
#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"
#include "gc/Version.h"

using namespace llvm;
namespace fs = llvm::sys::fs;

namespace gc::cache {
namespace {
bool isFileCacheEnabled() {
  static bool enabled = []() {
    bool env = gcGetEnv("GC_FILE_CACHE_ON",
#ifdef GC_FILE_CACHE_ON
                        true
#else
                        false
#endif
    );
    if (!env) return false;

    if (getGpuCompilerInfo().empty()) {
      gcLogE("Failed to get the GPU compiler info, file cache disabled");
      return false;
    }
    return true;
  }();
  return enabled;
}

bool isChecksumEnabled() {
  static bool enabled = gcGetEnv("GC_FILE_CACHE_CHECKSUM",
#ifdef GC_FILE_CACHE_CHECKSUM
                                 true
#else
                                 false
#endif
  );
  return enabled;
}

// Owned by the current user and not writable by group/others.
bool isTrusted([[maybe_unused]] const fs::file_status &st) {
#ifdef _WIN32
  return true;
#else
  return st.getUser() == geteuid() &&
         !(st.permissions() & (fs::group_write | fs::others_write));
#endif
}

void cacheCleanup(StringRef root, int cleanInterval) {
  cleanInterval = gcGetEnv("GC_CACHE_CLEAN_INTERVAL", cleanInterval);
  if (cleanInterval <= 0) return;

  auto now = sys::TimePoint<>::clock::now();
  { // Check/update the last cleanup timestamp
    SmallString<128> stamp = root;
    sys::path::append(stamp, "cleanup.stamp");
    if (fs::file_status st;
        !fs::status(stamp, st) && now - st.getLastModificationTime() <
                                      std::chrono::hours(cleanInterval)) {
      return;
    }
    std::error_code ec;
    raw_fd_ostream os(stamp, ec);
    if (ec) {
      gcLogE("Failed to update cleanup stamp ", stamp.c_str());
      return;
    }
  }

  auto staleTime =
      now - std::chrono::hours(gcGetEnv("GC_CACHE_MAX_HOURS", 24 * 30));
  std::error_code ec;
  for (fs::directory_iterator itr(root, ec, false), end; !ec && itr != end;
       itr.increment(ec)) {
    if (auto st = itr->status();
        st && fs::is_directory(*st) && st->getLastAccessedTime() < staleTime) {
      auto &dir = itr->path();
      gcLogD("Removing stale cache folder ", dir.c_str());
      if (fs::remove_directories(dir))
        gcLogE("Failed to remove stale cache folder ", dir.c_str());
    }
  }
}

// Returns the cache root directory or an empty string, if the file cache is
// disabled. Stale cache directories are removed on the first call.
StringRef cacheRoot() {
  static const auto root = []() -> SmallString<128> {
    SmallString<128> path;
    if (!isFileCacheEnabled()) return path;

    auto env = gcGetEnv("GC_CACHE_DIR");
    if (env) {
      path = env;
    } else if (sys::path::cache_directory(path)) {
      sys::path::append(path, "intel", "graph-compiler");
    } else {
      gcLogE("Failed to resolve the cache directory, file cache disabled");
      return {};
    }

    if (fs::create_directories(path, true, fs::owner_all)) {
      gcLogE("Failed to create cache directory, file cache disabled: ",
             path.c_str());
      return {};
    }

    // Resolve after creation, realpath() fails on non-existent paths.
    if (SmallString<128> real; !fs::real_path(path, real)) {
      path = real;
    } else {
      gcLogE("Failed to resolve the cache directory real path, file cache "
             "disabled: ",
             path.c_str());
      return {};
    }

    // GC_CACHE_DIR is trusted, the internal cache dir must be owned and
    // writable by the user only.
    if (!env) {
      if (fs::file_status st;
          fs::status(path, st) || !fs::is_directory(st) || !isTrusted(st)) {
        gcLogE("Untrusted cache directory, file cache disabled: ",
               path.c_str());
        return {};
      }
    }

    cacheCleanup(path, env ? 0 : 24);
    gcLogD("Cache directory: ", path.c_str());
    return path;
  }();
  return root;
}

// Read the file only if it's a trusted regular file. If hasChecksum is true,
// verify and strip the trailing BLAKE3 checksum.
std::unique_ptr<MemoryBuffer> readFile(StringRef path, bool hasChecksum) {
  auto fd = fs::openNativeFileForRead(path);
  if (!fd) {
    consumeError(fd.takeError());
    return nullptr;
  }
  scope_exit close([&] { (void)fs::closeFile(*fd); });

  fs::file_status st;
  if (fs::status(*fd, st)) {
    gcLogE("Failed to get the file status: ", path);
    return nullptr;
  }
  if (st.type() != fs::file_type::regular_file || !isTrusted(st)) {
    gcLogE("Untrusted cache file: ", path);
    return nullptr;
  }

  hasChecksum = hasChecksum && isChecksumEnabled();
  BLAKE3Result<> sum;
  uint64_t size = st.getSize();
  if (hasChecksum) {
    size = std::max<uint64_t>(size, sum.size()) - sum.size();
    MutableArrayRef<char> dst(reinterpret_cast<char *>(sum.data()), sum.size());
    if (!size || expectedToOptional(fs::readNativeFileSlice(*fd, dst, size))
                         .value_or(0) != sum.size()) {
      gcLogE("Failed to read the cache file checksum: ", path);
      return nullptr;
    }
  }

  auto buf = MemoryBuffer::getOpenFile(*fd, path, size, false);
  if (!buf) {
    gcLogE("Failed to read the cache file: ", path);
    return nullptr;
  }
  if (hasChecksum &&
      BLAKE3::hash(arrayRefFromStringRef((*buf)->getBuffer())) != sum) {
    gcLogE("The cache file checksum mismatch: ", path);
    return nullptr;
  }
  return std::move(*buf);
}

// Write to a unique owner-only temp file, then atomically rename it.
// If appendChecksum is true, append the BLAKE3 checksum of the written content.
bool writeFile(StringRef path, function_ref<bool(StringRef)> writer,
               bool appendChecksum) {
  auto tmp = fs::TempFile::create(path + ".%%%%%%%%.tmp",
                                  fs::owner_read | fs::owner_write);
  if (!tmp) {
    consumeError(tmp.takeError());
    return false;
  }
  auto appendHash = [&] {
    auto fd = fs::convertFDToNativeFile(tmp->FD);
    BLAKE3Result<> sum;
    uint64_t size;
    {
      auto buf = MemoryBuffer::getOpenFile(fd, tmp->TmpName, -1, false);
      if (!buf) return false;
      size = (*buf)->getBufferSize();
      sum = BLAKE3::hash(arrayRefFromStringRef((*buf)->getBuffer()));
    }
    raw_fd_ostream os(tmp->FD, /*shouldClose=*/false);
    os.seek(size);
    os << toStringRef(sum);
    os.flush();
    bool ok = !os.has_error();
    os.clear_error();
    return ok;
  };
  if (writer(tmp->TmpName) &&
      (!appendChecksum || !isChecksumEnabled() || appendHash()) &&
      !errorToBool(tmp->keep(path)))
    return true;
  consumeError(tmp->discard());
  return false;
}
} // namespace

Key::Key(ArrayRef<StringRef> inputs) {
  BLAKE3 h;
  // Length-prefixed, so that adjacent inputs can't shift into each other.
  auto add = [&h](StringRef s) {
    uint64_t len = s.size();
    h.update(ArrayRef(reinterpret_cast<const uint8_t *>(&len), sizeof(len)));
    h.update(s);
  };
  add(GC_VERSION_STRING);
  add(getGpuCompilerInfo());
  add(isChecksumEnabled() ? "checksum" : "");
  for (auto &i : inputs) add(i);
  hash = h.final<std::tuple_size_v<decltype(hash)>>();
}

SmallString<40> Key::hex() const {
  SmallString<40> s;
  toHex(hash, true, s);
  return s;
}

FileAccessor::FileAccessor(const Key &key) {
  if (auto root = cacheRoot(); !root.empty()) {
    path = root;
    sys::path::append(path, key.hex());
    dirLen = path.size();
  }
}

std::unique_ptr<MemoryBuffer> FileAccessor::read(StringRef name,
                                                 bool checksum) {
  return *this ? readFile(file(name), checksum) : nullptr;
}

bool FileAccessor::write(StringRef name, function_ref<bool(StringRef)> writer,
                         bool checksum) {
  if (!*this) return false;
  if (!dirCreated) {
    path.truncate(dirLen);
    if (auto ec = fs::create_directories(path, true, fs::owner_all)) {
      gcLogE("Failed to create cache directory: ", path, " (", ec.message(),
             ")");
      path.clear();
      return false;
    }
    dirCreated = true;
  }
  if (writeFile(file(name), writer, checksum)) {
    gcLogD("Saved in file cache: ", path);
    return true;
  }
  gcLogE("Failed to save in file cache: ", path);
  return false;
}

bool FileAccessor::write(StringRef name, StringRef content) {
  return write(name, [content](StringRef tmp) {
    std::error_code ec;
    raw_fd_ostream os(tmp, ec);
    if (ec) return false;
    os << content;
    os.close();
    bool ok = !os.has_error();
    os.clear_error();
    return ok;
  });
}

StringRef FileAccessor::file(StringRef name) {
  path.truncate(dirLen);
  sys::path::append(path, name);
  return path;
}
} // namespace gc::cache
