//===-- Log.h - Logging functions -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <iostream>

#ifndef NDEBUG
#include <cstdlib>
#include <regex>
#include <thread>
#endif

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>

namespace gc::log {
inline void insertArg(std::ostream &stream, llvm::StringRef value) {
  stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

template <unsigned N>
void insertArg(std::ostream &stream, const llvm::SmallString<N> &value) {
  stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

template <typename T> void insertArg(std::ostream &stream, const T &value) {
  stream << value;
}

inline void insertArgs(std::ostream &stream) { stream << std::endl; }

template <typename T, typename... Args>
void insertArgs(std::ostream &stream, const T &first, const Args &...args) {
  insertArg(stream, first);
  insertArgs(stream, args...);
}

template <typename... Args>
void log(
#ifndef NDEBUG
    const char *fileName, int lineNum,
#endif
    std::ostream &stream, const char *pref, const Args &...args) {
  stream << "[" << pref << "]";
#ifndef NDEBUG
  stream << "[" << std::this_thread::get_id() << "]";
  stream << "[" << fileName << ":" << lineNum << "] ";
#endif
  insertArgs(stream, args...);
}

#ifndef NDEBUG
// The debug logs are enabled by setting the environment variable GC_DEBUG to a
// regex pattern. The pattern is matched against the file name where the log is
// called. Examples:
//   GC_DEBUG=.*  - Enable all debug logs.
//   GC_DEBUG=/(CPU|GPU)Runtime/  - Enable debug logs in files containing
//   CPURuntime or GPURuntime in the path.
inline bool isDebugEnabled(const char *fileName) {
  static std::regex pattern = []() {
    auto env = std::getenv("GC_DEBUG");
    return env ? std::regex(env, std::regex::extended)
               : std::regex("", std::regex::basic);
  }();
  // The flag 'basic' is used as a marker for an empty regex.
  return pattern.flags() != std::regex::basic &&
         std::regex_search(fileName, pattern);
}

template <typename... Args>
void debug(const char *fileName, int lineNum, const Args &...args) {
  if (isDebugEnabled(fileName)) {
    log(fileName, lineNum, std::cout, "DEBUG", args...);
  }
}
#endif
} // namespace gc::log

#ifdef NDEBUG
#define gcLogD(...)
#define gcLogI(...) gc::log::log(std::cerr, "INFO", __VA_ARGS__)
#define gcLogE(...) gc::log::log(std::cerr, "ERROR", __VA_ARGS__)
#else

#define gcLogD(...) gc::log::debug(__FILE__, __LINE__, __VA_ARGS__)
#define gcLogE(...)                                                            \
  gc::log::log(__FILE__, __LINE__, std::cerr, "ERROR", __VA_ARGS__)
#define gcLogI(...)                                                            \
  gc::log::log(__FILE__, __LINE__, std::cerr, "INFO", __VA_ARGS__)
#define gcLogE(...)                                                            \
  gc::log::log(__FILE__, __LINE__, std::cerr, "ERROR", __VA_ARGS__)
#endif
