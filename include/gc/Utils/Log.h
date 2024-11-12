//===-- Log.h - Logging functions -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_LOG_H
#define GC_LOG_H
#include <iostream>

#ifndef NDEBUG
#include <regex>
#endif

namespace mlir::gc::log {
static void insertArgs(std::ostream &stream) { stream << std::endl; }

template <typename T, typename... Args>
static void insertArgs(std::ostream &stream, T first, Args... args) {
  stream << first;
  insertArgs(stream, args...);
}

template <typename... Args>
static void log(
#ifndef NDEBUG
    const char *fileName, int lineNum,
#endif
    std::ostream &stream, const char *pref, Args... args) {
  stream << "[" << pref << "] ";
#ifndef NDEBUG
  stream << "[" << fileName << ":" << lineNum << "] ";
#endif
  insertArgs(stream, args...);
}

#ifdef NDEBUG
#define gcLogD(...)
#define gcLogE(...) mlir::gc::log::insetLog(std::cerr, "ERROR", __VA_ARGS__)
#else

// The debug logs are enabled by setting the environment variable GC_DEBUG to a
// regex pattern. The pattern is matched against the file name where the log is
// called. Examples:
//   GC_DEBUG=.*  - Enable all debug logs.
//   GC_DEBUG=/(CPU|GPU)Runtime/  - Enable debug logs in files containing
//   CPURuntime or GPURuntime in the path.
static bool isDebugEnabled(const char *fileName) {
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
static void debug(const char *fileName, int lineNum, Args... args) {
  if (isDebugEnabled(fileName)) {
    log(fileName, lineNum, std::cout, "DEBUG", args...);
  }
}

#define gcLogD(...) mlir::gc::log::debug(__FILE__, __LINE__, __VA_ARGS__)
#define gcLogE(...)                                                            \
  mlir::gc::log::log(__FILE__, __LINE__, std::cerr, "ERROR", __VA_ARGS__)
#define gcRunD(...) if (mlir::gc::log::isDebugEnabled(__FILE__)) {__VA_ARGS__;}
#endif
} // namespace mlir::gc::log

#endif
