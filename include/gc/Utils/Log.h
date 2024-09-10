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

namespace mlir::gc::log {
static void insertArgs(std::ostream &stream) { stream << std::endl; }

template <typename T, typename... Args>
static void insertArgs(std::ostream &stream, T first, Args... args) {
  stream << first;
  insertArgs(stream, args...);
}

template <typename... Args>
static void insetLog(
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
#define _insetLog(stream, pref, ...)                                           \
  mlir::gc::log::insetLog(__FILE__, __LINE__, stream, pref, __VA_ARGS__)
#define gcLogD(...) _insetLog(std::cout, "DEBUG", __VA_ARGS__)
#define gcLogE(...) _insetLog(std::cerr, "ERROR", __VA_ARGS__)
#endif
} // namespace mlir::gc::log

#ifdef GC_LOG_NO_DEBUG
#undef gcLogD
#define gcLogD(...)
#endif

#endif
