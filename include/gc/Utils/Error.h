//===-- Error.h - Error processing functions --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_ERROR_H
#define GC_ERROR_H

#include <sstream>

#include "gc/Utils/Log.h"

#include "llvm/Support/Error.h"

namespace mlir::gc::err {
#ifdef NDEBUG
#define GC_ERR_LOC_DECL
#define GC_ERR_LOC_ARGS
#define GC_ERR_LOC
#else
#define GC_ERR_LOC_DECL const char *fileName, int lineNum,
#define GC_ERR_LOC_ARGS fileName, lineNum,
#define GC_ERR_LOC __FILE__, __LINE__,
#endif

#define gcMakeErr(...) mlir::gc::err::makeLlvmError(GC_ERR_LOC __VA_ARGS__)
#define gcReportErr(...)                                                       \
  mlir::gc::err::report(GC_ERR_LOC gcMakeErr(__VA_ARGS__))
#define gcGetOrReport(expected) mlir::gc::err::getOrReport(GC_ERR_LOC expected)

template <typename... Args>
[[nodiscard]] llvm::Error makeLlvmError(GC_ERR_LOC_DECL Args... args) {
  log::log(GC_ERR_LOC_ARGS std::cerr, "ERROR", args...);
  std::ostringstream oss;
  log::insertArgs(oss, args...);
  auto msg = oss.str();
  return llvm::make_error<llvm::StringError>(msg.substr(0, msg.length() - 1),
                                             llvm::inconvertibleErrorCode());
}

[[noreturn]] static void report(GC_ERR_LOC_DECL llvm::Error err) {
  log::log(GC_ERR_LOC_ARGS std::cerr, "ERROR", "Unrecoverable error!");
  report_fatal_error(std::move(err));
}

template <typename T>
T getOrReport(GC_ERR_LOC_DECL llvm::Expected<T> expected) {
  if (expected) {
    return *expected;
  }
  report(GC_ERR_LOC_ARGS expected.takeError());
}
} // namespace mlir::gc::err

#endif
