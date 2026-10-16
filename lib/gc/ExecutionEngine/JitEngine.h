//===-- JitEngine.h - JIT engine wrapper ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <utility>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

namespace mlir {
struct ExecutionEngine;
} // namespace mlir
namespace llvm::orc {
struct LLJIT;
} // namespace llvm::orc

namespace gc {
struct JitEngine {
  explicit JitEngine(std::unique_ptr<mlir::ExecutionEngine> eng);
  explicit JitEngine(std::unique_ptr<llvm::orc::LLJIT> jit);
  ~JitEngine();
  JitEngine(JitEngine &&) = default;

  llvm::Expected<void *> lookup(llvm::StringRef name) const;
  llvm::Expected<void (*)(void **)> lookupPacked(llvm::StringRef name) const;

  // Defines absolute symbols {name, address}.
  void registerSymbols(llvm::ArrayRef<std::pair<const char *, void *>> syms);

private:
  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::unique_ptr<llvm::orc::LLJIT> lljit;
};
} // namespace gc
