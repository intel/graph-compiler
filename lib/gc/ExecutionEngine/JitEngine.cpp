//===-- JitEngine.cpp - JIT engine wrapper ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <utility>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>

#include "gc/ExecutionEngine/JitEngine.h"

using namespace llvm;
using namespace llvm::orc;

namespace gc {

JitEngine::JitEngine(std::unique_ptr<mlir::ExecutionEngine> eng)
    : engine(std::move(eng)) {}

JitEngine::JitEngine(std::unique_ptr<LLJIT> jit) : lljit(std::move(jit)) {}

JitEngine::~JitEngine() = default;

Expected<void *> JitEngine::lookup(StringRef name) const {
  if (engine) return engine->lookup(name);
  auto sym = lljit->lookup(name);
  if (!sym) return sym.takeError();
  return sym->toPtr<void *>();
}

Expected<void (*)(void **)> JitEngine::lookupPacked(StringRef name) const {
  if (engine) return engine->lookupPacked(name);
  auto result = lookup("_mlir_" + name.str());
  if (!result) return result.takeError();
  return reinterpret_cast<void (*)(void **)>(*result);
}

void JitEngine::registerSymbols(
    ArrayRef<std::pair<const char *, void *>> syms) {
  auto symbolMap = [syms](MangleAndInterner interner) {
    SymbolMap map;
    for (auto [name, addr] : syms)
      map[interner(name)] = {ExecutorAddr::fromPtr(addr),
                             JITSymbolFlags::Exported};
    return map;
  };
  if (engine) {
    engine->registerSymbols(symbolMap);
    return;
  }
  auto &mainJD = lljit->getMainJITDylib();
  cantFail(mainJD.define(absoluteSymbols(symbolMap(MangleAndInterner(
      mainJD.getExecutionSession(), lljit->getDataLayout())))));
}
} // namespace gc
