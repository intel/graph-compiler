//===- Module.h - Jit module and Execution engine wrapper -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_JITWRAPPER_H
#define GC_EXECUTIONENGINE_JITWRAPPER_H

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <memory>
#include <string_view>

namespace mlir {
class DialectRegistry;
namespace gc {

const DialectRegistry &initAndGetDialects();

using JitModuleFuncT = void (*)(void **);

class JitModule : public std::enable_shared_from_this<JitModule> {
public:
  static llvm::Expected<std::shared_ptr<JitModule>>
  create(Operation *op, bool transform, llvm::StringRef entry_name = {},
         const ExecutionEngineOptions &options = {},
         std::unique_ptr<llvm::TargetMachine> tm = nullptr);

  void call(void **args) { entry(args); }

  JitModule(std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT entry);
  ~JitModule();

private:
  std::unique_ptr<ExecutionEngine> engine;
  JitModuleFuncT entry;
};

} // namespace gc
} // namespace mlir

#endif