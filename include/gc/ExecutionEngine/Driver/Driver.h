//===-- Driver.h - The top-level MLIR compiler driver -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_EXECUTIONENGINE_DRIVER_DRIVER_H
#define GC_EXECUTIONENGINE_DRIVER_DRIVER_H

#include "gc/ExecutionEngine/CPURuntime/ConstantCache.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <memory>
#include <string_view>

namespace mlir {
class DialectRegistry;
namespace gc {

const DialectRegistry &initAndGetDialects();

// the pointers to XXXMemRefType
using GeneralMemrefPtr = void *;
using JitModuleFuncT = void (*)(void **);

class JitModule : public std::enable_shared_from_this<JitModule> {
public:
  static llvm::Expected<std::shared_ptr<JitModule>>
  create(Operation *op, const ExecutionEngineOptions &options = {},
         std::unique_ptr<llvm::TargetMachine> tm = nullptr,
         bool transform = true);

  // args should be an array of XXXMemrefType*
  void call(GeneralMemrefPtr *args);

  JitModule(
      std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT compute,
      JitModuleFuncT fold, size_t numOrigArgs,
      // The code inside `engine` has the ownership of the buffer
      llvm::ArrayRef<uint32_t> computeArgs,
      // The code inside `engine` has the ownership  of the buffer
      llvm::ArrayRef<uint32_t> foldArgs,
      std::vector<std::shared_ptr<CachedGraphTensor>> &&cachekeepAlive = {});
  ~JitModule();

private:
  std::unique_ptr<ExecutionEngine> engine;
  JitModuleFuncT compute;
  JitModuleFuncT fold;
  size_t numOrigArgs;
  // `keepAlive` has the ownership of the objects pointed by this vector
  llvm::SmallVector<ConstCacheProxy *> cacheBases;
  struct CacheBufferInfo {
    // index in cacheBases
    size_t baseIdx;
    size_t offset;
  };
  // the info for each folded cached buffer
  llvm::SmallVector<CacheBufferInfo, 8> cacheInfo;
  // holding the pointers to StridedMemRefType<T, Rank> of folded cache
  // `keepAlive` holds the the ownership of the pointers
  llvm::SmallVector<GeneralMemrefPtr> fastFoldBuffers;
  // The code inside `engine` has the ownership of the buffer
  llvm::ArrayRef<uint32_t> foldArgs;
  // The code inside `engine` has the ownership of the buffer
  llvm::ArrayRef<uint32_t> computeArgs;

  std::vector<std::shared_ptr<CachedGraphTensor>> keepAlive;
};

} // namespace gc
} // namespace mlir

#endif