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

const DialectRegistry &initCompilerAndGetDialects();

// the pointers to XXXMemRefType
using GeneralMemrefPtr = void *;
using JitModuleFuncT = void (*)(void **);

struct DriverOptions {
  /// the optimization level for the LLVM-JIT
  llvm::CodeGenOptLevel jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  /// whether to run the MLIR transformation passes
  bool runTransforms = true;
  /// todo: target machine, etc.
};

class JitModule {
public:
  static llvm::Expected<std::shared_ptr<JitModule>>
  create(Operation *op, const DriverOptions &options = {});

  // args should be an array of XXXMemrefType*
  void call(GeneralMemrefPtr *args);

  /// args should be an array of XXXMemrefType*
  void call(GeneralMemrefPtr *args, std::size_t numArgs) {
    // Silly code, MLIR execution engine requires pointers of real args as
    // inputs
    llvm::SmallVector<void *, 32> realargs;
    realargs.reserve(numArgs);
    for (size_t i = 0; i < numArgs; i++) {
      realargs.push_back(&args[i]);
    }
    compute(realargs.data());
  }

  /// directly call compute(). args should be an array of void*. args[i] should
  /// be a pointer to the real data. For passing memref, users need to 1) create
  /// a pointer to XXXMemrefType 2) store the pointer to pointer to
  /// XXXMemrefType in args[i]
  void callRaw(void **args) { compute(args); }

  JitModule(std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT compute);

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