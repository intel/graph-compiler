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
  // numArgs: including input and output args.
  void call(GeneralMemrefPtr *args, int32_t numArgs);

  /// directly call entry(). args should be an array of void*. args[i] should
  /// be a pointer to the real data. For passing memref, users need to 1) create
  /// a pointer to XXXMemrefType 2) store the pointer to pointer to
  /// XXXMemrefType in args[i]
  void callRaw(void **args) { entry(args); }

  JitModule(std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT entry);

  JitModule(
      std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT entry,
      JitModuleFuncT fold, int32_t numOrigArgs,
      // The code inside `engine` has the ownership of the buffer
      llvm::ArrayRef<int32_t> entryArgs,
      // The code inside `engine` has the ownership  of the buffer
      llvm::ArrayRef<int32_t> foldArgs,
      std::vector<std::shared_ptr<CachedGraphTensor>> &&cachekeepAlive = {});
  ~JitModule();

private:
  std::unique_ptr<ExecutionEngine> engine;
  JitModuleFuncT entry;
  JitModuleFuncT fold;
  int32_t numOrigArgs; // only input args
  // The code inside `engine` has the ownership of the buffer
  llvm::ArrayRef<int32_t> foldArgs;
  // The code inside `engine` has the ownership of the buffer
  llvm::ArrayRef<int32_t> entryArgs;

  // The bases of CachedGraphTensors. For example, tensor1 (size 256) and
  // tensor2 (size 256) are in ConstCacheProxy base1, and tensor3 (size 256) in
  // base2. Then cacheBases is {base1, base2}, cacheInfo is {{baseIdx=0,
  // offset=0}, {baseIdx=0, offset=256}, {baseIdx=1, offset=0}}.

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
  llvm::SmallVector<GeneralMemrefPtr> fastFoldBuffers;
  // `keepAlive` holds the the ownership of the pointers
  std::vector<std::shared_ptr<CachedGraphTensor>> keepAlive;
};

} // namespace gc
} // namespace mlir

#endif