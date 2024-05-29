//===-- Driver.cpp - Top-level MLIR compiler driver -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "string.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace mlir {
namespace gc {

static DialectRegistry initDialects() {
  mlir::registerAllPasses();
  mlir::gc::registerGraphCompilerPasses();
  mlir::cpuruntime::registerCPURuntimePasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::cpuruntime::CPURuntimeDialect>();
  mlir::registerAllDialects(registry);
  mlir::cpuruntime::registerConvertCPURuntimeToLLVMInterface(registry);
  registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  mlir::registerAllToLLVMIRTranslations(registry);
  return registry;
}

const DialectRegistry &initAndGetDialects() {
  static DialectRegistry reg = initDialects();
  return reg;
}

static const char defaultComputeName[] = "_mlir_ciface_compute";
static const char defaultFoldName[] = "_mlir_ciface_fold";
llvm::Expected<std::shared_ptr<JitModule>>
JitModule::create(Operation *op, const ExecutionEngineOptions &options,
                  std::unique_ptr<llvm::TargetMachine> tm, bool transform) {
  if (transform) {
    mlir::PassManager pm{op->getContext()};
    populateCPUPipeline(pm);
    if (auto result = pm.run(op); failed(result)) {
      return llvm::make_error<llvm::StringError>(
          "MLIR pass error", llvm::inconvertibleErrorCode());
    }
  }
  auto exec = ExecutionEngine::create(op, options, std::move(tm));
  if (!exec) {
    return exec.takeError();
  }
  auto &engine = *exec;
  uint32_t numOrigArgs;
  {
    auto expectArgs = engine->lookup("__num_orig_num_args");
    if (!expectArgs) {
      return expectArgs.takeError();
    }
    numOrigArgs = *reinterpret_cast<uint32_t *>(*expectArgs);
  }
  JitModuleFuncT compute;
  {
    auto expectCompute = engine->lookupPacked(defaultComputeName);
    if (!expectCompute) {
      return expectCompute.takeError();
    }
    compute = *expectCompute;
  }
  llvm::ArrayRef<uint64_t> foldBufferIds;
  JitModuleFuncT fold = nullptr;
  llvm::ArrayRef<uint32_t> computeArgs;
  llvm::ArrayRef<uint32_t> foldArgs;
  do {
    auto expectBufferIds = engine->lookup("__fold_buffer_ids");
    if (!expectBufferIds) {
      // nothing to fold, It is OK.
      llvm::consumeError(expectBufferIds.takeError());
      // break out of the scope, don't need to lookup "fold" function
      break;
    } else {
      auto raw = reinterpret_cast<uint64_t *>(*expectBufferIds);
      foldBufferIds = llvm::ArrayRef<uint64_t>{raw + 1, raw[0]};
    }

    // find "fold" func
    {
      auto expectFold = engine->lookupPacked(defaultFoldName);
      if (!expectFold) {
        return expectFold.takeError();
      }
      fold = *expectFold;
    }

    // find "foldArgs"
    {
      auto expectFold = engine->lookup("__fold_args");
      if (!expectFold) {
        return expectFold.takeError();
      }
      auto raw = reinterpret_cast<uint32_t *>(*expectFold);
      foldArgs = llvm::ArrayRef<uint32_t>{raw + 1, raw[0]};
    }

    // find "computeArgs"
    {
      auto expect = engine->lookup("__compute_args");
      if (!expect) {
        return expect.takeError();
      }
      auto raw = reinterpret_cast<uint32_t *>(*expect);
      computeArgs = llvm::ArrayRef<uint32_t>{raw + 1, raw[0]};
    }
  } while (false);

  std::vector<std::shared_ptr<CachedGraphTensor>> foldInfo;
  foldInfo.reserve(foldBufferIds.size());
  for (auto bufId : foldBufferIds) {
    auto ret = queryCacheTensor(bufId);
    if (!ret) {
      return llvm::make_error<llvm::StringError>(
          "Failed to query the folded cached tensor",
          llvm::inconvertibleErrorCode());
    }
    foldInfo.emplace_back(std::move(ret));
  }

  return std::make_shared<JitModule>(std::move(engine), compute, fold,
                                     numOrigArgs, computeArgs, foldArgs,
                                     std::move(foldInfo));
}

JitModule::JitModule(
    std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT compute,
    JitModuleFuncT fold, size_t numOrigArgs,
    // The code inside `engine` has the ownership of the buffer
    llvm::ArrayRef<uint32_t> computeArgs,
    // The code inside `engine` has the ownership  of the buffer
    llvm::ArrayRef<uint32_t> foldArgs,
    std::vector<std::shared_ptr<CachedGraphTensor>> &&cachekeepAlive)
    : engine{std::move(engine)}, compute{compute}, fold{fold},
      numOrigArgs{numOrigArgs}, foldArgs{foldArgs},
      computeArgs{computeArgs}, keepAlive{std::move(cachekeepAlive)} {
  for (const auto &cache : keepAlive) {
    auto currentItr =
        std::find(cacheBases.begin(), cacheBases.end(), cache->base.get());
    if (currentItr == cacheBases.end()) {
      cacheBases.push_back(cache->base.get());
      currentItr = cacheBases.end() - 1;
    }
    cacheInfo.emplace_back(CacheBufferInfo{
        static_cast<size_t>(currentItr - cacheBases.begin()), cache->offset});
    fastFoldBuffers.push_back(&cache->ref);
  }
}
JitModule::~JitModule() = default;

static void prepareCallArgs(llvm::SmallVector<void *, 32> &realargs,
                            GeneralMemrefPtr *origargs, size_t numOrigArgs,
                            GeneralMemrefPtr *foldedCache,
                            llvm::ArrayRef<uint32_t> realArgIdx) {
  realargs.reserve(realArgIdx.size());
  for (auto argIdx : realArgIdx) {
    if (argIdx < numOrigArgs) {
      realargs.push_back(&origargs[argIdx]);
    } else {
      realargs.push_back(&foldedCache[argIdx - numOrigArgs]);
    }
  }
}

void JitModule::call(GeneralMemrefPtr *args) {
  if (unlikely(cacheInfo.empty())) {
    // fast path, no folded cached buffers
    // Silly code, MLIR execution engine requires pointers of real args as
    // inputs
    llvm::SmallVector<void *, 32> realargs;
    realargs.reserve(numOrigArgs);
    for (size_t i = 0; i < numOrigArgs; i++) {
      realargs.push_back(&args[i]);
    }
    compute(realargs.data());
    return;
  }

  // stage 1, acquire the foldBasePtr
  llvm::SmallVector<char *, 4> foldBasePtr;
  int32_t inited = 1;
  for (auto b : cacheBases) {
    auto ptr = b->acquire(&inited);
    if (unlikely(!ptr)) {
      ptr = std::aligned_alloc(/*alignment*/ 64, b->size);
      inited = 0;
    }
    foldBasePtr.push_back((char *)ptr);
  }

  // stage 2, run fold() if necessary
  GeneralMemrefPtr *foldedCache;
  // only used when !inited
  std::vector<GeneralMemrefPtr> slowFold;
  std::vector<StridedMemRefType<char, 8>> slowFoldObj;
  if (likely(inited)) {
    foldedCache = fastFoldBuffers.data();
  } else {
    slowFold.reserve(cacheInfo.size());
    slowFoldObj.reserve(cacheInfo.size());
    for (auto &info : cacheInfo) {
      slowFoldObj.emplace_back();
      auto &obj = slowFoldObj.back();
      obj.basePtr = foldBasePtr[info.baseIdx] + info.offset;
      obj.data = obj.basePtr;
      memset(obj.sizes, 0, sizeof(obj.sizes));
      memset(obj.strides, 0, sizeof(obj.strides));
      slowFold.push_back(&obj);
    }
    foldedCache = slowFold.data();
    llvm::SmallVector<void *, 32> realargs;
    prepareCallArgs(realargs, args, numOrigArgs, foldedCache, foldArgs);
    fold(realargs.data());
  }

  // stage 3, call compute
  {
    llvm::SmallVector<void *, 32> realargs;
    prepareCallArgs(realargs, args, numOrigArgs, foldedCache, computeArgs);
    compute(realargs.data());
  }

  // stage 4, cleanup
  for (size_t i = 0; i < cacheBases.size(); i++) {
    auto b = cacheBases[i];
    if (unlikely(!b->release())) {
      // if the cached buffer is already free'd, foldBasePtr[i] is allocated via
      // std::aligned_alloc by us, free it
      std::free(foldBasePtr[i]);
    }
  }
}

} // namespace gc
} // namespace mlir