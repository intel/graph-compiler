//===-- Driver.cpp - Top-level MLIR compiler driver -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#endif
#include "gc/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "string.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "driver"

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
#ifdef GC_HAS_ONEDNN_DIALECT
  registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
#endif
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  mlir::registerAllToLLVMIRTranslations(registry);
  return registry;
}

const DialectRegistry &initCompilerAndGetDialects() {
  static DialectRegistry reg = initDialects();
  return reg;
}

static const char defaultEntryName[] = "_mlir_ciface_entry";
static const char defaultFoldName[] = "_mlir_ciface_runtime_fold";
llvm::Expected<std::shared_ptr<JitModule>>
JitModule::create(Operation *op, const DriverOptions &options) {
  if (options.runTransforms) {
    mlir::PassManager pm{op->getContext()};
    populateCPUPipeline(pm);
    if (auto result = pm.run(op); failed(result)) {
      return llvm::make_error<llvm::StringError>(
          "MLIR pass error", llvm::inconvertibleErrorCode());
    }
  }
  ExecutionEngineOptions exeOptions;
  exeOptions.jitCodeGenOptLevel = options.jitCodeGenOptLevel;
  std::unique_ptr<llvm::TargetMachine> tm = nullptr;
  auto exec = ExecutionEngine::create(op, exeOptions, std::move(tm));
  if (!exec) {
    return exec.takeError();
  }
  auto &engine = *exec;

  auto expectEntry = engine->lookupPacked(defaultEntryName);
  if (!expectEntry) {
    // entry function must exist
    return expectEntry.takeError();
  }
  JitModuleFuncT entry = *expectEntry;

  int32_t numOrigArgs;
  llvm::ArrayRef<int64_t> foldBufferIds;
  JitModuleFuncT fold = nullptr;
  llvm::ArrayRef<int32_t> entryArgs;
  llvm::ArrayRef<int32_t> foldArgs;
  do {
    {
      auto expectArgs = engine->lookup("__num_orig_args");
      if (!expectArgs) { // nothing to fold, It is OK.
        llvm::consumeError(expectArgs.takeError());
        // break out of the scope, don't need to lookup other things
        break;
      }
      numOrigArgs = *reinterpret_cast<int32_t *>(*expectArgs);
    }

    // If lookup("__num_orig_num_args") succeeds, then all the following should
    // also succeed.
    {
      auto expectBufferIds = engine->lookup("__runtime_fold_buffer_ids");
      if (!expectBufferIds) {
        expectBufferIds.takeError();
        break;
      }
      auto raw = reinterpret_cast<int64_t *>(*expectBufferIds);
      foldBufferIds =
          llvm::ArrayRef<int64_t>{raw + 1, static_cast<size_t>(raw[0])};
    }

    // find "fold" func
    {
      auto expectFold = engine->lookupPacked(defaultFoldName);
      if (!expectFold) {
        expectFold.takeError();
        break;
      }
      fold = *expectFold;
    }

    // find "foldArgs"
    {
      auto expectFold = engine->lookup("__fold_args");
      if (!expectFold) {
        expectFold.takeError();
        break;
      }
      auto raw = reinterpret_cast<int32_t *>(*expectFold);
      foldArgs = llvm::ArrayRef<int32_t>{raw + 1, static_cast<size_t>(raw[0])};
    }

    // find "entryArgs"
    {
      auto expect = engine->lookup("__compute_args");
      if (!expect) {
        expect.takeError();
        break;
      }
      auto raw = reinterpret_cast<int32_t *>(*expect);
      entryArgs = llvm::ArrayRef<int32_t>{raw + 1, static_cast<size_t>(raw[0])};
    }
  } while (false);

  std::vector<std::shared_ptr<CachedGraphTensor>> foldInfo;
  foldInfo.reserve(foldBufferIds.size());
  auto cacheManager = ConstGraphTensorCacheManager::get();
  for (auto bufId : foldBufferIds) {
    auto ret = cacheManager->queryCacheTensor(bufId);
    if (!ret) {
      return llvm::make_error<llvm::StringError>(
          "Failed to query the folded cached tensor of id: " +
              std::to_string(bufId),
          llvm::inconvertibleErrorCode());
    }
    foldInfo.emplace_back(std::move(ret));
  }

  return std::make_shared<JitModule>(std::move(engine), entry, fold,
                                     numOrigArgs, entryArgs, foldArgs,
                                     std::move(foldInfo));
}

JitModule::JitModule(
    std::unique_ptr<ExecutionEngine> engine, JitModuleFuncT entry,
    JitModuleFuncT fold, int32_t numOrigArgs,
    // The code inside `engine` has the ownership of the buffer
    llvm::ArrayRef<int32_t> entryArgs,
    // The code inside `engine` has the ownership  of the buffer
    llvm::ArrayRef<int32_t> foldArgs,
    std::vector<std::shared_ptr<CachedGraphTensor>> &&cachekeepAlive)
    : engine{std::move(engine)}, entry{entry}, fold{fold},
      numOrigArgs{numOrigArgs}, foldArgs{foldArgs}, entryArgs{entryArgs},
      keepAlive{std::move(cachekeepAlive)} {
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
                            GeneralMemrefPtr *origargs, int32_t numArgs,
                            int32_t numOrigArgs, GeneralMemrefPtr *foldedCache,
                            llvm::ArrayRef<int32_t> realArgIdx) {
  // inputs, including unfolded and folded
  realargs.reserve(realArgIdx.size());
  for (auto argIdx : realArgIdx) {
    if (argIdx < numOrigArgs) {
      realargs.push_back(&origargs[argIdx]);
    } else {
      realargs.push_back(&foldedCache[argIdx - numOrigArgs]);
    }
  }
  // outputs
  for (int i = numOrigArgs; i < numArgs; ++i) {
    realargs.push_back(&origargs[i]);
  }
}

void JitModule::call(GeneralMemrefPtr *args, int32_t numArgs) {
  if (unlikely(cacheInfo.empty())) {
    // fast path, no folded cached buffers
    // Silly code, MLIR execution engine requires pointers of real args as
    // inputs
    llvm::SmallVector<void *, 32> realargs;
    realargs.reserve(numArgs);
    for (int i = 0; i < numArgs; i++) {
      realargs.push_back(&args[i]);
    }
    entry(realargs.data());
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
    prepareCallArgs(realargs, args, numArgs, numOrigArgs, foldedCache,
                    foldArgs);
    LLVM_DEBUG(llvm::dbgs()
               << "fold func args size: " << foldArgs.size() << '\n');
    fold(realargs.data());
  }

  // stage 3, call entry
  {
    llvm::SmallVector<void *, 32> realargs;
    prepareCallArgs(realargs, args, numArgs, numOrigArgs, foldedCache,
                    entryArgs);
    LLVM_DEBUG(llvm::dbgs()
               << "entry func args size: " << realargs.size() << '\n');
    entry(realargs.data());
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