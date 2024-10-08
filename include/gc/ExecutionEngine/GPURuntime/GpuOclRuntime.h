//===-- GpuOclRuntime.h - GPU OpenCL runtime --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_GPUOCLRUNTIME_H
#define GC_GPUOCLRUNTIME_H

namespace mlir::gc::gpu {
constexpr char GPU_OCL_MALLOC_DEV[] = "gcGpuOclMallocDev";
constexpr char GPU_OCL_MALLOC_SHARED[] = "gcGpuOclMallocShared";
constexpr char GPU_OCL_DEALLOC[] = "gcGpuOclDealloc";
constexpr char GPU_OCL_MEMCPY[] = "gcGpuOclMemcpy";
constexpr char GPU_OCL_KERNEL_CREATE[] = "gcGpuOclKernelCreate";
constexpr char GPU_OCL_KERNEL_DESTROY[] = "gcGpuOclKernelDestroy";
constexpr char GPU_OCL_KERNEL_LAUNCH[] = "gcGpuOclKernelLaunch";
constexpr char GPU_OCL_FINISH[] = "gcGpuOclFinish";
constexpr char GPU_OCL_MOD_DESTRUCTOR[] = "gcGpuOclModuleDestructor";
} // namespace mlir::gc::gpu

#ifndef GC_GPU_OCL_CONST_ONLY
#include <cstdarg>
#include <unordered_set>
#include <vector>

#include <CL/cl.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::gc::gpu {
struct OclDevCtxPair {
  cl_device_id device;
  cl_context context;
  explicit OclDevCtxPair(cl_device_id device, cl_context context)
      : device(device), context(context) {}

  bool operator==(const OclDevCtxPair &other) const {
    return device == other.device && context == other.context;
  }
};
} // namespace mlir::gc::gpu
template <> struct std::hash<const mlir::gc::gpu::OclDevCtxPair> {
  std::size_t
  operator()(const mlir::gc::gpu::OclDevCtxPair &pair) const noexcept {
    return std::hash<cl_device_id>()(pair.device) ^
           std::hash<cl_context>()(pair.context);
  }
}; // namespace std
namespace mlir::gc::gpu {
struct OclModule;
struct OclContext;
struct OclModuleBuilder;

struct OclRuntime {
  // Returns the available Intel GPU device ids.
  [[nodiscard]] static llvm::Expected<SmallVector<cl_device_id, 2>>
  gcIntelDevices(size_t max = std::numeric_limits<size_t>::max());

  [[nodiscard]] static llvm::Expected<OclRuntime> get();

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_device_id device);

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_command_queue queue);

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_device_id device,
                                                      cl_context context);

  static bool isOutOfOrder(cl_command_queue queue);

  [[nodiscard]] cl_context getContext() const;

  [[nodiscard]] cl_device_id getDevice() const;

  [[nodiscard]] llvm::Expected<cl_command_queue>
  createQueue(bool outOfOrder = false) const;

  [[nodiscard]] static llvm::Expected<bool>
  releaseQueue(cl_command_queue queue);

  [[nodiscard]] llvm::Expected<void *> usmAllocDev(size_t size) const;

  [[nodiscard]] llvm::Expected<void *> usmAllocShared(size_t size) const;

  [[nodiscard]] llvm::Expected<bool> usmFree(const void *ptr) const;

  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext &ctx, const void *src,
                                            void *dst, size_t size) const;

  template <typename T>
  [[nodiscard]] llvm::Expected<T *> usmNewDev(size_t size) const {
    auto expected = usmAllocDev(size * sizeof(T));
    if (expected) {
      return static_cast<T *>(*expected);
    }
    return expected.takeError();
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<T *> usmNewShared(size_t size) const {
    auto expected = usmAllocShared(size * sizeof(T));
    if (expected) {
      return static_cast<T *>(*expected);
    }
    return expected.takeError();
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext &ctx, const T *src,
                                            T *dst, size_t size) const {
    return usmCpy(ctx, static_cast<const void *>(src), static_cast<void *>(dst),
                  size * sizeof(T));
  }

  // Use with caution! This is safe to check validity of USM, but may be false
  // positive for any other kinds.
  bool isUsm(const void *ptr) const;

  bool operator==(const OclRuntime &other) const {
    return getDevice() == other.getDevice() &&
           getContext() == other.getContext();
  }

private:
  struct Ext;
  struct Exports;
  friend OclContext;
  friend OclModuleBuilder;
  template <unsigned N> friend struct DynamicExecutor;
  template <unsigned N> friend struct StaticExecutor;
  explicit OclRuntime(const Ext &ext);
  const Ext &ext;

#ifndef NDEBUG
  static void debug(const char *file, int line, const char *msg);
#endif
};

static constexpr int64_t ZERO = 0;
static constexpr auto ZERO_PTR = const_cast<int64_t *>(&ZERO);

// NOTE: The context is mutable and not thread-safe! It's expected to be used in
// a single thread only.
struct OclContext {
  const OclRuntime &runtime;
  const cl_command_queue queue;
  // Preserve the execution order. This is required in case of out-of-order
  // execution (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE). When the execution
  // is completed, the 'lastEvent' field contains the event of the last enqueued
  // command. If this field is false, 'waitList' is ignored.
  const bool preserveOrder;
  cl_uint waitListLen;
  cl_event *waitList;
  cl_event lastEvent;

  explicit OclContext(const OclRuntime &runtime, cl_command_queue queue,
                      cl_uint waitListLen = 0, cl_event *waitList = nullptr)
      : OclContext(runtime, queue, OclRuntime::isOutOfOrder(queue), waitListLen,
                   waitList) {}

  explicit OclContext(const OclRuntime &runtime, cl_command_queue queue,
                      bool preserveOrder, cl_uint waitListLen,
                      cl_event *waitList)
      : runtime(runtime), queue(queue), preserveOrder(preserveOrder),
        waitListLen(preserveOrder ? waitListLen : 0),
        waitList(preserveOrder ? waitList : nullptr), lastEvent(nullptr),
        clPtrs(nullptr) {
    assert(!OclRuntime::isOutOfOrder(queue) || preserveOrder);
    assert(preserveOrder || (waitListLen == 0 && waitList == nullptr));
  }

  OclContext(const OclContext &) = delete;
  OclContext &operator=(const OclContext &) = delete;

  [[nodiscard]] llvm::Expected<bool> finish();

private:
  friend OclRuntime;
  friend OclRuntime::Exports;
  template <unsigned N> friend struct DynamicExecutor;
  template <unsigned N> friend struct StaticExecutor;
  std::unordered_set<void *> *clPtrs;

  void setLastEvent(cl_event event) {
    lastEvent = event;
    if (event) {
      waitListLen = 1;
      waitList = &lastEvent;
    } else {
      waitListLen = 0;
      waitList = nullptr;
    }
  }
};

struct OclModule {
  const OclRuntime runtime;
  // If all the function arguments have static shapes, then this field is true
  // and main.staticMain is used. Otherwise, main.wrappedMain is used.
  const bool isStatic;

  ~OclModule();
  OclModule(const OclModule &) = delete;
  OclModule &operator=(const OclModule &) = delete;
  OclModule(const OclModule &&) = delete;
  OclModule &operator=(const OclModule &&) = delete;

  void dumpToObjectFile(StringRef filename) const {
    engine->dumpToObjectFile(filename);
  }

private:
  friend OclModuleBuilder;
  template <unsigned N> friend struct DynamicExecutor;
  template <unsigned N> friend struct OclModuleExecutorBase;
  template <unsigned N> friend struct StaticExecutor;
  // This function is only created when all args are memrefs with static shape.
  using StaticMainFunc = void (*)(OclContext *, void **);
  // Wrapper, generated by the engine. The arguments are pointers to the values.
  using WrappedMainFunc = void (*)(void **);
  union MainFunc {
    StaticMainFunc staticMain;
    WrappedMainFunc wrappedMain;
  };
  const MainFunc main;
  const ArrayRef<Type> argTypes;
  std::unique_ptr<ExecutionEngine> engine;

  explicit OclModule(const OclRuntime &runtime, const bool isStatic,
                     const MainFunc main, const ArrayRef<Type> argTypes,
                     std::unique_ptr<ExecutionEngine> engine)
      : runtime(runtime), isStatic(isStatic), main(main), argTypes(argTypes),
        engine(std::move(engine)) {}
};

struct OclModuleBuilderOpts {
  StringRef funcName = {};
  bool printIr = false;
  bool enableObjectDump = false;
  ArrayRef<StringRef> sharedLibPaths = {};
  void (*pipeline)(OpPassManager &) = nullptr;
};

struct OclModuleBuilder {
  friend OclRuntime;
  explicit OclModuleBuilder(ModuleOp module,
                            const OclModuleBuilderOpts &opts = {});
  explicit OclModuleBuilder(OwningOpRef<ModuleOp> &module,
                            const OclModuleBuilderOpts &opts = {})
      : OclModuleBuilder(module.release(), opts) {}
  explicit OclModuleBuilder(OwningOpRef<ModuleOp> &&module,
                            const OclModuleBuilderOpts &opts = {})
      : OclModuleBuilder(module.release(), opts) {}

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(const OclRuntime &runtime);

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(cl_command_queue queue);

  llvm::Expected<std::shared_ptr<const OclModule>> build(cl_device_id device,
                                                         cl_context context);

private:
  ModuleOp mlirModule;
  const bool printIr;
  const bool enableObjectDump;
  const ArrayRef<StringRef> sharedLibPaths;
  void (*const pipeline)(OpPassManager &);
  const StringRef funcName;
  const ArrayRef<Type> argTypes;
  std::shared_mutex mux;
  std::unordered_map<const OclDevCtxPair, std::shared_ptr<const OclModule>>
      cache;

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(const OclRuntime::Ext &ext);
};

// NOTE: This class is mutable and not thread-safe!
template <unsigned N> struct OclModuleExecutorBase {

  void reset() {
    args.clear();
    clPtrs.clear();
    argCounter = 0;
  }

  Type getArgType(unsigned idx) const { return mod->argTypes[idx]; }

  [[nodiscard]] bool isSmall() const { return args.small(); }

protected:
  struct Args : SmallVector<void *, N> {
    [[nodiscard]] bool small() const { return this->isSmall(); }
  };

  const std::shared_ptr<const OclModule> &mod;
  // Contains the pointers of all non-USM arguments. It's expected, that the
  // arguments are either USM or CL pointers and most probably are USM, thus,
  // in most cases, this set will be empty.
  std::unordered_set<void *> clPtrs;
  Args args;
  unsigned argCounter = 0;

  explicit OclModuleExecutorBase(std::shared_ptr<const OclModule> &mod)
      : mod(mod) {}

#ifndef NDEBUG
  void checkExec(const OclContext &ctx) const {
    auto rt = OclRuntime::get(ctx.queue);
    assert(rt);
    assert(*rt == mod->runtime);
    assert(argCounter == mod->argTypes.size());
  }

  void checkArg(const void *alignedPtr, bool isUsm = true) const {
    assert(!isUsm || mod->runtime.isUsm(alignedPtr));
  }
#endif
};

// NOTE: This executor can only be used if mod->isStatic == true!
template <unsigned N = 8> struct StaticExecutor : OclModuleExecutorBase<N> {
  explicit StaticExecutor(std::shared_ptr<const OclModule> &mod)
      : OclModuleExecutorBase<N>(mod) {
    assert(this->mod->isStatic);
  }

  void exec(OclContext &ctx) {
#ifndef NDEBUG
    this->checkExec(ctx);
#endif
    ctx.clPtrs = &this->clPtrs;
    this->mod->main.staticMain(&ctx, this->args.data());
  }

  void arg(void *alignedPtr, bool isUsm = true) {
#ifndef NDEBUG
    this->checkArg(alignedPtr, isUsm);
    std::ostringstream oss;
    oss << "Arg" << this->argCounter << ": alignedPtr=" << alignedPtr
        << ", isUsm=" << (isUsm ? "true" : "false");
    OclRuntime::debug(__FILE__, __LINE__, oss.str().c_str());
#endif
    ++this->argCounter;
    this->args.emplace_back(alignedPtr);
    if (!isUsm) {
      this->clPtrs.insert(alignedPtr);
    }
  }

  template <typename T> void arg(T *alignedPtr, bool isUsm = true) {
    arg(reinterpret_cast<void *>(alignedPtr), isUsm);
  }

  void operator()(OclContext &ctx) { exec(ctx); }

  template <typename T> void operator()(OclContext &ctx, T *ptr1, ...) {
    {
      this->reset();
      arg(reinterpret_cast<void *>(ptr1));
      va_list args;
      va_start(args, ptr1);
      for (unsigned i = 0, n = this->mod->argTypes.size() - 1; i < n; i++) {
        arg(va_arg(args, void *));
      }
      va_end(args);
      exec(ctx);
    }
  }
};

// The main function arguments are added in the following format -
// https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission.
// NOTE: This executor can only be used if mod->isStatic != true!
template <unsigned N = 64> struct DynamicExecutor : OclModuleExecutorBase<N> {
  explicit DynamicExecutor(std::shared_ptr<const OclModule> &mod)
      : OclModuleExecutorBase<N>(mod) {
    assert(!this->mod->isStatic);
  }

  void exec(OclContext &ctx) {
#ifndef NDEBUG
    this->checkExec(ctx);
#endif
    auto size = this->args.size();
    auto ctxPtr = &ctx;
    this->args.emplace_back(&ctxPtr);
    this->args.emplace_back(&ctxPtr);
    this->args.emplace_back(ZERO_PTR);
    this->mod->main.wrappedMain(this->args.data());
    this->args.truncate(size);
  }

  void arg(void *&alignedPtr, size_t rank, const int64_t *shape,
           const int64_t *strides, bool isUsm = true) {
    arg(alignedPtr, alignedPtr, ZERO, rank, shape, strides, isUsm);
  }

  // NOTE: The argument values are not copied, only the pointers are stored!
  void arg(void *&allocatedPtr, void *&alignedPtr, const int64_t &offset,
           size_t rank, const int64_t *shape, const int64_t *strides,
           bool isUsm = true) {
#ifndef NDEBUG
    this->checkArg(alignedPtr, isUsm);
    if (auto type =
            llvm::dyn_cast<MemRefType>(this->getArgType(this->argCounter))) {
      if (type.hasStaticShape()) {
        auto size = type.getShape();
        assert(rank == size.size());
        for (size_t i = 0; i < rank; i++) {
          assert(shape[i] == size[i]);
        }

        SmallVector<int64_t> expectedStrides;
        if (int64_t expectedOffset; !failed(
                getStridesAndOffset(type, expectedStrides, expectedOffset))) {
          assert(expectedOffset == offset);
          for (size_t i = 0; i < rank; i++) {
            assert(expectedStrides[i] == strides[i]);
          }
        }
      }
    }

    std::ostringstream oss;
    oss << "Arg" << this->argCounter << ": ptr=" << allocatedPtr
        << ", alignedPtr=" << alignedPtr
        << ", isUsm=" << (isUsm ? "true" : "false") << ", offset=" << offset
        << ", shape=[";
    for (unsigned i = 0; i < rank; i++) {
      oss << shape[i] << (i + 1 < rank ? ", " : "]");
    }
    oss << ", strides=[";
    for (unsigned i = 0; i < rank; i++) {
      oss << strides[i] << (i + 1 < rank ? ", " : "]");
    }
    OclRuntime::debug(__FILE__, __LINE__, oss.str().c_str());
#endif

    ++this->argCounter;
    this->args.emplace_back(&allocatedPtr);
    this->args.emplace_back(&alignedPtr);
    this->args.emplace_back(const_cast<int64_t *>(&offset));
    for (size_t i = 0; i < rank; i++) {
      this->args.emplace_back(const_cast<int64_t *>(&shape[i]));
    }
    for (size_t i = 0; i < rank; i++) {
      this->args.emplace_back(const_cast<int64_t *>(&strides[i]));
    }
    if (!isUsm) {
      this->clPtrs.insert(alignedPtr);
    }
  }

  template <typename T>
  void arg(T *&alignedPtr, size_t rank, const int64_t *shape,
           const int64_t *strides, bool isUsm = true) {
    arg(reinterpret_cast<void *&>(alignedPtr), rank, shape, strides, isUsm);
  }

  template <typename T>
  void arg(T *&allocatedPtr, T *&alignedPtr, const int64_t &offset, size_t rank,
           const int64_t *shape, const int64_t *strides, bool isUsm = true) {
    arg(reinterpret_cast<void *&>(allocatedPtr),
        reinterpret_cast<void *&>(alignedPtr), offset, rank, shape, strides,
        isUsm);
  }

  void operator()(OclContext &ctx) { exec(ctx); }
};
} // namespace mlir::gc::gpu
#else
#undef GC_GPU_OCL_CONST_ONLY
#endif
#endif
