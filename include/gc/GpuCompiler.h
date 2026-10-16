//===-- GpuCompiler.h - GPU Compiler interface -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gc/Runtime/GpuRuntime.h"

namespace mlir {
class ModuleOp;
template <typename OpTy> class OwningOpRef;
} // namespace mlir

namespace gc {
struct JitEngine;

namespace gpu {
struct Program;
using FutureProgram = std::shared_future<std::shared_ptr<Program>>;

struct CompileOptions {
  CompileOptions() = default;
  CompileOptions(unsigned id) : arch(DeviceInfo::idToArch(id)) {}
  CompileOptions(const std::string &arch) : arch(arch) {}
  std::string arch;
  std::string igcOpts;
  bool wait{true};
  bool dump{false};
};

struct GpuCompiler {
  using Options = CompileOptions;

  GpuCompiler(unsigned concurrency = 0);
  ~GpuCompiler();
  GpuCompiler(const GpuCompiler &) = delete;
  GpuCompiler &operator=(const GpuCompiler &) = delete;
  GpuCompiler(GpuCompiler &&) = delete;
  GpuCompiler &operator=(GpuCompiler &&) = delete;

  // Returns a platform-wide shared instance of the GPU compiler.
  static GpuCompiler &get();

  // The program is linked with Rt::symbols(). Src is either
  // mlir::OwningOpRef<mlir::ModuleOp> or a string.
  template <typename Rt, typename Src>
  FutureProgram compile(Src &&src, Options opts = {}) {
    return compile(std::forward<Src>(src), std::move(opts),
                   Rt::template symbols<Rt>());
  }

  // The program is linked with RtSymbols. Takes ownership of src and runs the
  // pipeline in its context; the caller must have loaded the required dialects.
  FutureProgram compile(mlir::OwningOpRef<mlir::ModuleOp> src, Options opts,
                        const RtSymbols &syms);
  // Prevents implicit OwningOpRef construction, erasing the caller's module.
  FutureProgram compile(mlir::ModuleOp src, Options opts,
                        const RtSymbols &syms) = delete;
  FutureProgram compile(std::string_view src, Options opts,
                        const RtSymbols &syms);
};

struct Program {
  struct Symbols {
#define GC_PROG_SYM(name, type)                                                \
  static constexpr char name##Name[] = #name;                                  \
  using name##Type = type;                                                     \
  name##Type name

    GC_PROG_SYM(main, void (*)(void **));
    // The number of events returned by the main function. If > 0, the last
    // main argument points to an array of `eventNum` Event*, filled by main.
    GC_PROG_SYM(eventNum, uint32_t (*)());
    GC_PROG_SYM(destructor, void (*)());
#undef GC_PROG_SYM
  } const sym;

  uint32_t eventNum() const { return evtNum; }

  // Launch the program with the specified arguments. The args list contains
  // pointers to the actual values. The list must be built with
  // `ArgsBuilder::build()`. It can be used for multiple non-concurrent
  // launches, if not modified. If `eventNum()` is > 0, the function returns the
  // pointer to the first event, otherwise - returns nullptr. The returned
  // events must be consumed by the caller after each call. Rt must be the
  // runtime the program was compiled for.
  template <typename Rt, typename Vector = std::vector<void *>>
  typename Rt::Event **main(Vector &args, Rt *runtime,
                            typename Rt::Event *const *deps = nullptr,
                            uint32_t nDeps = 0) {
    assert(args.size() >= argsTailSize());
    void **data = args.data();
    auto pos = args.size() - argsTailSize();
    data[pos] = &runtime;
    data[pos + 1] = &deps;
    data[pos + 2] = &nDeps;
    sym.main(data);
    // data[pos + 3] is a pointer to data[pos + 4] which points to the array
    // of events, if eventNum() > 0.
    return eventNum() ? static_cast<typename Rt::Event **>(data[pos + 4])
                      : nullptr;
  }

  Program(std::unique_ptr<::gc::JitEngine> jit);
  ~Program();

private:
  uint32_t const evtNum;
  std::unique_ptr<::gc::JitEngine> jit;

  Program() = delete;
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;
  Program(Program &&) = delete;
  Program &operator=(Program &&) = delete;

  template <typename Args, typename Storage> friend struct ArgsBuilder;
  size_t argsTailSize() const { return 3 + (evtNum ? evtNum + 2 : 0); }
};

union ArgStorage {
  void *p;
  int64_t i;
  ArgStorage() = default;
  explicit ArgStorage(void *ptr) : p(ptr) {}
  explicit ArgStorage(int64_t val) : i(val) {}
};
// Storage::push_back must not invalidate references to existing elements.
template <typename Args = std::vector<void *>,
          typename Storage = std::deque<ArgStorage>>
struct ArgsBuilder {
  ArgsBuilder(Args &args, Storage &storage) : args(args), storage(storage) {
    reset();
  }
  // The value must remain alive and stable until the program returns.
  template <typename T> void append(T &value) { args.push_back(&value); }
  // All the pointers must remain alive and stable until the program returns.
  void appendMemref(void *&ptr, int64_t &offset, unsigned rank, int64_t *shape,
                    int64_t *strides) {
    append(ptr);
    append(ptr);
    append(offset);
    for (unsigned i = 0; i < rank; ++i) append(shape[i]);
    for (unsigned i = 0; i < rank; ++i) append(strides[i]);
  }

  void store(ArgStorage value) {
    storage.push_back(value);
    append(storage.back());
  }
  void storeMemref(void *ptr, unsigned rank = 0, int64_t *shape = nullptr,
                   int64_t *strides = nullptr, int64_t offset = 0) {
    assert(rank == 0 || (shape != nullptr && strides != nullptr));
    storeMemref(
        ptr, rank, [&](unsigned i) { return shape[i]; },
        [&](unsigned i) { return strides[i]; }, offset);
  }
  template <typename GetDim, typename GetStride>
  void storeMemref(void *ptr, unsigned rank, GetDim getDim, GetStride getStride,
                   int64_t offset = 0) {
    store(ArgStorage(ptr));
    args.push_back(args.back()); // Aligned pointer shares the allocated one.
    store(ArgStorage(offset));
    for (unsigned i = 0; i < rank; ++i) store(ArgStorage(getDim(i)));
    for (unsigned i = 0; i < rank; ++i) store(ArgStorage(getStride(i)));
  }

  // Build the argument list for the program. The events, returned by the
  // program (if any), are stored in the tail of the argument list. The list
  // can be used to launch the same program multiple times, but only if it's
  // not copied or modified and not concurrently.
  Args &build(const Program &prog) {
    auto pos = args.size();
    args.resize(pos + prog.argsTailSize());
    if (prog.eventNum()) {
      auto data = args.data();
      data[pos + 3] = &data[pos + 4];
      data[pos + 4] = &data[pos + 5];
    }
    return args;
  }

  void reset() {
    args.clear();
    storage.clear();
  }

private:
  Args &args;
  Storage &storage;
};
// Inline storage for N elements. On overflow, a Stable storage keeps the
// inline elements in place and appends the rest to the Heap, preserving
// references. Otherwise, all the elements move to the Heap, staying
// contiguous.
template <typename T, unsigned N, bool Stable = false,
          typename Heap =
              std::conditional_t<Stable, std::deque<T>, std::vector<T>>>
struct SmallStorage {
  void push_back(const T &value) {
    if ((Stable || !heap) && sz < N) buf[sz++] = value;
    else spill(2 * N).push_back(value);
  }
  T &back() { return heap && !heap->empty() ? heap->back() : buf[sz - 1]; }
  void resize(size_t n) {
    static_assert(!Stable, "Unsupported!");
    if (heap || n > N) return spill(n).resize(n);
    if (n > sz) std::fill(buf + sz, buf + n, T{});
    sz = n;
  }
  T *data() {
    static_assert(!Stable, "Unsupported!");
    return heap ? heap->data() : buf;
  }
  size_t size() const { return sz + (heap ? heap->size() : 0); }
  void clear() {
    sz = 0;
    if (heap) heap->clear();
  }

private:
  T buf[N];
  std::optional<Heap> heap;
  size_t sz = 0;

  Heap &spill(size_t cap) {
    if (heap) return *heap;
    heap.emplace();
    if constexpr (!Stable) {
      heap->reserve(std::max(cap, sz));
      heap->assign(buf, buf + sz);
      sz = 0;
    }
    return *heap;
  }
};
template <unsigned N> struct SmallArgs {
  SmallStorage<void *, N> argsBuf;
  SmallStorage<ArgStorage, N, true> valuesBuf;
};
// SmallArgs is the first base to be constructed before ArgsBuilder binds it.
template <unsigned N = 64>
struct SmallBuilder final
    : private SmallArgs<N>,
      ArgsBuilder<SmallStorage<void *, N>, SmallStorage<ArgStorage, N, true>> {
  using Base =
      ArgsBuilder<SmallStorage<void *, N>, SmallStorage<ArgStorage, N, true>>;
  SmallBuilder() : Base(this->argsBuf, this->valuesBuf) {}
  SmallBuilder(const SmallBuilder &) = delete;
};
} // namespace gpu
} // namespace gc
