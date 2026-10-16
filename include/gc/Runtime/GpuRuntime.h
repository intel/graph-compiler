//===-- GpuRuntime.h - GPU runtime interface --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace gc::gpu {

struct DeviceInfo {
  size_t id{0};
  std::string name;
  std::string arch;
  uint32_t maxWgSize{0};

  DeviceInfo() = default;
  DeviceInfo(size_t id) : id(id), arch(idToArch(id)) {}

  static const char *idToArch(int id) {
    // Using device IDs from this source -
    // https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl
    switch (id) {
    case 0x674C:
    case 0x674D:
    case 0x674E:
    case 0x674F:
    case 0x6750:
      return "cri";
    case 0xE202:
    case 0xE209:
    case 0xE20B:
    case 0xE20C:
    case 0xE20D:
    case 0xE210:
    case 0xE211:
    case 0xE212:
    case 0xE215:
    case 0xE216:
    case 0xE220:
    case 0xE221:
    case 0xE222:
    case 0xE223:
      return "bmg";
    case 0x0BD0:
    case 0x0BD5:
    case 0x0BD6:
    case 0x0BD7:
    case 0x0BD8:
    case 0x0BD9:
    case 0x0BDA:
    case 0x0BDB:
    case 0x0B69:
    case 0x0B6E:
    case 0x0BD4:
      return "pvc";
    default:
      return "";
    }
  }
};

enum class ArgType : uint8_t {
  I8,
  UI8,
  I16,
  UI16,
  I32,
  UI32,
  I64,
  UI64,
  F32,
  F64,
  PTR,
};

constexpr uint8_t argSize(ArgType type) noexcept {
  switch (type) {
  case ArgType::I8:
  case ArgType::UI8:
    return sizeof(uint8_t);
  case ArgType::I16:
  case ArgType::UI16:
    return sizeof(uint16_t);
  case ArgType::I32:
  case ArgType::UI32:
    return sizeof(uint32_t);
  case ArgType::I64:
  case ArgType::UI64:
    return sizeof(uint64_t);
  case ArgType::F32:
    return sizeof(float);
  case ArgType::F64:
    return sizeof(double);
  case ArgType::PTR:
    return sizeof(void *);
  }
  return 0;
}

// Functions called by the generated code.
namespace sym {
#define GC_RT_SYM(name) inline constexpr char name##Name[] = #name
GC_RT_SYM(gcGpuCreateKernel);
GC_RT_SYM(gcGpuDestroyKernel);
GC_RT_SYM(gcGpuLaunch);
GC_RT_SYM(gcGpuAllocDevice);
GC_RT_SYM(gcGpuAllocShared);
GC_RT_SYM(gcGpuAllocHost);
GC_RT_SYM(gcGpuFree);
GC_RT_SYM(gcGpuMemcpy);
GC_RT_SYM(gcGpuWait);
#undef GC_RT_SYM
} // namespace sym

using SymbolAddr = std::pair<const char *, void *>;
using RtSymbols = std::array<SymbolAddr, 9>;

template <typename K, typename E, typename M> struct GpuRuntime {
  using Kernel = K;
  using Event = E;
  using Memory = M;

  GpuRuntime() = default;
  virtual ~GpuRuntime() = default;
  GpuRuntime(const GpuRuntime &) = delete;
  GpuRuntime &operator=(const GpuRuntime &) = delete;
  GpuRuntime(GpuRuntime &&) = delete;
  GpuRuntime &operator=(GpuRuntime &&) = delete;

  virtual const DeviceInfo &getDeviceInfo() const = 0;

  // The caller is responsible for the kernel destruction before the runtime,
  // via the static Rt::destroyKernel(Kernel *) of the implementation.
  virtual Kernel *createKernel(const uint8_t *data, size_t size,
                               const char *name, uint32_t nArgs,
                               ArgType *argTypes) = 0;
  // **args is an array of pointers to the kernel arguments.
  // The caller is responsible for releasing the events by calling wait().
  virtual Event *launch(Kernel *kernel, void **args, uint32_t nArgs,
                        uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                        uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                        Event *const *deps, uint32_t nDeps) = 0;
  virtual Memory *allocDevice(size_t size, size_t alignment) = 0;
  virtual Memory *allocShared(size_t size, size_t alignment) = 0;
  virtual Memory *allocHost(size_t size, size_t alignment) = 0;
  virtual void free(Memory *ptr) = 0;
  virtual Event *memcpy(const Memory *src, Memory *dst, size_t size,
                        Event *const *deps, uint32_t nDeps) = 0;
  // The events ownership is transferred back to the runtime.
  virtual void wait(Event *const *events, uint32_t nEvents) = 0;

  // The following symbols are registered in the ExecutionEngine.
  template <typename Rt> static const RtSymbols &symbols() {
    static_assert(std::is_base_of_v<GpuRuntime, Rt>);
    auto entry = [](const char *name, auto *fn) {
      return SymbolAddr{name, reinterpret_cast<void *>(fn)};
    };
    static const std::array syms{
        entry(
            sym::gcGpuCreateKernelName,
            +[](Rt *rt, const uint8_t *data, size_t size, const char *name,
                uint32_t nArgs, ArgType *argTypes) {
              return rt->createKernel(data, size, name, nArgs, argTypes);
            }),
        entry(
            sym::gcGpuDestroyKernelName,
            +[](Kernel *kernel) { Rt::destroyKernel(kernel); }),
        entry(
            sym::gcGpuLaunchName,
            +[](Rt *rt, Kernel *kernel, void **args, uint32_t nArgs,
                uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                uint32_t blockY, uint32_t blockZ, Event *const *deps,
                uint32_t nDeps) {
              return rt->launch(kernel, args, nArgs, gridX, gridY, gridZ,
                                blockX, blockY, blockZ, deps, nDeps);
            }),
        entry(
            sym::gcGpuAllocDeviceName,
            +[](Rt *rt, size_t size, size_t alignment) {
              return rt->allocDevice(size, alignment);
            }),
        entry(
            sym::gcGpuAllocSharedName,
            +[](Rt *rt, size_t size, size_t alignment) {
              return rt->allocShared(size, alignment);
            }),
        entry(
            sym::gcGpuAllocHostName,
            +[](Rt *rt, size_t size, size_t alignment) {
              return rt->allocHost(size, alignment);
            }),
        entry(
            sym::gcGpuFreeName, +[](Rt *rt, Memory *ptr) { rt->free(ptr); }),
        entry(
            sym::gcGpuMemcpyName,
            +[](Rt *rt, const Memory *src, Memory *dst, size_t size,
                Event *const *deps, uint32_t nDeps) {
              return rt->memcpy(src, dst, size, deps, nDeps);
            }),
        entry(
            sym::gcGpuWaitName,
            +[](Rt *rt, Event *const *events, uint32_t nEvents) {
              rt->wait(events, nEvents);
            }),
    };
    return syms;
  }
};

} // namespace gc::gpu
