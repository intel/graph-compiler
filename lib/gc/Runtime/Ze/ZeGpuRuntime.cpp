//===-- ZeGpuRuntime.cpp - Level Zero GpuRuntime ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <level_zero/ze_api.h>
#include <level_zero/ze_ddi.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include "gc/Runtime/ZeGpuRuntime.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"

#define ZE_CHECKR(expr, ...)                                                   \
  do {                                                                         \
    if (auto _ze_check_err = (expr); _ze_check_err != ZE_RESULT_SUCCESS) {     \
      gcReportErr("L0 error ", _ze_check_err, ": ", __VA_ARGS__);              \
    }                                                                          \
  } while (0)

// Non-fatal check for destructors: a fatal error may already be running exit.
#define ZE_CHECKL(expr, ...)                                                   \
  do {                                                                         \
    if (auto _ze_check_err = (expr); _ze_check_err != ZE_RESULT_SUCCESS) {     \
      gcLogE("L0 error ", _ze_check_err, ": ", __VA_ARGS__);                   \
    }                                                                          \
  } while (0)

namespace {
using namespace gc::gpu;

bool devInfo(ze_device_handle_t device, DeviceInfo &info) {
  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (zeDeviceGetProperties(device, &props) != ZE_RESULT_SUCCESS ||
      props.type != ZE_DEVICE_TYPE_GPU || props.vendorId != 0x8086 ||
      props.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)
    return false;
  ze_device_compute_properties_t cp{};
  cp.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  if (zeDeviceGetComputeProperties(device, &cp) != ZE_RESULT_SUCCESS)
    return false;
  info.id = props.deviceId;
  info.name = props.name;
  info.arch = DeviceInfo::idToArch(props.deviceId);
  info.maxWgSize = cp.maxTotalGroupSize;
  return true;
}

bool getDrivers(llvm::SmallVectorImpl<ze_driver_handle_t> &drivers) {
  uint32_t nDrivers = 0;
  if (zeDriverGet(&nDrivers, nullptr) != ZE_RESULT_SUCCESS || !nDrivers) {
    gcLogE("No L0 driver found.");
    return false;
  }
  drivers.resize(nDrivers);
  if (zeDriverGet(&nDrivers, drivers.data()) != ZE_RESULT_SUCCESS) {
    gcLogE("Failed to get L0 drivers.");
    return false;
  }
  return true;
}

// Prefers devices with a known arch, then the highest device id.
bool selectDevice(ze_driver_handle_t &driver, ze_device_handle_t &device,
                  DeviceInfo &info) {
  llvm::SmallVector<ze_driver_handle_t> drivers;
  if (!getDrivers(drivers)) return false;

  auto rank = [](const DeviceInfo &i) {
    return std::pair(!i.arch.empty(), i.id);
  };
  device = nullptr;
  for (auto drv : drivers) {
    uint32_t nDev = 0;
    if (zeDeviceGet(drv, &nDev, nullptr) != ZE_RESULT_SUCCESS || !nDev)
      continue;
    llvm::SmallVector<ze_device_handle_t> devices(nDev);
    if (zeDeviceGet(drv, &nDev, devices.data()) != ZE_RESULT_SUCCESS) {
      gcLogE("Failed to get L0 devices.");
      continue;
    }
    for (auto dev : devices) {
      DeviceInfo di;
      if (devInfo(dev, di) && (!device || rank(di) > rank(info))) {
        driver = drv;
        device = dev;
        info = std::move(di);
      }
    }
  }
  if (!device) return false;
  gcLogD("Selected GPU device ", info.name, ", arch: ", info.arch,
         ", id: ", info.id);
  return true;
}

ze_command_list_handle_t createCmdList(ze_context_handle_t context,
                                       ze_device_handle_t device) {
  ze_command_queue_desc_t qd{};
  qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.ordinal = [&]() -> uint32_t {
    uint32_t n = 0;
    if (zeDeviceGetCommandQueueGroupProperties(device, &n, nullptr) !=
            ZE_RESULT_SUCCESS ||
        !n)
      return 0;
    ze_command_queue_group_properties_t gp{};
    gp.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    llvm::SmallVector<ze_command_queue_group_properties_t> groups(n, gp);
    if (zeDeviceGetCommandQueueGroupProperties(device, &n, groups.data()) !=
        ZE_RESULT_SUCCESS)
      return 0;
    auto it = llvm::find_if(groups, [](auto &g) {
      return g.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    });
    return it == groups.end() ? 0 : it - groups.begin();
  }();
  ze_command_list_handle_t cmdList = nullptr;
  ZE_CHECKR(zeCommandListCreateImmediate(context, device, &qd, &cmdList),
            "Failed to create immediate command list.");
  return cmdList;
}

// Returns nullptr if not supported by the driver.
ze_pfnCommandListAppendLaunchKernelWithArguments_t
getLaunchWithArgs(ze_driver_handle_t driver) {
  ze_api_version_t version{};
  if (zeDriverGetApiVersion(driver, &version) != ZE_RESULT_SUCCESS ||
      version < ZE_API_VERSION_1_14)
    return nullptr;
  ze_command_list_dditable_t table{};
  if (zeGetCommandListProcAddrTable(ZE_API_VERSION_1_14, &table) !=
      ZE_RESULT_SUCCESS)
    return nullptr;
  return table.pfnAppendLaunchKernelWithArguments;
}

// L0 takes non-const wait lists, but does not modify them.
ze_event_handle_t *waitList(ze_event_handle_t const *deps, uint32_t nDeps) {
  return nDeps ? const_cast<ze_event_handle_t *>(deps) : nullptr;
}

struct EventPool {
  static constexpr uint32_t maxCapacity = 32 * 1024;
  EventPool(ze_device_handle_t device, ze_context_handle_t context) {
    ze_event_pool_desc_t pd{};
    pd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pd.count = maxCapacity;
    ZE_CHECKR(zeEventPoolCreate(context, &pd, 1, &device, &handle),
              "Failed to create event pool.");
  }

  // All events must be destroyed before the pool, including unreleased ones.
  ~EventPool() {
    for (auto *e : events)
      ZE_CHECKL(zeEventDestroy(e), "Failed to destroy event.");
    ZE_CHECKL(zeEventPoolDestroy(handle), "Failed to destroy event pool.");
  }

  EventPool(const EventPool &) = delete;
  EventPool &operator=(const EventPool &) = delete;

  ze_event_handle_t get() {
    if (!released.empty()) return released.pop_back_val();
    if (events.size() >= maxCapacity) gcReportErr("L0 event pool exhausted.");
    ze_event_desc_t ed{};
    ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    ed.index = events.size();
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ze_event_handle_t e = nullptr;
    ZE_CHECKR(zeEventCreate(handle, &ed, &e), "Failed to create event.");
    events.push_back(e);
    return e;
  }

  // Each event must be released once and not used afterwards.
  void release(ze_event_handle_t e) {
    assert(zeEventQueryStatus(e) != ZE_RESULT_NOT_READY);
    assert(!llvm::is_contained(released, e) && "Event released twice.");
    ZE_CHECKR(zeEventHostReset(e), "Failed to reset event.");
    released.push_back(e);
  }

private:
  ze_event_pool_handle_t handle{nullptr};
  llvm::SmallVector<ze_event_handle_t, 0> events;
  llvm::SmallVector<ze_event_handle_t, 0> released;
};
} // namespace

namespace gc::gpu {
struct ZeKernel {
  ZeKernel(ze_module_handle_t module, ze_kernel_handle_t kernel,
           std::vector<uint8_t> argSizes)
      : module(module), kernel(kernel), argSizes(std::move(argSizes)) {}
  ~ZeKernel() {
    ZE_CHECKL(zeKernelDestroy(kernel), "Failed to destroy kernel.");
    ZE_CHECKL(zeModuleDestroy(module), "Failed to destroy module.");
  }
  ZeKernel(const ZeKernel &) = delete;
  ZeKernel &operator=(const ZeKernel &) = delete;

  ze_module_handle_t module;
  ze_kernel_handle_t kernel;
  // Only used if launchWithArgs is not supported.
  std::vector<uint8_t> argSizes;
  std::array<uint32_t, 3> groupSize{};
};

void ZeGpuRuntime::destroyKernel(Kernel *kernel) { delete kernel; }

struct ZeGpuRuntime::Impl {
  Impl(ze_driver_handle_t driver, ze_device_handle_t device,
       ze_context_handle_t context, ze_command_list_handle_t cmdList,
       const DeviceInfo &info, bool destroyContext)
      : device(device), context(context), cmdList(cmdList), info(info),
        destroyContext(destroyContext),
        launchWithArgs(getLaunchWithArgs(driver)),
        eventPool(std::make_unique<EventPool>(device, context)) {}

  ~Impl() {
    ZE_CHECKL(zeCommandListHostSynchronize(cmdList, UINT64_MAX),
              "Failed to synchronize command list.");
    // Members are destroyed after the body, the pool must go before context.
    eventPool.reset();
    ZE_CHECKL(zeCommandListDestroy(cmdList), "Failed to destroy command list.");
    if (destroyContext)
      ZE_CHECKL(zeContextDestroy(context), "Failed to destroy context.");
  }

  Impl(const Impl &) = delete;
  Impl &operator=(const Impl &) = delete;

  std::mutex mux;
  ze_device_handle_t device;
  ze_context_handle_t context;
  ze_command_list_handle_t cmdList;
  DeviceInfo info;
  bool destroyContext;
  // nullptr if not supported by the driver.
  ze_pfnCommandListAppendLaunchKernelWithArguments_t launchWithArgs;
  std::unique_ptr<EventPool> eventPool;
};

ZeGpuRuntime::ZeGpuRuntime(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

ZeGpuRuntime::~ZeGpuRuntime() = default;

const DeviceInfo &ZeGpuRuntime::getDeviceInfo() const { return impl->info; }

ZeGpuRuntime::Kernel *ZeGpuRuntime::createKernel(const uint8_t *data,
                                                 size_t size, const char *name,
                                                 uint32_t nArgs,
                                                 ArgType *argTypes) {
  ze_module_desc_t md{};
  md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  md.format = ZE_MODULE_FORMAT_NATIVE;
  md.inputSize = size;
  md.pInputModule = data;
  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t blog = nullptr;
  auto r = zeModuleCreate(impl->context, impl->device, &md, &module, &blog);
  if (blog) {
    if (r != ZE_RESULT_SUCCESS) {
      size_t n = 0;
      zeModuleBuildLogGetString(blog, &n, nullptr);
      if (n) {
        std::string log(n, '\0');
        zeModuleBuildLogGetString(blog, &n, log.data());
        gcLogE("L0 module build: ", log.c_str());
      }
    }
    ZE_CHECKR(zeModuleBuildLogDestroy(blog), "Failed to destroy build log.");
  }
  ZE_CHECKR(r, "Failed to create module.");

  ze_kernel_desc_t kd{};
  kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kd.pKernelName = name;
  ze_kernel_handle_t kernel = nullptr;
  ZE_CHECKR(zeKernelCreate(module, &kd, &kernel),
            "Failed to create kernel: ", name);
  gcLogD("Created kernel: ", name);
  std::vector<uint8_t> argSizes(impl->launchWithArgs ? 0 : nArgs);
  std::transform(argTypes, argTypes + argSizes.size(), argSizes.begin(),
                 argSize);
  return new Kernel(module, kernel, std::move(argSizes));
}

ZeGpuRuntime::Event *ZeGpuRuntime::launch(Kernel *kernel, void **args,
                                          uint32_t nArgs, uint32_t gridX,
                                          uint32_t gridY, uint32_t gridZ,
                                          uint32_t blockX, uint32_t blockY,
                                          uint32_t blockZ, Event *const *deps,
                                          uint32_t nDeps) {
  ze_group_count_t counts{gridX, gridY, gridZ};
  auto &s = *impl;
  std::lock_guard<std::mutex> lock(s.mux);
  auto *ev = s.eventPool->get();
  if (s.launchWithArgs) {
    ze_group_size_t sizes{blockX, blockY, blockZ};
    ZE_CHECKR(s.launchWithArgs(s.cmdList, kernel->kernel, counts, sizes, args,
                               nullptr, ev, nDeps, waitList(deps, nDeps)),
              "Failed to launch kernel.");
    return ev;
  }

  assert(nArgs == kernel->argSizes.size());
  if (std::array groupSize{blockX, blockY, blockZ};
      groupSize != kernel->groupSize) {
    ZE_CHECKR(zeKernelSetGroupSize(kernel->kernel, blockX, blockY, blockZ),
              "Failed to set kernel group size.");
    kernel->groupSize = groupSize;
  }
  for (auto [i, arg] : llvm::enumerate(llvm::ArrayRef(args, nArgs)))
    ZE_CHECKR(
        zeKernelSetArgumentValue(kernel->kernel, i, kernel->argSizes[i], arg),
        "Failed to set kernel argument.");
  ZE_CHECKR(zeCommandListAppendLaunchKernel(s.cmdList, kernel->kernel, &counts,
                                            ev, nDeps, waitList(deps, nDeps)),
            "Failed to launch kernel.");
  return ev;
}

void *ZeGpuRuntime::allocDevice(size_t size, size_t alignment) {
  gcLogD("Allocating device memory: ", size, " bytes");
  ze_device_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void *ptr = nullptr;
  ZE_CHECKR(zeMemAllocDevice(impl->context, &desc, size ? size : 1, alignment,
                             impl->device, &ptr),
            "Failed to allocate ", size, " bytes of device memory.");
  return ptr;
}

void *ZeGpuRuntime::allocShared(size_t size, size_t alignment) {
  gcLogD("Allocating shared memory: ", size, " bytes");
  ze_device_mem_alloc_desc_t ddesc{};
  ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  ze_host_mem_alloc_desc_t hdesc{};
  hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  void *ptr = nullptr;
  ZE_CHECKR(zeMemAllocShared(impl->context, &ddesc, &hdesc, size ? size : 1,
                             alignment, impl->device, &ptr),
            "Failed to allocate ", size, " bytes of shared memory.");
  return ptr;
}

void *ZeGpuRuntime::allocHost(size_t size, size_t alignment) {
  gcLogD("Allocating host memory: ", size, " bytes");
  ze_host_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  void *ptr = nullptr;
  ZE_CHECKR(
      zeMemAllocHost(impl->context, &desc, size ? size : 1, alignment, &ptr),
      "Failed to allocate ", size, " bytes of host memory.");
  return ptr;
}

void ZeGpuRuntime::free(void *ptr) {
  if (!ptr) return;
  gcLogD("Freeing memory: ", ptr);
  ZE_CHECKR(zeMemFree(impl->context, ptr), "Failed to free memory: ", ptr);
}

ZeGpuRuntime::Event *ZeGpuRuntime::memcpy(const void *src, void *dst,
                                          size_t size, Event *const *deps,
                                          uint32_t nDeps) {
  gcLogD("Copying memory: ", size, " bytes from ", src, " to ", dst);
  auto &s = *impl;
  std::lock_guard<std::mutex> lock(s.mux);
  auto *ev = s.eventPool->get();
  ZE_CHECKR(zeCommandListAppendMemoryCopy(s.cmdList, dst, src, size, ev, nDeps,
                                          waitList(deps, nDeps)),
            "Failed to copy ", size, " bytes from ", src, " to ", dst);
  return ev;
}

void ZeGpuRuntime::wait(Event *const *events, uint32_t nEvents) {
  llvm::ArrayRef evs(events, nEvents);
  for (auto *e : evs)
    ZE_CHECKR(zeEventHostSynchronize(e, UINT64_MAX),
              "Failed to wait for event.");
  std::lock_guard<std::mutex> lock(impl->mux);
  for (auto *e : evs) impl->eventPool->release(e);
}

std::shared_ptr<ZeGpuRuntime> ZeGpuRuntime::create() {
  auto r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (r != ZE_RESULT_SUCCESS) r = zeInit(0);
  if (r != ZE_RESULT_SUCCESS) {
    gcLogE("L0 zeInit failed: ", r);
    return nullptr;
  }

  ze_driver_handle_t driver;
  ze_device_handle_t device;
  DeviceInfo info;
  if (!selectDevice(driver, device, info)) {
    gcLogE("No suitable Intel GPU devices found.");
    return nullptr;
  }
  ze_context_desc_t cd{};
  cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context = nullptr;
  ZE_CHECKR(zeContextCreate(driver, &cd, &context),
            "Failed to create L0 context.");
  return std::shared_ptr<ZeGpuRuntime>(new ZeGpuRuntime(std::make_unique<Impl>(
      driver, device, context, createCmdList(context, device), info, true)));
}

std::shared_ptr<ZeGpuRuntime>
ZeGpuRuntime::create(ze_device_handle_t device, ze_driver_handle_t driver,
                     ze_context_handle_t context) {
  DeviceInfo info;
  if (!device || !devInfo(device, info)) {
    gcLogE("Unsupported device.");
    return nullptr;
  }
  return std::shared_ptr<ZeGpuRuntime>(new ZeGpuRuntime(std::make_unique<Impl>(
      driver, device, context, createCmdList(context, device), info, false)));
}

ZeGpuRuntime *ZeGpuRuntime::get() {
  // Leaked on purpose: destruction during static teardown (including exit from
  // report_fatal_error) races live users and the L0 loader unload.
  static auto *instance = new auto(create());
  return instance->get();
}
} // namespace gc::gpu
