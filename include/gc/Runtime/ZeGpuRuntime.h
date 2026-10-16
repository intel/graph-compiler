//===-- ZeGpuRuntime.h - Level Zero GpuRuntime ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "gc/Runtime/GpuRuntime.h"

// Level Zero handles, identical to the ze_api.h declarations.
struct _ze_event_handle_t;
typedef struct _ze_driver_handle_t *ze_driver_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;

namespace gc::gpu {
struct ZeKernel;

class ZeGpuRuntime final
    : public GpuRuntime<ZeKernel, _ze_event_handle_t, void> {
public:
  // Returns a platform-wide shared instance of the runtime or nullptr
  // if no suitable device found.
  static ZeGpuRuntime *get();
  static std::shared_ptr<ZeGpuRuntime> create();
  static std::shared_ptr<ZeGpuRuntime> create(ze_device_handle_t device,
                                              ze_driver_handle_t driver,
                                              ze_context_handle_t context);
  ~ZeGpuRuntime() override;

  const DeviceInfo &getDeviceInfo() const override;
  Kernel *createKernel(const uint8_t *data, size_t size, const char *name,
                       uint32_t nArgs, ArgType *argTypes) override;
  static void destroyKernel(Kernel *kernel);
  Event *launch(Kernel *kernel, void **args, uint32_t nArgs, uint32_t gridX,
                uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                uint32_t blockY, uint32_t blockZ, Event *const *deps,
                uint32_t nDeps) override;
  void *allocDevice(size_t size, size_t alignment) override;
  void *allocShared(size_t size, size_t alignment) override;
  void *allocHost(size_t size, size_t alignment) override;
  void free(void *ptr) override;
  Event *memcpy(const void *src, void *dst, size_t size, Event *const *deps,
                uint32_t nDeps) override;
  void wait(Event *const *events, uint32_t nEvents) override;

private:
  struct Impl;

  explicit ZeGpuRuntime(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl;
};
} // namespace gc::gpu
