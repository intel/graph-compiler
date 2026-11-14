// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "gc/Runtime/GpuRuntime.h"

namespace gc::gpu {
struct OclKernel;

class OclRuntime final : public GpuRuntime<OclKernel, _cl_event, _cl_mem> {
public:
  static std::shared_ptr<OclRuntime> create(cl_command_queue queue);
  ~OclRuntime() override;

  const DeviceInfo &getDeviceInfo() const override;
  // Retains an external buffer for use in compiled programs.
  Memory *importBuffer(cl_mem buffer);
  // Source objects start with "// OpenCL C\n". SPIR-V uses its magic word.
  Kernel *createKernel(const uint8_t *data, size_t size, const char *name,
                       uint32_t nArgs, ArgType *argTypes) override;
  static void destroyKernel(Kernel *kernel);
  Event *launch(Kernel *kernel, void **args, uint32_t nArgs, uint32_t gridX,
                uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                uint32_t blockY, uint32_t blockZ, Event *const *deps,
                uint32_t nDeps) override;
  Memory *allocDevice(size_t size, size_t alignment) override;
  Memory *allocShared(size_t size, size_t alignment) override;
  Memory *allocHost(size_t size, size_t alignment) override;
  void free(Memory *ptr) override;
  Event *memcpy(const Memory *src, Memory *dst, size_t size, Event *const *deps,
                uint32_t nDeps) override;
  void wait(Event *const *events, uint32_t nEvents) override;

private:
  struct Impl;
  explicit OclRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};
} // namespace gc::gpu
