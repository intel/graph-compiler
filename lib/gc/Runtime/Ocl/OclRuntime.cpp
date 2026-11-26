// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/Runtime/OclRuntime.h"
#include "gc/Utils/Error.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemAlloc.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
void check(cl_int status) {
  if (status != CL_SUCCESS) gcReportErr("OpenCL error: ", status);
}
} // namespace

namespace gc::gpu {
struct OclKernel {
  cl_program program;
  cl_kernel kernel;
  std::vector<ArgType> types;
  std::mutex mutex;
  ~OclKernel() {
    check(clReleaseKernel(kernel));
    check(clReleaseProgram(program));
  }
};

struct OclRuntime::Impl {
  cl_command_queue queue;
  cl_context context;
  cl_device_id device;
  DeviceInfo info;
  std::mutex mutex;
  std::unordered_set<cl_mem> buffers;
  std::unordered_map<void *, std::pair<size_t, size_t>> hosts;

  ~Impl() {
    check(clFinish(queue));
    for (auto buffer : buffers) check(clReleaseMemObject(buffer));
    for (auto [ptr, allocation] : hosts)
      llvm::deallocate_buffer(ptr, allocation.first, allocation.second);
    check(clReleaseCommandQueue(queue));
  }
};

OclRuntime::OclRuntime(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}
OclRuntime::~OclRuntime() = default;

std::shared_ptr<OclRuntime> OclRuntime::create(cl_command_queue queue) {
  auto impl = std::make_unique<Impl>();
  check(clRetainCommandQueue(queue));
  impl->queue = queue;
  check(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(impl->context),
                              &impl->context, nullptr));
  check(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(impl->device),
                              &impl->device, nullptr));
  size_t size = 0;
  check(clGetDeviceInfo(impl->device, CL_DEVICE_NAME, 0, nullptr, &size));
  std::vector<char> name(size);
  check(clGetDeviceInfo(impl->device, CL_DEVICE_NAME, size, name.data(),
                        nullptr));
  impl->info.name = name.data();
  size_t maxGroup = 0;
  check(clGetDeviceInfo(impl->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(maxGroup), &maxGroup, nullptr));
  impl->info.maxWgSize = maxGroup;
  return std::shared_ptr<OclRuntime>(new OclRuntime(std::move(impl)));
}

const DeviceInfo &OclRuntime::getDeviceInfo() const { return impl->info; }

OclRuntime::Memory *OclRuntime::importBuffer(cl_mem buffer) {
  cl_context context;
  check(clGetMemObjectInfo(buffer, CL_MEM_CONTEXT, sizeof(context), &context,
                           nullptr));
  if (context != impl->context) gcReportErr("OpenCL buffer context mismatch");
  std::lock_guard<std::mutex> lock(impl->mutex);
  if (impl->buffers.insert(buffer).second) check(clRetainMemObject(buffer));
  return buffer;
}

OclRuntime::Kernel *OclRuntime::createKernel(const uint8_t *data, size_t size,
                                             const char *name, uint32_t nArgs,
                                             ArgType *argTypes) {
  cl_int status = CL_SUCCESS;
  cl_program program;
  uint32_t magic = 0;
  if (size >= sizeof(magic)) std::memcpy(&magic, data, sizeof(magic));
  bool source = llvm::StringRef(reinterpret_cast<const char *>(data), size)
                    .starts_with("// OpenCL C\n");
  if (source) {
    auto text = reinterpret_cast<const char *>(data);
    program =
        clCreateProgramWithSource(impl->context, 1, &text, &size, &status);
  } else if (magic == 0x07230203) {
#ifdef CL_VERSION_2_1
    program = clCreateProgramWithIL(impl->context, data, size, &status);
#else
    cl_platform_id platform;
    check(clGetDeviceInfo(impl->device, CL_DEVICE_PLATFORM, sizeof(platform),
                          &platform, nullptr));
    using CreateIL =
        cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
    auto create =
        reinterpret_cast<CreateIL>(clGetExtensionFunctionAddressForPlatform(
            platform, "clCreateProgramWithILKHR"));
    if (!create) gcReportErr("OpenCL device does not support SPIR-V");
    program = create(impl->context, data, size, &status);
#endif
  } else {
    cl_int binaryStatus;
    program = clCreateProgramWithBinary(impl->context, 1, &impl->device, &size,
                                        &data, &binaryStatus, &status);
    check(status);
    check(binaryStatus);
  }
  check(status);
  status = clBuildProgram(program, 1, &impl->device,
                          source ? "-cl-std=CL1.2" : nullptr, nullptr, nullptr);
  if (status != CL_SUCCESS) {
    size_t length = 0;
    check(clGetProgramBuildInfo(program, impl->device, CL_PROGRAM_BUILD_LOG, 0,
                                nullptr, &length));
    std::vector<char> log(length + 1);
    check(clGetProgramBuildInfo(program, impl->device, CL_PROGRAM_BUILD_LOG,
                                length, log.data(), nullptr));
    gcReportErr("OpenCL program build failed: ", log.data());
  }
  auto kernel = clCreateKernel(program, name, &status);
  check(status);
  std::vector<ArgType> types;
  if (nArgs) types.assign(argTypes, argTypes + nArgs);
  return new Kernel{program, kernel, std::move(types), {}};
}

void OclRuntime::destroyKernel(Kernel *kernel) { delete kernel; }

OclRuntime::Event *OclRuntime::launch(Kernel *kernel, void **args,
                                      uint32_t nArgs, uint32_t gridX,
                                      uint32_t gridY, uint32_t gridZ,
                                      uint32_t blockX, uint32_t blockY,
                                      uint32_t blockZ, Event *const *deps,
                                      uint32_t nDeps) {
  if (!nDeps) deps = nullptr;
  if (nArgs != kernel->types.size())
    gcReportErr("OpenCL argument count mismatch");
  std::lock_guard<std::mutex> lock(kernel->mutex);
  for (uint32_t i = 0; i < nArgs; ++i)
    check(
        clSetKernelArg(kernel->kernel, i, argSize(kernel->types[i]), args[i]));
  size_t groups[] = {gridX, gridY, gridZ};
  size_t local[] = {blockX, blockY, blockZ};
  size_t global[] = {groups[0] * local[0], groups[1] * local[1],
                     groups[2] * local[2]};
  cl_event event;
  if (!global[0] || !global[1] || !global[2])
    check(clEnqueueMarkerWithWaitList(impl->queue, nDeps, deps, &event));
  else
    check(clEnqueueNDRangeKernel(impl->queue, kernel->kernel, 3, nullptr,
                                 global, local, nDeps, deps, &event));
  return event;
}

OclRuntime::Memory *OclRuntime::allocDevice(size_t size, size_t) {
  cl_int status;
  auto buffer = clCreateBuffer(impl->context, CL_MEM_READ_WRITE,
                               std::max<size_t>(size, 1), nullptr, &status);
  check(status);
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->buffers.insert(buffer);
  return buffer;
}

OclRuntime::Memory *OclRuntime::allocShared(size_t size, size_t alignment) {
  return allocDevice(size, alignment);
}

OclRuntime::Memory *OclRuntime::allocHost(size_t size, size_t alignment) {
  size = std::max<size_t>(size, 1);
  if (!alignment) alignment = alignof(std::max_align_t);
  auto ptr = llvm::allocate_buffer(size, alignment);
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->hosts.emplace(ptr, std::make_pair(size, alignment));
  return static_cast<Memory *>(ptr);
}

void OclRuntime::free(Memory *ptr) {
  if (!ptr) return;
  std::lock_guard<std::mutex> lock(impl->mutex);
  if (impl->buffers.erase(ptr)) check(clReleaseMemObject(ptr));
  else if (auto it = impl->hosts.find(ptr); it != impl->hosts.end()) {
    llvm::deallocate_buffer(ptr, it->second.first, it->second.second);
    impl->hosts.erase(it);
  } else gcReportErr("Unknown OpenCL allocation");
}

OclRuntime::Event *OclRuntime::memcpy(const Memory *src, Memory *dst,
                                      size_t size, Event *const *deps,
                                      uint32_t nDeps) {
  if (!nDeps) deps = nullptr;
  std::lock_guard<std::mutex> lock(impl->mutex);
  auto source = const_cast<Memory *>(src);
  bool srcBuffer = impl->buffers.count(source);
  bool dstBuffer = impl->buffers.count(dst);
  cl_event event;
  if (!size)
    check(clEnqueueMarkerWithWaitList(impl->queue, nDeps, deps, &event));
  else if (srcBuffer && dstBuffer)
    check(clEnqueueCopyBuffer(impl->queue, source, dst, 0, 0, size, nDeps, deps,
                              &event));
  else if (dstBuffer)
    check(clEnqueueWriteBuffer(impl->queue, dst, CL_FALSE, 0, size, src, nDeps,
                               deps, &event));
  else if (srcBuffer)
    check(clEnqueueReadBuffer(impl->queue, source, CL_FALSE, 0, size, dst,
                              nDeps, deps, &event));
  else {
    if (nDeps) check(clWaitForEvents(nDeps, deps));
    std::memcpy(dst, src, size);
    check(clEnqueueMarkerWithWaitList(impl->queue, 0, nullptr, &event));
  }
  return event;
}

void OclRuntime::wait(Event *const *events, uint32_t nEvents) {
  if (nEvents) check(clWaitForEvents(nEvents, events));
  for (uint32_t i = 0; i < nEvents; ++i) check(clReleaseEvent(events[i]));
}
} // namespace gc::gpu
