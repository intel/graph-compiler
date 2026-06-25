//===-- GpuOclRuntime.cpp - GPU OpenCL Runtime ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <variant>

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include <CL/cl_ext.h>

#include "gc/Transforms/Passes.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"
#include "gc/Utils/Transform.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::gc::gpu {

#define makeClErrPref(code) "OpenCL error ", code, ": "
#define makeClErr(code, ...) gcMakeErr(makeClErrPref(code), __VA_ARGS__)
#define reportClErr(code, ...) gcReportErr(makeClErrPref(code), __VA_ARGS__)

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) return gcMakeErr(__VA_ARGS__);                                \
  } while (0)
#define CHECKE(expected, ...)                                                  \
  do {                                                                         \
    if (!expected) {                                                           \
      gcLogE(__VA_ARGS__);                                                     \
      return expected.takeError();                                             \
    }                                                                          \
  } while (0)
#define CL_CHECK(expr, ...)                                                    \
  do {                                                                         \
    if (auto _cl_check_err = (expr); _cl_check_err != CL_SUCCESS)              \
      return makeClErr(_cl_check_err, __VA_ARGS__);                            \
  } while (0)
#define CL_CHECKR(expr, ...)                                                   \
  do {                                                                         \
    if (auto _cl_check_err = (expr); _cl_check_err != CL_SUCCESS) {            \
      reportClErr(_cl_check_err, __VA_ARGS__);                                 \
    }                                                                          \
  } while (0)

#define clGetDevInfo(type, dev, prop) clGetDeviceInfo<type>(dev, prop, #prop)
template <typename T>
auto clGetDeviceInfo(cl_device_id dev, cl_device_info prop, const char *name) {
  if constexpr (std::is_integral_v<T>) {
    T v;
    CL_CHECKR(clGetDeviceInfo(dev, prop, sizeof(T), &v, nullptr),
              "Failed to get the device property ", name);
    gcLogD("Device property ", name, "=", v);
    return v;
  } else if constexpr (std::is_convertible_v<T, StringRef>) {
    size_t v;
    CL_CHECKR(clGetDeviceInfo(dev, prop, 0, nullptr, &v),
              "Failed to get the size of device property ", name);
    std::string value(v, '\0');
    CL_CHECKR(clGetDeviceInfo(dev, prop, v, value.data(), nullptr),
              "Failed to get the device property ", name);
    value.erase(value.find_last_not_of('\0') + 1);
    gcLogD("Device property ", name, "=", value);
    return value;
  } else if constexpr (std::is_convertible_v<
                           T, ArrayRef<typename T::value_type>>) {
    size_t v;
    CL_CHECKR(clGetDeviceInfo(dev, prop, 0, nullptr, &v),
              "Failed to get the size of device property ", name);
    T values(v / sizeof(typename T::value_type));
    CL_CHECKR(clGetDeviceInfo(dev, prop, v, values.data(), nullptr),
              "Failed to get the device property ", name);
    gcLogD("Device property ", name, "=",
           llvm::formatv("{0:$[,]}",
                         llvm::make_range(values.begin(), values.end()))
               .str());
    return values;
  }
}

// cl_ext function pointers
struct OclRuntime::Ext : OclDevCtxPair {
  clDeviceMemAllocINTEL_fn clDeviceMemAllocINTEL;
  clSharedMemAllocINTEL_fn clSharedMemAllocINTEL;
  clMemFreeINTEL_fn clMemFreeINTEL;
  clEnqueueMemcpyINTEL_fn clEnqueueMemcpyINTEL;
  clGetMemAllocInfoINTEL_fn clGetMemAllocInfoINTEL;
  clSetKernelArgMemPointerINTEL_fn clSetKernelArgMemPointerINTEL;

  explicit Ext(cl_device_id device, cl_context context,
               clDeviceMemAllocINTEL_fn clDeviceMemAllocINTEL,
               clSharedMemAllocINTEL_fn clSharedMemAllocINTEL,
               clMemFreeINTEL_fn clMemFreeINTEL,
               clEnqueueMemcpyINTEL_fn clEnqueueMemcpyINTEL,
               clGetMemAllocInfoINTEL_fn clGetMemAllocInfoINTEL,
               clSetKernelArgMemPointerINTEL_fn clSetKernelArgMemPointerINTEL)
      : OclDevCtxPair(device, context),
        clDeviceMemAllocINTEL(clDeviceMemAllocINTEL),
        clSharedMemAllocINTEL(clSharedMemAllocINTEL),
        clMemFreeINTEL(clMemFreeINTEL),
        clEnqueueMemcpyINTEL(clEnqueueMemcpyINTEL),
        clGetMemAllocInfoINTEL(clGetMemAllocInfoINTEL),
        clSetKernelArgMemPointerINTEL(clSetKernelArgMemPointerINTEL) {}

  static llvm::Expected<const Ext *> get(cl_device_id device,
                                         cl_context context) {
    static std::shared_mutex mux;
    static std::unordered_map<const OclDevCtxPair, const Ext *> cache;

    OclDevCtxPair pair{device, context};
    {
      std::shared_lock<std::shared_mutex> lock(mux);
      if (auto it = cache.find(pair); it != cache.end()) {
        return it->second;
      }
    }

    cl_platform_id platform;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                             &platform, nullptr),
             "Failed to get the device platform.");

#define FIND_FUNC(name)                                                        \
  auto name = reinterpret_cast<name##_fn>(                                     \
      clGetExtensionFunctionAddressForPlatform(platform, #name));              \
  CHECK(name, "Failed to get the " #name " function address.")

    FIND_FUNC(clDeviceMemAllocINTEL);
    FIND_FUNC(clSharedMemAllocINTEL);
    FIND_FUNC(clMemFreeINTEL);
    FIND_FUNC(clEnqueueMemcpyINTEL);
    FIND_FUNC(clGetMemAllocInfoINTEL);
    FIND_FUNC(clSetKernelArgMemPointerINTEL);

    std::lock_guard<std::shared_mutex> lock(mux);
    if (auto it = cache.find(pair); it != cache.end()) {
      return it->second;
    }
    auto ext =
        new Ext(device, context, clDeviceMemAllocINTEL, clSharedMemAllocINTEL,
                clMemFreeINTEL, clEnqueueMemcpyINTEL, clGetMemAllocInfoINTEL,
                clSetKernelArgMemPointerINTEL);
    cache.emplace(pair, ext);
    return ext;
  }
};

struct Kernel {
  cl_program program;
  cl_kernel kernel;
  const size_t localSize[3];
  const SmallVector<size_t> argSize;

  explicit Kernel(cl_program program, cl_kernel kernel, const size_t *blockSize,
                  size_t argNum, const size_t *argSize)
      : program(program), kernel(kernel),
        localSize{blockSize[0], blockSize[1], blockSize[2]},
        argSize(argSize, argSize + argNum) {
#ifndef NDEBUG
    std::string args;
    for (size_t i = 0; i < argNum; i++) {
      args += std::to_string(argSize[i]);
      if (i < argNum - 1) {
        args += ", ";
      }
    }
    gcLogD("Kernel ", kernel, " params: localSize=[", localSize[0], ", ",
           localSize[1], ", ", localSize[2], "], argSize=[", args.c_str(), "]");
#endif
  }

  ~Kernel() {
    if (kernel != nullptr) {
      CL_CHECKR(clReleaseKernel(kernel), "Failed to release OpenCL kernel.");
      gcLogD("Released OpenCL kernel: ", kernel);
      CL_CHECKR(clReleaseProgram(program), "Failed to release OpenCL program.");
      gcLogD("Released OpenCL program: ", program);
    }
  }
};

// Functions exported to the ExecutionEngine
struct OclRuntime::Exports {
  static llvm::orc::SymbolMap symbolMap(llvm::orc::MangleAndInterner interner) {
    return llvm::orc::SymbolMap{
        {interner(GPU_OCL_MALLOC_DEV),
         {llvm::orc::ExecutorAddr::fromPtr(&allocDev),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_MALLOC_SHARED),
         {llvm::orc::ExecutorAddr::fromPtr(&allocShared),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_DEALLOC),
         {llvm::orc::ExecutorAddr::fromPtr(&dealloc),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_MEMCPY),
         {llvm::orc::ExecutorAddr::fromPtr(&memcpy),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_KERNEL_CREATE),
         {llvm::orc::ExecutorAddr::fromPtr(&kernelCreate),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_KERNEL_DESTROY),
         {llvm::orc::ExecutorAddr::fromPtr(&kernelDestroy),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_KERNEL_LAUNCH),
         {llvm::orc::ExecutorAddr::fromPtr(&kernelLaunch),
          llvm::JITSymbolFlags::Exported}},
        {interner(GPU_OCL_FINISH),
         {llvm::orc::ExecutorAddr::fromPtr(&finish),
          llvm::JITSymbolFlags::Exported}},
        // Stub mgpu functions (unused but required by LLVM GPU lowering)
        {interner("mgpuModuleLoad"),
         {llvm::orc::ExecutorAddr::fromPtr(&mgpuModuleLoadStub),
          llvm::JITSymbolFlags::Exported}},
        {interner("mgpuModuleLoadJIT"),
         {llvm::orc::ExecutorAddr::fromPtr(&mgpuModuleLoadStub),
          llvm::JITSymbolFlags::Exported}},
        {interner("mgpuModuleUnload"),
         {llvm::orc::ExecutorAddr::fromPtr(&mgpuModuleUnloadStub),
          llvm::JITSymbolFlags::Exported}}};
  }

private:
  // Stubs for mgpu functions (not used in OpenCL runtime)
  static void *mgpuModuleLoadStub(const void *, size_t) { return nullptr; }
  static void mgpuModuleUnloadStub(void *) {}

  static void *allocDev(const OclContext *ctx, size_t size) {
    return gcGetOrReport(ctx->runtime.usmAllocDev(size));
  }

  static void *allocShared(const OclContext *ctx, size_t size) {
    return gcGetOrReport(ctx->runtime.usmAllocShared(size));
  }

  static void dealloc(const OclContext *ctx, const void *ptr) {
    gcGetOrReport(ctx->runtime.usmFree(ptr));
  }

  static void memcpy(OclContext *ctx, const void *src, void *dst, size_t size) {
    gcGetOrReport(ctx->runtime.usmCpy(*ctx, src, dst, size));
  }

  static Kernel *kernelCreate(const OclContext *ctx, size_t spirvLen,
                              const unsigned char *spirv, const char *name,
                              const size_t *blockSize, size_t argNum,
                              const size_t *argSize) {
    cl_int err;
    auto program =
        clCreateProgramWithIL(ctx->runtime.ext.context, spirv, spirvLen, &err);
    CL_CHECKR(err, "Failed to create OpenCL program with IL.");

    gcLogD("Created new OpenCL program: ", program);
    clBuildProgram(program, 1, &ctx->runtime.ext.device, nullptr, nullptr,
                   nullptr);
    CL_CHECKR(err, "Failed to build the program: ", program);
    gcLogD("The program has been built: ", program);

    auto kernel = clCreateKernel(program, name, &err);
    if (err != CL_SUCCESS) {
      clReleaseProgram(program);
      CL_CHECKR(err, "Failed to create OpenCL kernel from program: ", program);
    }
    gcLogD("Created new OpenCL kernel ", kernel, " from program ", program);

    cl_bool enable = CL_TRUE;
    err = clSetKernelExecInfo(kernel,
                              CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                              sizeof(enable), &enable);
    CL_CHECKR(err, "Failed to set indirect host access.");
    err = clSetKernelExecInfo(kernel,
                              CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                              sizeof(enable), &enable);
    CL_CHECKR(err, "Failed to set indirect device access.");
    err = clSetKernelExecInfo(kernel,
                              CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                              sizeof(enable), &enable);
    CL_CHECKR(err, "Failed to set indirect shared access.");

    return new Kernel(program, kernel, blockSize, argNum, argSize);
  }

  static void kernelDestroy(size_t count, Kernel **kernels) {
    gcLogD("Destroying kernels.");
    for (size_t i = 0; i < count; i++) {
      if (kernels[i]) {
        delete kernels[i];
      }
    }
  }

  static void kernelLaunch(OclContext *ctx, Kernel *kernel, size_t gridX,
                           size_t gridY, size_t gridZ, ...) {
    struct ClonedKernel {
      cl_kernel kernel;

      explicit ClonedKernel(cl_kernel kernel) : kernel(kernel) {}

      ~ClonedKernel() {
        gcLogD("Releasing cloned OpenCL kernel: ", kernel);
        CL_CHECKR(clReleaseKernel(kernel),
                  "Failed to release the kernel: ", kernel);
      }
    };

    const size_t globalSize[3] = {gridX * kernel->localSize[0],
                                  gridY * kernel->localSize[1],
                                  gridZ * kernel->localSize[2]};

    va_list args;
    va_start(args, gridZ);
    gcLogD("Launching kernel: ", kernel->kernel);

    cl_int err;
    ClonedKernel cloned{clCloneKernel(kernel->kernel, &err)};
    CL_CHECKR(err, "Failed to clone OpenCL kernel: ", kernel->kernel);
    gcLogD("Cloned OpenCL kernel ", kernel->kernel, ": ", cloned.kernel);

    for (size_t i = 0, n = kernel->argSize.size(); i < n; i++) {
      auto size = kernel->argSize[i];
      void *ptr = va_arg(args, void *);

      if (size) {
        gcLogD("Setting kernel ", cloned.kernel, " argument ", i, " to ",
               *static_cast<int64_t *>(ptr));
        err = clSetKernelArg(cloned.kernel, i, size, ptr);
      } else if (ctx->clPtrs->find(ptr) == ctx->clPtrs->end()) {
        gcLogD("Setting kernel ", cloned.kernel, " argument ", i,
               " to USM pointer ", ptr);
        err = ctx->runtime.ext.clSetKernelArgMemPointerINTEL(cloned.kernel, i,
                                                             ptr);
      } else {
        gcLogD("Setting kernel ", cloned.kernel, " argument ", i,
               " to CL pointer ", ptr);
        err = clSetKernelArg(cloned.kernel, i, sizeof(cl_mem), &ptr);
      }

      CL_CHECKR(err, "Failed to set kernel ", cloned.kernel, " argument ", i,
                " of size ", size);
    }
    va_end(args);

    if (ctx->createEvents) {
      cl_event event = nullptr;
      err = clEnqueueNDRangeKernel(ctx->queue, cloned.kernel, 3, nullptr,
                                   globalSize, kernel->localSize,
                                   ctx->waitListLen, ctx->waitList, &event);
      ctx->setLastEvent(event);
    } else {
      err = clEnqueueNDRangeKernel(ctx->queue, cloned.kernel, 3, nullptr,
                                   globalSize, kernel->localSize, 0, nullptr,
                                   nullptr);
    }

    if (err == CL_INVALID_WORK_GROUP_SIZE) {
      size_t wgSize, compileWgSize[3];
      CL_CHECKR(
          clGetKernelWorkGroupInfo(cloned.kernel, ctx->runtime.ext.device,
                                   CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                   &wgSize, nullptr),
          "Failed to get kernel work group size for kernel: ", cloned.kernel);
      CL_CHECKR(
          clGetKernelWorkGroupInfo(cloned.kernel, ctx->runtime.ext.device,
                                   CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                   sizeof(compileWgSize), &compileWgSize,
                                   nullptr),
          "Failed to get compile work group size for kernel: ", cloned.kernel);

      size_t requestedSize =
          kernel->localSize[0] * kernel->localSize[1] * kernel->localSize[2];

      gcReportErr("Invalid work group size for kernel: ", cloned.kernel,
                  "\n  Requested: [", kernel->localSize[0], ", ",
                  kernel->localSize[1], ", ", kernel->localSize[2],
                  "] (total=", requestedSize, ")", "\n  Max allowed: ", wgSize,
                  "\n  Compile-time: [", compileWgSize[0], ", ",
                  compileWgSize[1], ", ", compileWgSize[2], "]");
    }

    CL_CHECKR(err, "Failed to enqueue kernel execution: ", cloned.kernel);
    gcLogD("Enqueued kernel execution: ", cloned.kernel);
  }

  static void finish(OclContext *ctx) { gcGetOrReport(ctx->finish()); }
};

OclRuntime::OclRuntime(const Ext &ext) : ext(ext) {}

llvm::Expected<SmallVector<cl_device_id, 2>>
OclRuntime::gcIntelDevices(size_t max) {
  SmallVector<cl_device_id, 2> intelDevices;
  if (max == 0) {
    return intelDevices;
  }

  cl_uint numPlatforms;
  CL_CHECK(clGetPlatformIDs(0, nullptr, &numPlatforms),
           "Failed to get the number of platforms.");

  if (numPlatforms == 0) {
    gcLogD("No platforms found.");
    return intelDevices;
  }

  SmallVector<cl_platform_id> platforms(numPlatforms);
  auto err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    gcLogE("Failed to get the platform ids. Error: ", err);
    return intelDevices;
  }

  for (auto platform : platforms) {
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
      gcLogD("Failed to get the number of devices on the platform ", platform,
             ". Error: ", err);
      continue;
    }
    if (numDevices == 0) {
      continue;
    }

    SmallVector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices,
                         devices.data(), nullptr);
    if (err != CL_SUCCESS) {
      gcLogE("Failed to get the device ids on the platform ", platform,
             ". Error: ", err);
      continue;
    }

    for (auto dev : devices) {
      cl_uint vendorId;
      err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(cl_uint),
                            &vendorId, nullptr);
      if (err != CL_SUCCESS) {
        gcLogE("Failed to get info about the device ", dev, ". Error: ", err);
        continue;
      }
      if (vendorId == 0x8086) {
        intelDevices.emplace_back(dev);
#ifndef NDEBUG
        size_t nameSize;
        std::string name;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &nameSize);
        name.resize(nameSize);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, nameSize, &name[0], nullptr);
        gcLogD("[ INFO ] GPU device ", name.c_str(), " id: ", dev);
#endif
        if (intelDevices.size() == max) {
          return intelDevices;
        }
      }
    }
  }

  return intelDevices;
}

llvm::Expected<OclRuntime> OclRuntime::get() {
  static OclRuntime *defaultRuntimePtr = nullptr;
  if (OclRuntime *rt = defaultRuntimePtr) {
    return *rt;
  }

  auto devices = gcIntelDevices(1);
  CHECKE(devices, "Failed to get Intel GPU devices.");
  if (devices->empty()) {
    return gcMakeErr("No Intel GPU devices found.");
  }

  auto rt = get(devices.get()[0]);
  CHECKE(rt, "Failed to create OclRuntime.");

  static OclRuntime defaultRuntime = rt.get();
  defaultRuntimePtr = &defaultRuntime;
  return defaultRuntime;
}

llvm::Expected<OclRuntime> OclRuntime::get(cl_device_id device) {
  static std::shared_mutex mux;
  static std::unordered_map<cl_device_id, cl_context> cache;
  cl_context context = nullptr;

  {
    std::shared_lock<std::shared_mutex> lock(mux);
    if (auto it = cache.find(device); it != cache.end()) {
      context = it->second;
    }
  }

  if (context) {
    return get(device, context);
  }

  cl_int err;
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  CL_CHECK(err, "Failed to create OpenCL context.");
  gcLogD("Created new OpenCL context: ", context);

  {
    std::lock_guard<std::shared_mutex> lock(mux);
    if (auto it = cache.find(device); it != cache.end()) {
      if (clReleaseContext(context) != CL_SUCCESS) {
        gcLogE("Failed to release OpenCL context: ", context);
      } else {
        gcLogD("Released OpenCL context: ", context);
      }
      context = it->second;
    } else {
      cache.emplace(device, context);
    }
  }

  return get(device, context);
}

llvm::Expected<OclRuntime> OclRuntime::get(cl_command_queue queue) {
  cl_device_id device;
  cl_context context;
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id),
                                 &device, nullptr),
           "Failed to get CL_QUEUE_DEVICE.");
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context),
                                 &context, nullptr),
           "Failed to get CL_QUEUE_CONTEXT.");
  assert(device);
  assert(context);
  return get(device, context);
}

llvm::Expected<OclRuntime> OclRuntime::get(cl_device_id device,
                                           cl_context context) {
  auto ext = Ext::get(device, context);
  CHECKE(ext, "Failed to create OclRuntime::Ext.");
  return OclRuntime{*ext.get()};
}

bool OclRuntime::isOutOfOrder(cl_command_queue queue) {
  cl_command_queue_properties properties;
  cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                                     sizeof(cl_command_queue_properties),
                                     &properties, nullptr);
  if (err != CL_SUCCESS) {
    gcLogE("clGetCommandQueueInfo() failed with error code ", err);
    // Enforcing out-of-order execution mode
    return true;
  }
  return properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}

cl_context OclRuntime::getContext() const { return ext.context; }

cl_device_id OclRuntime::getDevice() const { return ext.device; }

llvm::Expected<cl_command_queue>
OclRuntime::createQueue(bool outOfOrder) const {
  cl_int err;
  cl_command_queue queue;
#ifdef CL_VERSION_2_0
  cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES,
      static_cast<cl_queue_properties>(
          outOfOrder ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0),
      0};
  queue = clCreateCommandQueueWithProperties(ext.context, ext.device,
                                             properties, &err);
#else
  const cl_command_queue_properties properties =
      outOfOrder ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
  queue = clCreateCommandQueue(context, device, properties, &err);
#endif
  CL_CHECK(err, "Failed to create ", outOfOrder ? "out-of-order " : "",
           "OpenCL command queue.");
  gcLogD("Created new ", outOfOrder ? "out-of-order " : "",
         "OpenCL command queue: ", queue);
  return queue;
}

llvm::Expected<bool> OclRuntime::releaseQueue(cl_command_queue queue) {
  CL_CHECK(clReleaseCommandQueue(queue),
           "Failed to release OpenCL command queue: ", queue);
  gcLogD("Released OpenCL command queue: ", queue);
  return true;
}

llvm::Expected<void *> OclRuntime::usmAllocDev(size_t size) const {
  cl_int err;
  void *ptr = ext.clDeviceMemAllocINTEL(ext.context, ext.device, nullptr, size,
                                        0, &err);
  CL_CHECK(err, "Failed to allocate ", size, " bytes of device USM memory.");
  gcLogD("Allocated ", size, " bytes of device USM memory: ", ptr);
  return ptr;
}

llvm::Expected<void *> OclRuntime::usmAllocShared(size_t size) const {
  cl_int err;
  void *ptr = ext.clSharedMemAllocINTEL(ext.context, ext.device, nullptr, size,
                                        0, &err);
  CL_CHECK(err, "Failed to allocate ", size, " bytes of shared USM memory.");
  gcLogD("Allocated ", size, " bytes of shared USM memory: ", ptr);
  return ptr;
}

llvm::Expected<bool> OclRuntime::usmFree(const void *ptr) const {
  CL_CHECK(ext.clMemFreeINTEL(ext.context, const_cast<void *>(ptr)),
           "Failed to free USM memory: ", ptr);
  gcLogD("Deallocated USM memory: ", ptr);
  return true;
}

llvm::Expected<bool> OclRuntime::usmCpy(OclContext &ctx, const void *src,
                                        void *dst, size_t size) const {
  cl_int err;
  if (ctx.createEvents) {
    cl_event event;
    err = ext.clEnqueueMemcpyINTEL(ctx.queue, false, dst, src, size,
                                   ctx.waitListLen, ctx.waitList, &event);
    ctx.setLastEvent(event);
  } else {
    err = ext.clEnqueueMemcpyINTEL(ctx.queue, false, dst, src, size, 0, nullptr,
                                   nullptr);
  }
  CL_CHECK(err, "Failed to copy ", size, " bytes from ", src, " to ", dst);
  gcLogD("Enqueued USM memory copy of ", size, " bytes from ", src, " to ",
         dst);
  return true;
}

bool OclRuntime::isUsm(const void *ptr) const {
  cl_unified_shared_memory_type_intel allocType;
  auto err = ext.clGetMemAllocInfoINTEL(
      ext.context, ptr, CL_MEM_ALLOC_TYPE_INTEL,
      sizeof(cl_unified_shared_memory_type_intel), &allocType, nullptr);
  return err == CL_SUCCESS && allocType != CL_MEM_TYPE_UNKNOWN_INTEL;
}

#ifndef NDEBUG
void OclRuntime::debug(const char *file, int line, const char *msg) {
#ifndef GC_LOG_NO_DEBUG
  log::debug(file, line, msg);
#endif
}
#endif

OclContext::OclContext(const OclRuntime &runtime, cl_command_queue queue,
                       bool createEvents, cl_uint waitListLen,
                       cl_event *waitList)
    : runtime(runtime), queue(queue), createEvents(createEvents),
      waitListLen(createEvents ? waitListLen : 0),
      waitList(createEvents ? waitList : nullptr), lastEvent(nullptr),
      clPtrs(nullptr) {
  assert(!OclRuntime::isOutOfOrder(queue) || createEvents);
  assert(createEvents || (waitListLen == 0 && waitList == nullptr));
  for (cl_uint i = 0; i < waitListLen; i++) {
    gcLogD("Retaining OpenCL event: ", waitList[i]);
    CL_CHECKR(clRetainEvent(waitList[i]),
              "Failed to retain OpenCL event: ", waitList[i]);
  }
}

OclContext::~OclContext() {
  for (cl_uint i = 0; i < waitListLen; i++) {
    gcLogD("Releasing OpenCL event: ", waitList[i]);
    CL_CHECKR(clReleaseEvent(waitList[i]),
              "Failed to release OpenCL event: ", waitList[i]);
  }
}

llvm::Expected<bool> OclContext::finish() {
  if (createEvents) {
    if (waitListLen) {
      gcLogD("Waiting for ", waitListLen, " OpenCL events to finish.");
      CL_CHECK(clWaitForEvents(waitListLen, waitList),
               "Failed to wait for OpenCL events.");

      for (cl_uint i = 0; i < waitListLen; i++) {
        gcLogD("Releasing OpenCL event: ", waitList[i]);
        CL_CHECK(clReleaseEvent(waitList[i]),
                 "Failed to release OpenCL event: ", waitList[i]);
      }
      waitListLen = 0;
      waitList = nullptr;
    }
  } else {
    gcLogD("Waiting for the enqueued OpenCL commands to finish: ", queue);
    CL_CHECK(clFinish(queue),
             "Failed to finish the OpenCL command queue: ", queue);
  }
  return true;
}

void OclContext::setLastEvent(cl_event event) {
  for (cl_uint i = 0; i < waitListLen; i++) {
    gcLogD("Releasing OpenCL event: ", waitList[i]);
    CL_CHECKR(clReleaseEvent(waitList[i]),
              "Failed to release OpenCL event: ", waitList[i]);
  }

  gcLogD("Setting the last OpenCL event: ", event);
  lastEvent = event;
  if (event) {
    waitListLen = 1;
    waitList = &lastEvent;
  } else {
    waitListLen = 0;
    waitList = nullptr;
  }
}

static void destroyKernels(const std::unique_ptr<ExecutionEngine> &engine) {
  if (auto fn = engine->lookup(GPU_OCL_MOD_DESTRUCTOR)) {
    reinterpret_cast<void (*)()>(fn.get())();
  } else {
    llvm::consumeError(fn.takeError());
    gcLogE("Module function ", GPU_OCL_MOD_DESTRUCTOR, " not found!");
  }
}

OclModule::~OclModule() {
  assert(engine);
  destroyKernels(engine);
}

// If all arguments of 'origFunc' are memrefs with static shape, create a new
// function called gcGpuOclStaticMain, that accepts 2 arguments: a pointer to
// OclContext and a pointer to an array, containing pointers to aligned memory
// buffers. The function will call the original function with the context,
// buffers and the offset/shape/strides, statically created from the
// memref descriptor.
StringRef createStaticMain(ModuleOp &module, const StringRef &funcName,
                           const ArrayRef<Type> argTypes) {
  auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!mainFunc) {
    gcReportErr("The function '", funcName.begin(), "' not found.");
  }

  // Check that the last 3 args are added by AddContextArg
  auto mainArgTypes = mainFunc.getArgumentTypes();
  auto nargs = mainArgTypes.size();
  if (nargs < 3) {
    gcReportErr("The function '", funcName.begin(),
                "' must have an least 3 arguments.");
  }

  OpBuilder builder(module.getContext());
  auto i64Type = builder.getI64Type();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  if (mainArgTypes[nargs - 3] != ptrType ||
      mainArgTypes[nargs - 2] != ptrType ||
      mainArgTypes[nargs - 1] != i64Type) {
    gcReportErr("The last 3 arguments of the function '", funcName.begin(),
                "' must be of type (!llvm.ptr, !llvm.ptr, i64).");
  }

  // argTypes contains only the original arguments, before lowering
  nargs = argTypes.size();
  if (nargs == 0) {
    // This is a no-arg function with the context param added by AddContextArg
    return funcName;
  }

  mainFunc.setAlwaysInline(true);
  SmallVector<int64_t, 64> constArgs;
  unsigned argsCounter = 0;

  for (unsigned i = 0; i < nargs; ++i) {
    auto type = mlir::dyn_cast<MemRefType>(argTypes[i]);
    if (!type) {
      if (auto tt = mlir::dyn_cast<TensorType>(argTypes[i])) {
        type = MemRefType::get(tt.getShape(), tt.getElementType());
      }
    }

    if (type) {
      if (!type.hasStaticShape()) {
        gcLogD("The argument ", i, " of the function ", funcName.begin(),
               " has a dynamic shape.");
        return {};
      }

      auto shape = type.getShape();
      auto offsetPtr = constArgs.end();
      constArgs.emplace_back(0);
      constArgs.append(shape.begin(), shape.end());
      if (failed(type.getStridesAndOffset(constArgs, *offsetPtr))) {
        gcLogD("Failed to get strides and offset of arg", i,
               " of the function ", funcName.begin());
        return {};
      }
      argsCounter += shape.size() * 2 + 3;
    } else {
      gcLogD("The argument ", i, " of the function ", funcName.begin(),
             " is not of type memref or tensor.");
      return {};
    }
  }

  auto loc = mainFunc.getLoc();
  auto newFuncType = LLVM::LLVMFunctionType::get(
      mainFunc.getNumResults() ? mainFunc->getResult(0).getType()
                               : LLVM::LLVMVoidType::get(builder.getContext()),
      {ptrType, ptrType});
  builder.setInsertionPointToEnd(module.getBody());
  auto newFunc =
      LLVM::LLVMFuncOp::create(builder, loc, "gcGpuOclStaticMain", newFuncType);
  auto &entryBlock = *newFunc.addEntryBlock(builder);
  builder.setInsertionPointToStart(&entryBlock);
  Value arrayPtr = entryBlock.getArgument(1);

  std::unordered_map<int64_t, Value> constMap;
  auto createConst = [&](int64_t i) {
    if (auto v = constMap.find(i); v != constMap.end()) {
      return v->second;
    }
    return constMap
        .emplace(i,
                 LLVM::ConstantOp::create(builder, loc, i64Type,
                                          builder.getIntegerAttr(i64Type, i)))
        .first->second;
  };
  Value zero = createConst(0);
  Value one = nargs ? createConst(1) : Value{};
  SmallVector<Value, 64> args;
  args.reserve(argsCounter);

  for (unsigned i = 0, j = 0; i < nargs; i++) {
    if (i != 0) {
      arrayPtr =
          LLVM::GEPOp::create(builder, loc, ptrType, ptrType, arrayPtr, one);
    }

    auto ptr = LLVM::LoadOp::create(builder, loc, ptrType, arrayPtr);
    args.emplace_back(ptr);
    args.emplace_back(ptr);
    args.emplace_back(createConst(constArgs[j++]));

    for (unsigned k = 0,
                  m = 2 * mlir::cast<ShapedType>(argTypes[i]).getShape().size();
         k < m; k++) {
      args.emplace_back(createConst(constArgs[j++]));
    }
  }

  auto oclCtxArg = entryBlock.getArgument(0);
  args.emplace_back(oclCtxArg);
  args.emplace_back(oclCtxArg);
  args.emplace_back(zero);

  auto call = LLVM::CallOp::create(builder, loc, mainFunc, args);
  LLVM::ReturnOp::create(builder, loc, call.getResults());
  return newFunc.getName();
}

StringRef getFuncName(const OclModuleBuilderOpts &opts, ModuleOp &mod) {
  if (!opts.funcName.empty()) {
    return opts.funcName;
  }
  for (auto &op : mod.getBody()->getOperations()) {
    if (auto fn = dyn_cast<func::FuncOp>(op);
        fn && !fn.isExternal() && fn.isPublic()) {
      return fn.getName();
    }
    if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
        fn && !fn.isExternal() && fn.isPublic()) {
      return fn.getName();
    }
  }
  gcReportErr("Failed to find a public function in the module.");
}

OclModuleBuilder::OclModuleBuilder(ModuleOp module,
                                   const OclModuleBuilderOpts &opts)
    : mlirModule(module), dumpIr(opts.dumpIr), dumpSpirv(opts.dumpSpirv),
      enableObjectDump(opts.enableObjectDump),
      sharedLibPaths(opts.sharedLibPaths),
      pipeline(opts.pipeline ? opts.pipeline
                             : [callFinish = opts.callFinish](OpPassManager &pm,
                                  GPUPipelineOptions &
                                      opts) {
                                        opts.callFinish = callFinish;
                                        populateGPUPipeline(pm, opts);
                                       }),
      funcName(getFuncName(opts, mlirModule)), argTypes(),
      outArgsMask(0xFFFFFFFFFFFFFFFFULL) {
  if (auto fn = mlirModule.lookupSymbol<FunctionOpInterface>(funcName)) {
    auto args = fn.getArgumentTypes();
    auto rets = fn.getResultTypes();
    argTypes.reserve(args.size() + rets.size());
    argTypes.append(args.begin(), args.end());
    argTypes.append(rets.begin(), rets.end());
    if (fn.getNumResults()) {
      outArgsMask = 0xFFFFFFFFFFFFFFFFULL << args.size();
      for (unsigned i = 0, n = args.size(); i < n; ++i) {
        if (fn.getArgAttr(i, "bufferize.result")) {
          outArgsMask |= 1ULL << i;
        }
      }
    }
  } else {
    gcReportErr("Failed to find the function '", funcName.begin(),
                "' in the module.");
  }
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(const OclRuntime &runtime) {
  {
    std::shared_lock<std::shared_mutex> lock(mux);
    if (auto it = cache.find(runtime.ext); it != cache.end()) {
      return it->second;
    }
  }
  return build(runtime.ext);
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(cl_command_queue queue) {
  auto rt = OclRuntime::get(queue);
  CHECKE(rt, "Failed to create OclRuntime.");
  return build(rt.get());
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(cl_device_id device, cl_context context) {
  {
    OclDevCtxPair pair{device, context};
    std::shared_lock<std::shared_mutex> lock(mux);
    if (auto it = cache.find(pair); it != cache.end()) {
      return it->second;
    }
  }

  auto ext = OclRuntime::Ext::get(device, context);
  CHECKE(ext, "Failed to create OclRuntime::Ext.");
  return build(*ext.get());
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(const OclRuntime::Ext &ext) {
  OclRuntime rt(ext);
  auto expectedQueue = rt.createQueue();
  CHECKE(expectedQueue, "Failed to create queue!");
  struct OclQueue {
    cl_command_queue queue;
    ~OclQueue() { clReleaseCommandQueue(queue); }
  } queue{*expectedQueue};
  OclContext oclCtx{rt, queue.queue, false};

  ModuleOp mod;
  StringRef staticMain;
  std::unique_ptr<ExecutionEngine> eng;
  ExecutionEngineOptions opts;
  opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  opts.enableObjectDump = enableObjectDump;
  opts.sharedLibPaths = sharedLibPaths;
#ifdef NDEBUG
  opts.enableGDBNotificationListener = false;
  opts.enablePerfNotificationListener = false;
#endif

  auto dev = ext.device;
  GpuDevicePropsOptions devProps;
  GPUPipelineOptions pipelineOpts;
  pipelineOpts.deviceProps = &devProps;
  pipelineOpts.dump = dumpIr;
  devProps.id = clGetDevInfo(cl_uint, dev, CL_DEVICE_ID_INTEL);
  devProps.name = clGetDevInfo(std::string, dev, CL_DEVICE_NAME);
  devProps.maxWgSize = clGetDevInfo(size_t, dev, CL_DEVICE_MAX_WORK_GROUP_SIZE);
  devProps.sgSizes =
      clGetDevInfo(SmallVector<size_t>, dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL);

  mod = mlirModule.clone();
  PassManager pm{mod.getContext()};
  pipeline(pm, pipelineOpts);
  CHECK(!pm.run(mod).failed(), "GPU pipeline failed!");
  staticMain = createStaticMain(mod, funcName, argTypes);
  auto expectedEng = ExecutionEngine::create(mod, opts);
  CHECKE(expectedEng, "Failed to create ExecutionEngine!");
  eng = std::move(*expectedEng);
  eng->registerSymbols(OclRuntime::Exports::symbolMap);

  if (dumpSpirv) {
    mod->walk([&](LLVM::GlobalOp global) {
      auto isaKernel = [&](LLVM::GlobalOp op) {
        return op.getName().starts_with("gcGpuOclKernel_") &&
               op.getName().ends_with("SPIRV");
      };

      if (!isaKernel(global)) return WalkResult::skip();

      auto name = global.getName();
      gcLogD("Found a kernel to dump (", name.str(), ")");

      std::error_code ec;
      std::string filename = "GC_" + name.str() + ".spv";
      llvm::raw_fd_ostream spvStream(filename, ec);
      if (ec) {
        gcLogE("Failed to create a file `", filename,
               "`, error message: ", ec.message());
        return WalkResult::skip();
      }

      auto val = global.getValue();
      assert(val && "unexpected empty kernel");
      auto string = llvm::cast<mlir::StringAttr>(*val);
      spvStream.write(string.data(), string.size());

      if (spvStream.has_error()) {
        gcLogE("An error occured while writing to `", filename, "`.");
        return WalkResult::skip();
      }

      spvStream.flush();
      return WalkResult::skip();
    });
  }

  OclModule::MainFunc main = {nullptr};

  if (staticMain.empty()) {
    auto expect = eng->lookupPacked(funcName);
    CHECKE(expect, "Packed function '", funcName.begin(), "' not found!");
    main.wrappedMain = *expect;
  } else {
    auto expect = eng->lookup(staticMain);
    CHECKE(expect, "Compiled function '", staticMain.begin(), "' not found!");
    main.staticMain = reinterpret_cast<OclModule::StaticMainFunc>(*expect);
  }

  std::lock_guard<std::shared_mutex> lock(mux);
  if (auto it = cache.find(ext); it != cache.end()) {
    return it->second;
  }
  std::shared_ptr<const OclModule> ptr(new OclModule(
      rt, !staticMain.empty(), main, argTypes, outArgsMask, std::move(eng)));
  return cache.emplace(OclDevCtxPair(ext.device, ext.context), ptr)
      .first->second;
}
} // namespace mlir::gc::gpu