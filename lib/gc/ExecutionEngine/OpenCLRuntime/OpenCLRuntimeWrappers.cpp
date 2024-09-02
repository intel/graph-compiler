//===-- OpenCLRuntimeWrappers.cpp - OpenCLRuntimeWrappers -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define OCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define OCL_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

namespace {

#define CL_SAFE_CALL2(a)                                                       \
  do {                                                                         \
    (a);                                                                       \
    if (err != CL_SUCCESS) {                                                   \
      fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a));   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CL_SAFE_CALL(call)                                                     \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != CL_SUCCESS) {                                                \
      fprintf(stderr, "CL error %d @ line=%d (%s)\n", status, __LINE__,        \
              (#call));                                                        \
      abort();                                                                 \
    }                                                                          \
  }

constexpr char DeviceMemAllocName[] = "clDeviceMemAllocINTEL";
constexpr char SharedMemAllocName[] = "clSharedMemAllocINTEL";
constexpr char MemBlockingFreeName[] = "clMemBlockingFreeINTEL";
constexpr char SetKernelArgMemPointerName[] = "clSetKernelArgMemPointerINTEL";
static constexpr char EnqueueMemcpyName[] = "clEnqueueMemcpyINTEL";

void *queryCLExtFunc(cl_platform_id CurPlatform, const char *FuncName) {
  void *ret = clGetExtensionFunctionAddressForPlatform(CurPlatform, FuncName);

  if (!ret) {
    fflush(stderr);
    abort();
  }
  return ret;
}

void *queryCLExtFunc(cl_device_id dev, const char *FuncName) {
  cl_platform_id CurPlatform;
  CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                               &CurPlatform, nullptr));
  return queryCLExtFunc(CurPlatform, FuncName);
}

struct CLExtTable {
  clDeviceMemAllocINTEL_fn allocDev;
  clSharedMemAllocINTEL_fn allocShared;
  clMemBlockingFreeINTEL_fn blockingFree;
  clSetKernelArgMemPointerINTEL_fn setKernelArgMemPtr;
  clEnqueueMemcpyINTEL_fn enqueueMemcpy;
  CLExtTable() = default;
  CLExtTable(cl_platform_id plat) {
    allocDev =
        (clDeviceMemAllocINTEL_fn)queryCLExtFunc(plat, DeviceMemAllocName);
    allocShared =
        (clSharedMemAllocINTEL_fn)queryCLExtFunc(plat, SharedMemAllocName);
    blockingFree =
        (clMemBlockingFreeINTEL_fn)queryCLExtFunc(plat, MemBlockingFreeName);
    setKernelArgMemPtr = (clSetKernelArgMemPointerINTEL_fn)queryCLExtFunc(
        plat, SetKernelArgMemPointerName);
    enqueueMemcpy =
        (clEnqueueMemcpyINTEL_fn)queryCLExtFunc(plat, EnqueueMemcpyName);
  }
};

struct CLExtTableCache {
  cl_platform_id platform;
  CLExtTable table;
  CLExtTableCache(cl_platform_id CurPlatform)
      : platform{CurPlatform}, table{CurPlatform} {}
  static CLExtTable *get(cl_device_id dev) {
    cl_platform_id CurPlatform;
    CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id), &CurPlatform,
                                 nullptr));
    static CLExtTableCache v{CurPlatform};
    if (v.platform == CurPlatform) {
      return &v.table;
    }
    return nullptr;
  }
};

struct ParamDesc {
  void *data;
  size_t size;

  bool operator==(const ParamDesc &rhs) const {
    return data == rhs.data && size == rhs.size;
  }

  bool operator!=(const ParamDesc &rhs) const { return !(*this == rhs); }
};

template <typename T> size_t countUntil(T *ptr, T &&elem) {
  assert(ptr);
  auto curr = ptr;
  while (*curr != elem) {
    ++curr;
  }
  return static_cast<size_t>(curr - ptr);
}
} // namespace

static cl_device_id getDevice(cl_device_type *devtype) {
  cl_uint uintValue;
  CL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &uintValue)) // get num platforms

  std::vector<cl_platform_id> platforms(uintValue);
  CL_SAFE_CALL(clGetPlatformIDs(uintValue, platforms.data(),
                                nullptr)); // get available platforms

  for (auto &platform : platforms) {
    size_t valueSize;
    CL_SAFE_CALL(
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &valueSize));
    std::string name(valueSize, 0);
    CL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, valueSize,
                                   &name[0], nullptr));
    if (name.find("Intel") == std::string::npos) { // Ignore non-Intel platforms
      continue;
    }

    // Get GPU device IDs for each platform
    cl_int status =
        clGetDeviceIDs(platform, *devtype, 0, /*devices.data()=*/nullptr,
                       &uintValue); // get num devices with 'devtype'
    if (status != CL_SUCCESS) {
      if (status == CL_DEVICE_NOT_FOUND) {
        continue; // No GPU devices found on this platform
      }
      fprintf(stderr, "CL error %d @ line=%d (%s)\n", status, __LINE__,
              "Error getting device IDs");
      abort();
    }

    std::vector<cl_device_id> devices(uintValue);
    clGetDeviceIDs(platform, *devtype, uintValue, devices.data(), nullptr);

    for (auto &device : devices) {
      CL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(cl_uint),
                                   &uintValue, nullptr));
      if (uintValue == 0x8086) { // Make sure this is an Intel device
        return device;
      }
    }
  }

  fprintf(stderr, "No suitable devices found.");
  abort();
}

struct GPUCLQUEUE {
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  bool context_owned_ = false;
  bool queue_owned_ = false;
  CLExtTable *ext_table_ = nullptr;
  std::vector<cl_program> programs_;
  std::vector<cl_kernel> kernels_;

  GPUCLQUEUE(cl_device_type *device, cl_context context,
             cl_command_queue queue) {
    cl_device_type defaultdev = CL_DEVICE_TYPE_GPU;
    if (!device) {
      device = &defaultdev;
    }
    device_ = getDevice(device);
    init_context(context, queue, device_);
    ext_table_ = CLExtTableCache::get(device_);
  }
  GPUCLQUEUE(cl_device_id device, cl_context context, cl_command_queue queue) {
    if (!device) {
      cl_device_type defaultdev = CL_DEVICE_TYPE_GPU;
      device = getDevice(&defaultdev);
    }
    device_ = device;
    init_context(context, queue, device_);
    ext_table_ = CLExtTableCache::get(device_);
  }
  ~GPUCLQUEUE() {
    for (auto p : kernels_) {
      clReleaseKernel(p);
    }
    for (auto p : programs_) {
      clReleaseProgram(p);
    }
    if (queue_ && queue_owned_)
      clReleaseCommandQueue(queue_);
    if (context_ && context_owned_)
      clReleaseContext(context_);
  }

private:
  void init_context(cl_context context, cl_command_queue queue,
                    cl_device_id device) {
    if (queue) {
      if (!context) {
        throw std::runtime_error(
            "Cannot create QUEUE wrapper with queue and without context");
      }
      queue_ = queue;
      queue_owned_ = true;
      context_ = context;
      context_owned_ = true;
      return;
    }
    cl_int err;
    if (!context) {
      CL_SAFE_CALL2(context_ =
                        clCreateContext(NULL, 1, &device, NULL, NULL, &err));
      context_owned_ = true;
    } else {
      context_ = context;
    }
    CL_SAFE_CALL2(
        queue_ = clCreateCommandQueueWithProperties(context_, device, 0, &err));
    queue_owned_ = true;
  }
}; // end of GPUCLQUEUE

static void *allocDeviceMemory(GPUCLQUEUE *queue, size_t size, size_t alignment,
                               bool isShared) {
  void *memPtr = nullptr;
  cl_int err;
  if (isShared) {
    auto func = queue->ext_table_ ? queue->ext_table_->allocShared
                                  : (clSharedMemAllocINTEL_fn)queryCLExtFunc(
                                        queue->device_, SharedMemAllocName);
    CL_SAFE_CALL2(memPtr = func(queue->context_, queue->device_, nullptr, size,
                                alignment, &err));
  } else {
    auto func = queue->ext_table_ ? queue->ext_table_->allocDev
                                  : (clDeviceMemAllocINTEL_fn)queryCLExtFunc(
                                        queue->device_, DeviceMemAllocName);
    CL_SAFE_CALL2(memPtr = func(queue->context_, queue->device_, nullptr, size,
                                alignment, &err));
  }
  return memPtr;
}

static void deallocDeviceMemory(GPUCLQUEUE *queue, void *ptr) {
  auto func = queue->ext_table_ ? queue->ext_table_->blockingFree
                                : (clMemBlockingFreeINTEL_fn)queryCLExtFunc(
                                      queue->device_, MemBlockingFreeName);
  CL_SAFE_CALL(func(queue->context_, ptr));
}

static cl_program loadModule(GPUCLQUEUE *queue, const unsigned char *data,
                             size_t dataSize, bool takeOwnership) {
  assert(data);
  cl_int errNum = 0;
  const unsigned char *codes[1] = {data};
  size_t sizes[1] = {dataSize};
  cl_program program;
  cl_int err;
  CL_SAFE_CALL2(program = clCreateProgramWithBinary(queue->context_, 1,
                                                    &queue->device_, sizes,
                                                    codes, &err, &errNum));
  const char *build_flags = "-cl-kernel-arg-info -x spir";
  // enable large register file if needed
  if (getenv("IMEX_ENABLE_LARGE_REG_FILE")) {
    build_flags =
        "-vc-codegen -doubleGRF -Xfinalizer -noLocalSplit -Xfinalizer "
        "-DPASTokenReduction -Xfinalizer -SWSBDepReduction -Xfinalizer "
        "'-printregusage -enableBCR' -cl-kernel-arg-info -x spir";
  }
  CL_SAFE_CALL(clBuildProgram(program, 0, NULL, build_flags, NULL, NULL));
  if (takeOwnership)
    queue->programs_.push_back(program);
  return program;
}

static cl_kernel getKernel(GPUCLQUEUE *queue, cl_program program,
                           const char *name) {
  cl_kernel kernel;
  cl_int err;
  CL_SAFE_CALL2(kernel = clCreateKernel(program, name, &err));
  cl_bool TrueVal = CL_TRUE;
  CL_SAFE_CALL(clSetKernelExecInfo(
      kernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof(cl_bool),
      &TrueVal));
  CL_SAFE_CALL(clSetKernelExecInfo(
      kernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(cl_bool),
      &TrueVal));
  CL_SAFE_CALL(clSetKernelExecInfo(
      kernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(cl_bool),
      &TrueVal));
  queue->kernels_.push_back(kernel);
  return kernel;
}

template <typename NumArgsFuncT, typename GetParamFuncT>
static void launchKernel(GPUCLQUEUE *queue, cl_kernel kernel, size_t gridX,
                         size_t gridY, size_t gridZ, size_t blockX,
                         size_t blockY, size_t blockZ, size_t sharedMemBytes,
                         NumArgsFuncT &&fnGetNumArgs,
                         GetParamFuncT &&fnGetParamFunc) {
  auto clSetKernelArgMemPointerINTEL =
      queue->ext_table_ ? queue->ext_table_->setKernelArgMemPtr
                        : (clSetKernelArgMemPointerINTEL_fn)queryCLExtFunc(
                              queue->device_, SetKernelArgMemPointerName);
  auto paramsCount = fnGetNumArgs();
  for (size_t i = 0; i < paramsCount; i++) {
    cl_kernel_arg_address_qualifier name;
    size_t nameSize = sizeof(name);
    // we can do better here, to cache the arginfo for the kernel
    CL_SAFE_CALL(clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                                    sizeof(name), &name, &nameSize));
    auto [paramData, paramSize] = fnGetParamFunc(i);
    if (paramSize == sizeof(void *) && name == CL_KERNEL_ARG_ADDRESS_GLOBAL) {
      // pass the value of the pointer instead of the pointer of the pointer
      CL_SAFE_CALL(
          clSetKernelArgMemPointerINTEL(kernel, i, *(void **)paramData));
    } else {
      CL_SAFE_CALL(clSetKernelArg(kernel, i, paramSize, paramData));
    }
  }
  if (sharedMemBytes) {
    CL_SAFE_CALL(clSetKernelArg(kernel, paramsCount, sharedMemBytes, nullptr));
  }
  size_t globalSize[3] = {gridX * blockX, gridY * blockY, gridZ * blockZ};
  size_t localSize[3] = {blockX, blockY, blockZ};
  CL_SAFE_CALL(clEnqueueNDRangeKernel(queue->queue_, kernel, 3, NULL,
                                      globalSize, localSize, 0, NULL, NULL));
}

static GPUCLQUEUE *getDefaultQueue() {
  static GPUCLQUEUE defaultq(static_cast<cl_device_id>(nullptr), nullptr,
                             nullptr);
  return &defaultq;
}

// Wrappers

extern "C" OCL_RUNTIME_EXPORT GPUCLQUEUE *gpuCreateStream(void *device,
                                                          void *context) {
  // todo: this is a workaround of issue of gpux generating multiple streams
  if (!device && !context) {
    return getDefaultQueue();
  }
  return new GPUCLQUEUE(reinterpret_cast<cl_device_id>(device),
                        reinterpret_cast<cl_context>(context), nullptr);
}

extern "C" OCL_RUNTIME_EXPORT void gpuStreamDestroy(GPUCLQUEUE *queue) {
  // todo: this is a workaround of issue of gpux generating multiple streams
  // should uncomment the below line to release the queue
  // delete queue;
}

extern "C" OCL_RUNTIME_EXPORT void *
gpuMemAlloc(GPUCLQUEUE *queue, size_t size, size_t alignment, bool isShared) {
  if (queue) {
    return allocDeviceMemory(queue, size, alignment, isShared);
  }
  return nullptr;
}

extern "C" OCL_RUNTIME_EXPORT void gpuMemFree(GPUCLQUEUE *queue, void *ptr) {
  if (queue && ptr) {
    deallocDeviceMemory(queue, ptr);
  }
}

extern "C" OCL_RUNTIME_EXPORT void gpuMemCopy(GPUCLQUEUE *queue, void *dst,
                                              void *src, uint64_t size) {
  auto func = queue->ext_table_ ? queue->ext_table_->enqueueMemcpy
                                : (clEnqueueMemcpyINTEL_fn)queryCLExtFunc(
                                      queue->device_, EnqueueMemcpyName);
  CL_SAFE_CALL(func(queue->queue_, true, dst, src, size, 0, nullptr, nullptr));
}

extern "C" OCL_RUNTIME_EXPORT cl_program
gpuModuleLoad(GPUCLQUEUE *queue, const unsigned char *data, size_t dataSize) {
  if (queue) {
    return loadModule(queue, data, dataSize, false);
  }
  return nullptr;
}

extern "C" OCL_RUNTIME_EXPORT cl_kernel gpuKernelGet(GPUCLQUEUE *queue,
                                                     cl_program module,
                                                     const char *name) {
  if (queue) {
    return getKernel(queue, module, name);
  }
  return nullptr;
}

extern "C" OCL_RUNTIME_EXPORT void
gpuLaunchKernel(GPUCLQUEUE *queue, cl_kernel kernel, size_t gridX, size_t gridY,
                size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                size_t sharedMemBytes, void *params) {
  if (queue) {
    auto typedParams = static_cast<ParamDesc *>(params);
    launchKernel(
        queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
        sharedMemBytes,
        [&]() {
          // The assumption is, if there is a param for the shared local memory,
          // then that will always be the last argument.
          auto paramsCount = countUntil(typedParams, ParamDesc{nullptr, 0});
          if (sharedMemBytes) {
            paramsCount = paramsCount - 1;
          }
          return paramsCount;
        },
        [&](size_t i) -> const ParamDesc & { return typedParams[i]; });
  }
}

extern "C" OCL_RUNTIME_EXPORT void gpuWait(GPUCLQUEUE *queue) {
  if (queue) {
    CL_SAFE_CALL(clFinish(queue->queue_));
  }
}

////////////////////////////////////////////////////////////////
// Here starts the upstream OCL wrappers
////////////////////////////////////////////////////////////////

// a silly workaround for mgpuModuleLoad. OCL needs context and device to load
// the module. We remember the last call to any mgpu* APIs
static thread_local GPUCLQUEUE *lastQueue;
extern "C" OCL_RUNTIME_EXPORT GPUCLQUEUE *mgpuStreamCreate() {
  auto ret =
      new GPUCLQUEUE(static_cast<cl_device_id>(nullptr), nullptr, nullptr);
  lastQueue = ret;
  return ret;
}

extern "C" OCL_RUNTIME_EXPORT void mgpuStreamDestroy(GPUCLQUEUE *queue) {
  lastQueue = nullptr;
  delete queue;
}

extern "C" OCL_RUNTIME_EXPORT void *
mgpuMemAlloc(uint64_t size, GPUCLQUEUE *queue, bool isShared) {
  lastQueue = queue;
  return allocDeviceMemory(queue, size, /*alignment*/ 64, isShared);
}

extern "C" OCL_RUNTIME_EXPORT void mgpuMemFree(void *ptr, GPUCLQUEUE *queue) {
  lastQueue = queue;
  if (ptr) {
    deallocDeviceMemory(queue, ptr);
  }
}

// mgpuModuleLoad and mgpuModuleGetFunction does not have
// queue in parameters, but OCL APIs requires them. We implicitly use the queue
// pointer of the last mgpu* API of the current thread as the queue for these
// functions. This is ugly and error-prone. We might need another workaround.
extern "C" OCL_RUNTIME_EXPORT cl_program mgpuModuleLoad(const void *data,
                                                        size_t gpuBlobSize) {
  return loadModule(lastQueue, (const unsigned char *)data, gpuBlobSize, false);
}

extern "C" OCL_RUNTIME_EXPORT cl_kernel
mgpuModuleGetFunction(cl_program module, const char *name) {
  // we need to push the kernel to lastQueue to avoid cl_kernel resource leak
  return getKernel(lastQueue, module, name);
}

extern "C" OCL_RUNTIME_EXPORT void mgpuModuleUnload(cl_program module) {
  CL_SAFE_CALL(clReleaseProgram(module));
}

extern "C" OCL_RUNTIME_EXPORT void
mgpuLaunchKernel(cl_kernel kernel, size_t gridX, size_t gridY, size_t gridZ,
                 size_t blockX, size_t blockY, size_t blockZ,
                 size_t sharedMemBytes, GPUCLQUEUE *queue, void **params,
                 void ** /*extra*/, size_t paramsCount) {
  launchKernel(
      queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
      sharedMemBytes,
      [&]() {
        // todo (yijie): do we need to handle shared mem? If there is dynamic
        // shared mem required, which value should paramsCount be?
        return paramsCount;
      },
      [&](size_t i) {
        // todo (yijie): assuming all parameters are passed with pointer size
        return std::make_pair(params[i], sizeof(void *));
      });
}

extern "C" OCL_RUNTIME_EXPORT void mgpuStreamSynchronize(GPUCLQUEUE *queue) {
  CL_SAFE_CALL(clFinish(queue->queue_));
}
