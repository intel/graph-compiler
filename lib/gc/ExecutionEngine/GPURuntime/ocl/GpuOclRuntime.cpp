//===-- GpuOclRuntime.cpp - GPU OpenCL Runtime ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl_ext.h>

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"

#ifdef GC_ENABLE_GPU_PROFILE
#include "PtiGpuUtils.h"
#include "pti/pti_view.h"
std::map<std::pair<pti_view_external_kind, uint64_t>, std::vector<uint32_t>>
    external_corr_map;
std::map<uint32_t, std::string> runtime_enq_2_gpu_kernel_name_map;
std::map<uint32_t, std::string> runtime_enq_2_gpu_mem_op_name_map;

class GPUKernelTracer {
public:
  GPUKernelTracer() {
    gcLogD("Enable Profiling.");
    ptiViewSetCallbacks(
        [](auto **buf, auto *buf_size) {
          *buf_size = sizeof(pti_view_record_kernel) * 100;
          void *ptr = ::operator new(*buf_size);
          ptr = std::align(8, sizeof(unsigned char), ptr, *buf_size);
          *buf = reinterpret_cast<unsigned char *>(ptr);
          if (!*buf) {
            std::abort();
          }
          return;
        },
        [](auto *buf, auto buf_size, auto valid_buf_size) {
          if (!buf_size || !valid_buf_size || !buf_size) {
            std::cerr << "Received empty buffer" << '\n';
            if (valid_buf_size) {
              ::operator delete(buf);
            }
            return;
          }
          pti_view_record_base *ptr = nullptr;
          while (true) {
            auto buf_status = ptiViewGetNextRecord(buf, valid_buf_size, &ptr);
            if (buf_status == pti_result::PTI_STATUS_END_OF_BUFFER) {
              std::cout << "Reached End of buffer" << '\n';
              break;
            }
            if (buf_status != pti_result::PTI_SUCCESS) {
              std::cerr << "Found Error Parsing Records from PTI" << '\n';
              break;
            }
            switch (ptr->_view_kind) {
            case pti_view_kind::PTI_VIEW_INVALID: {
              std::cout << "Found Invalid Record" << '\n';
              break;
            }
            case pti_view_kind::PTI_VIEW_DEVICE_GPU_MEM_COPY: {
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              pti_view_record_memory_copy *rec =
                  reinterpret_cast<pti_view_record_memory_copy *>(ptr);
              runtime_enq_2_gpu_mem_op_name_map[rec->_correlation_id] =
                  rec->_name;
              std::cout << "Found Memory Record" << '\n';
              samples_utils::dump_record(rec);
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              break;
            }
            case pti_view_kind::PTI_VIEW_DEVICE_GPU_MEM_FILL: {
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              pti_view_record_memory_fill *rec =
                  reinterpret_cast<pti_view_record_memory_fill *>(ptr);
              runtime_enq_2_gpu_mem_op_name_map[rec->_correlation_id] =
                  rec->_name;
              std::cout << "Found Memory Record" << '\n';
              samples_utils::dump_record(rec);
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              break;
            }
            case pti_view_kind::PTI_VIEW_DEVICE_GPU_KERNEL: {
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              pti_view_record_kernel *rec =
                  reinterpret_cast<pti_view_record_kernel *>(ptr);
              runtime_enq_2_gpu_kernel_name_map[rec->_correlation_id] =
                  rec->_name;
              std::cout << "Found Kernel Record" << '\n';
              samples_utils::dump_record(rec);

              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              if (samples_utils::isMonotonic(
                      {rec->_sycl_task_begin_timestamp,
                       rec->_sycl_enqk_begin_timestamp, rec->_append_timestamp,
                       rec->_submit_timestamp, rec->_start_timestamp,
                       rec->_end_timestamp})) {
                std::cout << "------------>     All Monotonic" << std::endl;
              } else {
                std::cerr
                    << "------------>     Something wrong: NOT All monotonic"
                    << std::endl;
              };
              if (rec->_sycl_task_begin_timestamp == 0) {
                std::cerr << "------------>     Something wrong: Sycl Task "
                             "Begin Time is 0"
                          << std::endl;
              }
              if (rec->_sycl_enqk_begin_timestamp == 0) {
                std::cerr << "------------>     Something wrong: Sycl Enq "
                             "Launch Kernel Time is 0"
                          << std::endl;
              }

              break;
            }
            case pti_view_kind::PTI_VIEW_EXTERNAL_CORRELATION: {
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              pti_view_record_external_correlation *rec =
                  reinterpret_cast<pti_view_record_external_correlation *>(ptr);

              external_corr_map[std::pair{rec->_external_kind,
                                          rec->_external_id}]
                  .push_back(rec->_correlation_id);
              samples_utils::dump_record(rec);
              break;
            }
            case pti_view_kind::PTI_VIEW_OPENCL_CALLS: {
              std::cout << "---------------------------------------------------"
                           "-----------------------------"
                        << '\n';
              pti_view_record_oclcalls *rec =
                  reinterpret_cast<pti_view_record_oclcalls *>(ptr);
              samples_utils::dump_record(rec);
              break;
            }
            default: {
              std::cerr << "This shouldn't happen" << '\n';
              break;
            }
            }
          }
          ::operator delete(buf);
        });
    ptiViewSetOclProfiling();

    ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL);
    ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY);
    ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL);
    ptiViewEnable(PTI_VIEW_OPENCL_CALLS);
    ptiViewEnable(PTI_VIEW_EXTERNAL_CORRELATION);
  }

  ~GPUKernelTracer() {
    gcLogD("Profiling is finished.");
    ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL);
    ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY);
    ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL);
    ptiViewEnable(PTI_VIEW_OPENCL_CALLS);
    ptiViewDisable(PTI_VIEW_EXTERNAL_CORRELATION);
    ptiFlushAllViews();
  }
};

/*
Create an RAII tracer with a static life cycle to trace all device kernel
execution during the program. When the tracer's constructor is called, the
EnableProfiling will also be called, registering some metric collection
call-back function into the opencl function call. When the tracer is destroyed,
the DisableProfiling is also called which will statistic the collected metric
during the tracer lifetime and print the result. The concrete implementation of
EnableProfiling and DisableProfiling could refer to
https://github.com/intel/pti-gpu/blob/master/tools/onetrace/tool.cc.
*/
static GPUKernelTracer tracer;

#endif

namespace mlir::gc::gpu {

#define makeClErrPref(code) "OpenCL error ", code, ": "
#define makeClErr(code, ...) gcMakeErr(makeClErrPref(code), __VA_ARGS__)
#define reportClErr(code, ...) gcReportErr(makeClErrPref(code), __VA_ARGS__)

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond))                                                               \
      return gcMakeErr(__VA_ARGS__);                                           \
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
  const size_t globalSize[3];
  const size_t localSize[3];
  const SmallVector<size_t> argSize;

  explicit Kernel(cl_program program, cl_kernel kernel, const size_t *gridSize,
                  const size_t *blockSize, size_t argNum, const size_t *argSize)
      : program(program),
        kernel(kernel), globalSize{gridSize[0] * blockSize[0],
                                   gridSize[1] * blockSize[1],
                                   gridSize[2] * blockSize[2]},
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
    gcLogD("Kernel ", kernel, " params: globalSize=[", globalSize[0], ", ",
           globalSize[1], ", ", globalSize[2], "], localSize=[", localSize[0],
           ", ", localSize[1], ", ", localSize[2], "], argSize=[", args.c_str(),
           "]");
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
    llvm::orc::SymbolMap map;
    map.reserve(8);
    map.try_emplace(interner(GPU_OCL_MALLOC_DEV),
                    llvm::orc::ExecutorAddr::fromPtr(&allocDev),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_MALLOC_SHARED),
                    llvm::orc::ExecutorAddr::fromPtr(&allocShared),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_DEALLOC),
                    llvm::orc::ExecutorAddr::fromPtr(&dealloc),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_MEMCPY),
                    llvm::orc::ExecutorAddr::fromPtr(&memcpy),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_KERNEL_CREATE),
                    llvm::orc::ExecutorAddr::fromPtr(&kernelCreate),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_KERNEL_DESTROY),
                    llvm::orc::ExecutorAddr::fromPtr(&kernelDestroy),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_KERNEL_LAUNCH),
                    llvm::orc::ExecutorAddr::fromPtr(&kernelLaunch),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_FINISH),
                    llvm::orc::ExecutorAddr::fromPtr(&finish),
                    llvm::JITSymbolFlags::Exported);
    return map;
  }

private:
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
                              const size_t *gridSize, const size_t *blockSize,
                              size_t argNum, const size_t *argSize) {
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
      // This is a special case, handled by OclModuleBuilder::build(), that
      // allows rebuilding the kernel with different options in case of failure.
      clReleaseProgram(program);
      gcLogD("OpenCL error ", err,
             ": Failed to create OpenCL kernel from program: ", program);
      return new Kernel(nullptr, nullptr, gridSize, blockSize, argNum, argSize);
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

    return new Kernel(program, kernel, gridSize, blockSize, argNum, argSize);
  }

  static void kernelDestroy(size_t count, Kernel **kernels) {
    gcLogD("Destroying kernels.");
    for (size_t i = 0; i < count; i++) {
      if (kernels[i]) {
        delete kernels[i];
      }
    }
  }

  static void kernelLaunch(OclContext *ctx, Kernel *kernel, ...) {
    struct ClonedKernel {
      cl_kernel kernel;

      explicit ClonedKernel(cl_kernel kernel) : kernel(kernel) {}

      ~ClonedKernel() {
        gcLogD("Releasing cloned OpenCL kernel: ", kernel);
        CL_CHECKR(clReleaseKernel(kernel),
                  "Failed to release the kernel: ", kernel);
      }
    };

    gcLogD("Launching kernel: ", kernel->kernel);

    cl_int err;
    ClonedKernel cloned{clCloneKernel(kernel->kernel, &err)};
    CL_CHECKR(err, "Failed to clone OpenCL kernel: ", kernel->kernel);
    gcLogD("Cloned OpenCL kernel ", kernel->kernel, ": ", cloned.kernel);

    va_list args;
    va_start(args, kernel);
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
                                   kernel->globalSize, kernel->localSize,
                                   ctx->waitListLen, ctx->waitList, &event);
      ctx->setLastEvent(event);
    } else {
      err = clEnqueueNDRangeKernel(ctx->queue, cloned.kernel, 3, nullptr,
                                   kernel->globalSize, kernel->localSize, 0,
                                   nullptr, nullptr);
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
      gcLogE("Failed to get the number of devices on the platform.", platform,
             " Error: ", err);
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
  cl_mem_info_intel allocType;
  auto err = ext.clGetMemAllocInfoINTEL(
      ext.context, ptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(cl_mem_info_intel),
      &allocType, nullptr);
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
  auto fn = engine->lookup(GPU_OCL_MOD_DESTRUCTOR);
  if (fn) {
    reinterpret_cast<void (*)()>(fn.get())();
  } else {
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
StringRef createStaticMain(OpBuilder &builder, ModuleOp &module,
                           const StringRef &funcName,
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
    if (auto type = mlir::dyn_cast<MemRefType>(argTypes[i])) {
      if (!type.hasStaticShape()) {
        gcLogD("The argument ", i, " of the function ", funcName.begin(),
               " has a dynamic shape.");
        return {};
      }

      auto shape = type.getShape();
      auto offsetPtr = constArgs.end();
      constArgs.emplace_back(0);
      constArgs.append(shape.begin(), shape.end());
      if (failed(getStridesAndOffset(type, constArgs, *offsetPtr))) {
        gcLogD("Failed to get strides and offset of arg", i,
               " of the function ", funcName.begin());
        return {};
      }
      argsCounter += shape.size() * 2 + 3;
    } else {
      gcLogD("The argument ", i, " of the function ", funcName.begin(),
             " is not of type MemRefType.");
      return {};
    }
  }

  auto loc = mainFunc.getLoc();
  auto newFuncType = LLVM::LLVMFunctionType::get(
      mainFunc.getNumResults() ? mainFunc->getResult(0).getType()
                               : LLVM::LLVMVoidType::get(builder.getContext()),
      {ptrType, ptrType});
  auto newFunc =
      OpBuilder::atBlockEnd(module.getBody())
          .create<LLVM::LLVMFuncOp>(loc, "gcGpuOclStaticMain", newFuncType);
  auto &entryBlock = *newFunc.addEntryBlock(builder);
  builder.setInsertionPointToStart(&entryBlock);
  Value arrayPtr = entryBlock.getArgument(1);

  std::unordered_map<int64_t, Value> constMap;
  auto createConst = [&](int64_t i) {
    if (auto v = constMap.find(i); v != constMap.end()) {
      return v->second;
    }
    return constMap
        .emplace(i, builder.create<LLVM::ConstantOp>(
                        loc, i64Type, builder.getIntegerAttr(i64Type, i)))
        .first->second;
  };
  Value zero = createConst(0);
  Value one = nargs ? createConst(1) : Value{};
  SmallVector<Value, 64> args;
  args.reserve(argsCounter);

  for (unsigned i = 0, j = 0; i < nargs; i++) {
    if (i != 0) {
      arrayPtr =
          builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, arrayPtr, one);
    }

    auto ptr = builder.create<LLVM::LoadOp>(loc, ptrType, arrayPtr);
    args.emplace_back(ptr);
    args.emplace_back(ptr);
    args.emplace_back(createConst(constArgs[j++]));

    for (unsigned k = 0,
                  m = 2 * mlir::cast<MemRefType>(argTypes[i]).getShape().size();
         k < m; k++) {
      args.emplace_back(createConst(constArgs[j++]));
    }
  }

  auto oclCtxArg = entryBlock.getArgument(0);
  args.emplace_back(oclCtxArg);
  args.emplace_back(oclCtxArg);
  args.emplace_back(zero);

  auto call = builder.create<LLVM::CallOp>(loc, mainFunc, args);
  builder.create<LLVM::ReturnOp>(loc, call.getResults());
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

ArrayRef<Type> getArgTypes(const StringRef &funcName, ModuleOp &mod) {
  if (auto fn = mod.lookupSymbol<func::FuncOp>(funcName)) {
    return fn.getArgumentTypes();
  }
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    return fn.getArgumentTypes();
  }
  gcReportErr("Failed to find the function '", funcName.begin(),
              "' in the module.");
}

OclModuleBuilder::OclModuleBuilder(ModuleOp module,
                                   const OclModuleBuilderOpts &opts)
    : mlirModule(module), printIr(opts.printIr),
      enableObjectDump(opts.enableObjectDump),
      sharedLibPaths(opts.sharedLibPaths),
      pipeline(opts.pipeline
                   ? opts.pipeline
                   : [](OpPassManager &pm) { populateGPUPipeline(pm, {}); }),
      funcName(getFuncName(opts, mlirModule)),
      argTypes(getArgTypes(funcName, mlirModule)) {}

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
  auto ctx = mlirModule.getContext();
  ctx->getOrLoadDialect<DLTIDialect>();
  ctx->getOrLoadDialect<LLVM::LLVMDialect>();
  OpBuilder builder(ctx);
  DataLayoutEntryInterface dltiAttrs[6];

  {
    struct DevInfo {
      cl_device_info key;
      const char *attrName;
    };
    DevInfo devInfo[]{
        {CL_DEVICE_MAX_COMPUTE_UNITS, "num_exec_units"},
        {CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, "num_exec_units_per_slice"},
        {CL_DEVICE_NUM_THREADS_PER_EU_INTEL, "num_threads_per_eu"},
        {CL_DEVICE_LOCAL_MEM_SIZE, "local_mem_size"},
    };

    unsigned i = 0;
    for (auto &[key, attrName] : devInfo) {
      int64_t value = 0;
      CL_CHECK(
          clGetDeviceInfo(ext.device, key, sizeof(cl_ulong), &value, nullptr),
          "Failed to get the device property ", attrName);
      gcLogD("Device property ", attrName, "=", value);
      dltiAttrs[i++] =
          DataLayoutEntryAttr::get(ctx, builder.getStringAttr(attrName),
                                   builder.getI64IntegerAttr(value));
    }

    // There is no a corresponding property in the OpenCL API, using the
    // hardcoded value.
    // TODO: Get the real value.
    dltiAttrs[i] = DataLayoutEntryAttr::get(
        ctx, builder.getStringAttr("max_vector_op_width"),
        builder.getI64IntegerAttr(512));
  }

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
  auto devStr = builder.getStringAttr("GPU" /* device ID*/);
  ExecutionEngineOptions opts;
  opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  opts.enableObjectDump = enableObjectDump;
  opts.sharedLibPaths = sharedLibPaths;
#ifdef NDEBUG
  opts.enableGDBNotificationListener = false;
  opts.enablePerfNotificationListener = false;
#endif

  // Build the module and check the kernels workgroup size. If the workgroup
  // size is different, rebuild the module with the new size.
  for (size_t wgSize = 64, maxSize = std::numeric_limits<size_t>::max();;) {
    dltiAttrs[sizeof(dltiAttrs) / sizeof(DataLayoutEntryInterface) - 1] =
        DataLayoutEntryAttr::get(
            ctx, builder.getStringAttr("max_work_group_size"),
            builder.getI64IntegerAttr(static_cast<int64_t>(wgSize)));
    TargetDeviceSpecInterface devSpec =
        TargetDeviceSpecAttr::get(ctx, dltiAttrs);
    auto sysSpec =
        TargetSystemSpecAttr::get(ctx, ArrayRef(std::pair(devStr, devSpec)));
    mod = mlirModule.clone();
    mod.getOperation()->setAttr("#dlti.sys_spec", sysSpec);
    PassManager pm{ctx};
    pipeline(pm);
    CHECK(!pm.run(mod).failed(), "GPU pipeline failed!");
    staticMain = createStaticMain(builder, mod, funcName, argTypes);
    auto expectedEng = ExecutionEngine::create(mod, opts);
    CHECKE(expectedEng, "Failed to create ExecutionEngine!");
    expectedEng->get()->registerSymbols(OclRuntime::Exports::symbolMap);

    // Find all kernels and query the workgroup size
    size_t minSize = maxSize;
    mod.walk<>([&](LLVM::LLVMFuncOp func) {
      auto name = func.getName();
      if (!name.starts_with("createGcGpuOclKernel_")) {
        return WalkResult::skip();
      }
      auto fn = expectedEng.get()->lookup(name);
      if (!fn) {
        gcLogE("Function not found: ", name.data());
        return WalkResult::skip();
      }

      Kernel *kernel =
          reinterpret_cast<Kernel *(*)(OclContext *)>(fn.get())(&oclCtx);

      if (kernel->kernel == nullptr) {
        maxSize = wgSize / 2;
        if (maxSize == 0) {
          gcReportErr("Failed to build the kernel.");
        }
        minSize = maxSize;
        return WalkResult::interrupt();
      }

      size_t s = 0;
      auto err = clGetKernelWorkGroupInfo(kernel->kernel, ext.device,
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(size_t), &s, nullptr);
      if (err == CL_SUCCESS) {
        minSize = std::min(minSize, s);
      } else {
        gcLogE("Failed to get the kernel workgroup size: ", err);
      }
      return WalkResult::skip();
    });

    if (minSize == wgSize || minSize == std::numeric_limits<size_t>::max()) {
      eng = std::move(*expectedEng);
      break;
    }

    destroyKernels(expectedEng.get());
    gcLogD("Changing the workgroup size from ", wgSize, " to ", minSize);
    wgSize = minSize;
  }

  if (printIr) {
    mod.dump();
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
  std::shared_ptr<const OclModule> ptr(
      new OclModule(rt, !staticMain.empty(), main, argTypes, std::move(eng)));
  return cache.emplace(OclDevCtxPair(ext.device, ext.context), ptr)
      .first->second;
}
} // namespace mlir::gc::gpu
