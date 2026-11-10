// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/GpuCompiler.h"
#include "gc/Runtime/OclRuntime.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace mlir;
using ::gc::gpu::GpuCompiler;
using ::gc::gpu::OclRuntime;

template <typename T>
static bool run(MLIRContext &ctx, cl_context context, cl_command_queue queue,
                const char *ir, std::vector<std::vector<T>> inputs,
                const std::vector<T> &expected) {
  inputs.emplace_back(expected.size(), -999.0f);
  for (bool wait : {true, false}) {
    std::fill(inputs.back().begin(), inputs.back().end(), T(-999));
    auto runtime = OclRuntime::create(queue);
    auto module = parseSourceString<ModuleOp>(ir, &ctx);
    if (!module) return false;
    auto fn = *module->getOps<func::FuncOp>().begin();
    std::vector<std::vector<int64_t>> shapes, strides;
    for (auto type : fn.getArgumentTypes()) {
      auto memref = cast<MemRefType>(type);
      auto &shape = shapes.emplace_back(memref.getShape().begin(),
                                        memref.getShape().end());
      int64_t product = 1;
      for (int64_t dim : shape)
        if (dim != ShapedType::kDynamic) product *= dim;
      for (auto &dim : shape)
        if (dim == ShapedType::kDynamic)
          dim = inputs[shapes.size() - 1].size() / product;
      auto &stride = strides.emplace_back(shape.size());
      int64_t value = 1;
      for (size_t i = shape.size(); i-- > 0;) {
        stride[i] = value;
        value *= shape[i];
      }
    }
    ::gc::gpu::CompileOptions options;
    options.source = true;
    options.wait = wait;
    auto program = GpuCompiler::get()
                       .compile<OclRuntime>(std::move(module), options)
                       .get();
    if (!program || (wait && program->eventNum())) return false;
    ::gc::gpu::SmallBuilder<> builder;
    std::vector<cl_mem> args;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cl_int status;
      auto buffer = clCreateBuffer(
          context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          inputs[i].size() * sizeof(T), inputs[i].data(), &status);
      if (status != CL_SUCCESS) return false;
      args.push_back(buffer);
      builder.storeMemref(runtime->importBuffer(buffer), shapes[i].size(),
                          shapes[i].data(), strides[i].data());
    }
    cl_event ready = nullptr;
    clEnqueueMarkerWithWaitList(queue, 0, nullptr, &ready);
    auto events =
        program->main(builder.build(*program), runtime.get(), &ready, 1);
    auto &out = inputs.back();
    bool ok =
        clEnqueueReadBuffer(queue, args.back(), CL_TRUE, 0,
                            out.size() * sizeof(T), out.data(),
                            program->eventNum(), events, nullptr) == CL_SUCCESS;
    if (events) runtime->wait(events, program->eventNum());
    clReleaseEvent(ready);
    for (auto arg : args) clReleaseMemObject(arg);
    for (size_t i = 0; i < expected.size(); ++i) {
      bool matches;
      if constexpr (std::is_floating_point_v<T>)
        matches = std::isnan(expected[i])
                      ? std::isnan(out[i])
                      : std::isfinite(out[i]) &&
                            std::abs(out[i] - expected[i]) <= 1e-5f;
      else matches = out[i] == expected[i];
      if (!matches) {
        std::cerr << "Mismatch at " << i << ": " << out[i]
                  << " != " << expected[i] << '\n';
        ok = false;
      }
    }
    if (!ok) return false;
  }
  return true;
}

static bool runBinary(cl_context context, cl_device_id device,
                      cl_command_queue queue) {
  const char *text = "__kernel void twice(__global float *a,__global float *b){"
                     "int i=get_global_id(0);b[i]=a[i]*2;}";
  cl_int status;
  auto program = clCreateProgramWithSource(context, 1, &text, nullptr, &status);
  if (status != CL_SUCCESS || clBuildProgram(program, 1, &device, nullptr,
                                             nullptr, nullptr) != CL_SUCCESS)
    return false;
  size_t size;
  if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size,
                       nullptr) != CL_SUCCESS)
    return false;
  std::vector<uint8_t> binary(size);
  auto data = binary.data();
  if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(data), &data,
                       nullptr) != CL_SUCCESS)
    return false;
  clReleaseProgram(program);
  auto runtime = OclRuntime::create(queue);
  ::gc::gpu::ArgType types[] = {::gc::gpu::ArgType::PTR,
                                ::gc::gpu::ArgType::PTR};
  auto kernel =
      runtime->createKernel(binary.data(), binary.size(), "twice", 2, types);
  auto a = runtime->allocDevice(4 * sizeof(float), 0);
  auto b = runtime->allocShared(4 * sizeof(float), 0);
  float input[] = {-1, 0, 1, 2}, output[4]{};
  auto host = runtime->allocHost(sizeof(input), 64);
  if (reinterpret_cast<uintptr_t>(host) % 64) return false;
  std::copy(std::begin(input), std::end(input),
            reinterpret_cast<float *>(host));
  auto upload = runtime->memcpy(host, a, sizeof(input), nullptr, 0);
  auto copy = runtime->memcpy(a, b, sizeof(input), &upload, 1);
  void *args[] = {&b, &a};
  auto launch = runtime->launch(kernel, args, 2, 4, 1, 1, 1, 1, 1, &copy, 1);
  auto read = runtime->memcpy(a, reinterpret_cast<OclRuntime::Memory *>(output),
                              sizeof(output), &launch, 1);
  cl_event events[] = {upload, copy, launch, read};
  runtime->wait(events, 4);
  runtime->free(a);
  runtime->free(b);
  runtime->free(host);
  OclRuntime::destroyKernel(kernel);
  for (size_t i = 0; i < 4; ++i)
    if (output[i] != 2 * input[i]) return false;
  return true;
}

int main() {
  cl_uint count = 0;
  if (clGetPlatformIDs(0, nullptr, &count) != CL_SUCCESS || !count) return 77;
  std::vector<cl_platform_id> platforms(count);
  clGetPlatformIDs(count, platforms.data(), nullptr);
  cl_device_id device = nullptr;
  for (auto platform : platforms)
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) ==
        CL_SUCCESS)
      break;
  if (!device) return 77;
  cl_int status;
  auto context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  if (status != CL_SUCCESS) return 1;
  MLIRContext ctx(mlir::gc::getDialectRegistry());
  ctx.loadAllAvailableDialects();
  const char *fusion = R"mlir(
func.func @entry(%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>) {
  %x = bufferization.to_tensor %a restrict writable : memref<4xf32> to tensor<4xf32>
  %y = bufferization.to_tensor %b restrict writable : memref<4xf32> to tensor<4xf32>
  %empty = tensor.empty() : tensor<4xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<4xf32>, tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%u: f32, %v: f32, %z: f32):
    %s = arith.addf %u, %v : f32
    linalg.yield %s : f32
  } -> tensor<4xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%sum : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%u: f32, %z: f32):
    %zero = arith.constant 0.0 : f32
    %relu = arith.maximumf %u, %zero : f32
    linalg.yield %relu : f32
  } -> tensor<4xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<4xf32>, memref<4xf32>) -> ()
  return
})mlir";
  const char *matmul = R"mlir(
func.func @entry(%a: memref<2x3xf32>, %b: memref<3x2xf32>, %out: memref<2x2xf32>) {
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%out : memref<2x2xf32>)
  linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x2xf32>) outs(%out : memref<2x2xf32>)
  return
})mlir";
  const char *views = R"mlir(
func.func @entry(%a: memref<2x3xf32>, %out: memref<2xf32>) {
  %view = memref.subview %a[0, 1] [2, 1] [1, 1] : memref<2x3xf32> to memref<2xf32, strided<[3], offset: 1>>
  memref.copy %view, %out : memref<2xf32, strided<[3], offset: 1>> to memref<2xf32>
  return
})mlir";
  const char *compare = R"mlir(
func.func @entry(%a: memref<4xf32>, %out: memref<4xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%a : memref<4xf32>) outs(%out : memref<4xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %p = arith.cmpf ugt, %x, %zero : f32
    %y = arith.select %p, %one, %zero : f32
    linalg.yield %y : f32
  }
  return
})mlir";
  const char *integers = R"mlir(
func.func @entry(%a: memref<4xi32>, %b: memref<4xi32>, %out: memref<4xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%a, %b : memref<4xi32>, memref<4xi32>) outs(%out : memref<4xi32>) {
  ^bb0(%x: i32, %y: i32, %unused: i32):
    %sum = arith.addi %x, %y : i32
    linalg.yield %sum : i32
  }
  return
})mlir";
  const char *dynamic = R"mlir(
func.func @entry(%a: memref<?xf32>, %out: memref<?xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%a : memref<?xf32>) outs(%out : memref<?xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %e = math.erf %x : f32
    %s = math.sin %e : f32
    %c = math.cos %s : f32
    linalg.yield %c : f32
  }
  return
})mlir";
  const char *constant = R"mlir(
memref.global "private" constant @values : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]>
func.func @entry(%out: memref<4xf32>) {
  %values = memref.get_global @values : memref<4xf32>
  memref.copy %values, %out : memref<4xf32> to memref<4xf32>
  return
})mlir";
  const char *attention = R"mlir(
func.func @entry(%q: memref<1x2x2x2xf32>, %k: memref<1x2x2x2xf32>, %v: memref<1x2x2x3xf32>, %out: memref<1x2x2x3xf32>) {
  %qt = bufferization.to_tensor %q restrict writable : memref<1x2x2x2xf32> to tensor<1x2x2x2xf32>
  %kt = bufferization.to_tensor %k restrict writable : memref<1x2x2x2xf32> to tensor<1x2x2x2xf32>
  %vt = bufferization.to_tensor %v restrict writable : memref<1x2x2x3xf32> to tensor<1x2x2x3xf32>
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 0.5 : f32
  %empty = tensor.empty() : tensor<1x2x2x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  %result = linalgx.attention <{indexing_maps = [affine_map<(b,h,m,k1,k2,n)->(b,h,m,k1)>, affine_map<(b,h,m,k1,k2,n)->(b,h,k2,k1)>, affine_map<(b,h,m,k1,k2,n)->(b,h,k2,n)>, affine_map<(b,h,m,k1,k2,n)->()>, affine_map<(b,h,m,k1,k2,n)->(b,h,m,n)>]}> ins(%qt, %kt, %vt, %scale : tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x3xf32>, f32) outs(%init : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x2x2x3xf32>, memref<1x2x2x3xf32>) -> ()
  return
})mlir";
  char deviceName[256] = {};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName,
                  nullptr);
  std::cout << deviceName << '\n';
  cl_command_queue_properties supported = 0;
  clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(supported),
                  &supported, nullptr);
  bool ok = true;
  for (auto properties :
       {cl_command_queue_properties(0),
        cl_command_queue_properties(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)}) {
    if (properties && !(supported & properties)) continue;
    auto queue = clCreateCommandQueue(context, device, properties, &status);
    if (status != CL_SUCCESS) {
      ok = false;
      break;
    }
    ok &= runBinary(context, device, queue);
    std::cout << "queue properties " << properties << '\n';
    ok &= run<float>(ctx, context, queue, fusion,
                     {{-3, 1, 2, -4}, {1, 2, 3, 5}}, {0, 3, 5, 1});
    ok &=
        run<float>(ctx, context, queue, matmul,
                   {{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}}, {22, 28, 49, 64});
    ok &= run<float>(ctx, context, queue, views, {{1, 2, 3, 4, 5, 6}}, {2, 5});
    ok &= run<float>(ctx, context, queue, compare, {{NAN, 1, 0, -1}},
                     {1, 1, 0, 0});
    ok &= run<float>(ctx, context, queue, fusion,
                     {{NAN, 1, 2, -4}, {0, 2, 3, 5}}, {NAN, 3, 5, 1});
    ok &= run<int32_t>(ctx, context, queue, integers,
                       {{INT_MAX, INT_MIN, -2, 0}, {1, -1, -2, 0}},
                       {INT_MIN, INT_MAX, -4, 0});
    std::vector<float> mathExpected;
    for (float x : {-1.0f, 0.0f, 0.5f, 1.0f})
      mathExpected.push_back(std::cos(std::sin(std::erf(x))));
    ok &= run<float>(ctx, context, queue, dynamic, {{-1, 0, 0.5, 1}},
                     mathExpected);
    ok &= run<float>(ctx, context, queue, dynamic, {{0}}, {1});
    ok &= run<float>(ctx, context, queue, constant, {}, {1, 2, 3, 4});
    ok &= run<float>(ctx, context, queue, attention,
                     {std::vector<float>(8, 0),
                      std::vector<float>(8, 0),
                      {0, 2, 4, 2, 4, 6, 4, 6, 8, 8, 10, 12}},
                     {1, 3, 5, 1, 3, 5, 6, 8, 10, 6, 8, 10});
    clReleaseCommandQueue(queue);
  }
  clReleaseContext(context);
  if (ok)
    std::cout << "OpenCL compilation, dynamic shapes, constants and event "
                 "ordering passed\n";
  return ok ? 0 : 1;
}
