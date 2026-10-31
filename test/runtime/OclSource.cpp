// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/Runtime/OclSourceModule.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace mlir;
using namespace mlir::gc::gpu;

template <typename T>
static bool run(MLIRContext &ctx, cl_context context, cl_device_id device,
                cl_command_queue queue, const char *ir,
                std::vector<std::vector<T>> inputs,
                const std::vector<T> &expected) {
  auto module = parseSourceString<ModuleOp>(ir, &ctx);
  if (!module) return false;
  auto compiled = OclSourceModule::build(*module, device, context);
  if (!compiled) {
    std::cerr << llvm::toString(compiled.takeError()) << '\n';
    return false;
  }
  inputs.emplace_back(expected.size(), -999.0f);
  std::vector<cl_mem> args;
  for (auto &input : inputs) {
    cl_int status;
    auto buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       input.size() * sizeof(T), input.data(), &status);
    if (status != CL_SUCCESS) return false;
    args.push_back(buffer);
  }
  cl_event ready = nullptr, done = nullptr;
  clEnqueueMarkerWithWaitList(queue, 0, nullptr, &ready);
  auto err = compiled->invoke(queue, args, {ready}, &done);
  clReleaseEvent(ready);
  bool ok = !err;
  if (err) std::cerr << llvm::toString(std::move(err)) << '\n';
  if (ok) {
    auto &out = inputs.back();
    ok = clEnqueueReadBuffer(queue, args.back(), CL_TRUE, 0,
                             out.size() * sizeof(T), out.data(), 1, &done,
                             nullptr) == CL_SUCCESS;
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
  }
  if (done) clReleaseEvent(done);
  for (auto arg : args) clReleaseMemObject(arg);
  return ok;
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
  MLIRContext ctx(gc::getDialectRegistry());
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
    std::cout << "queue properties " << properties << '\n';
    ok &= run<float>(ctx, context, device, queue, fusion,
                     {{-3, 1, 2, -4}, {1, 2, 3, 5}}, {0, 3, 5, 1});
    ok &=
        run<float>(ctx, context, device, queue, matmul,
                   {{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}}, {22, 28, 49, 64});
    ok &= run<float>(ctx, context, device, queue, views, {{1, 2, 3, 4, 5, 6}},
                     {2, 5});
    ok &= run<float>(ctx, context, device, queue, compare, {{NAN, 1, 0, -1}},
                     {1, 1, 0, 0});
    ok &= run<float>(ctx, context, device, queue, fusion,
                     {{NAN, 1, 2, -4}, {0, 2, 3, 5}}, {NAN, 3, 5, 1});
    ok &= run<int32_t>(ctx, context, device, queue, integers,
                       {{INT_MAX, INT_MIN, -2, 0}, {1, -1, -2, 0}},
                       {INT_MIN, INT_MAX, -4, 0});
    ok &= run<float>(ctx, context, device, queue, attention,
                     {std::vector<float>(8, 0),
                      std::vector<float>(8, 0),
                      {0, 2, 4, 2, 4, 6, 4, 6, 8, 8, 10, 12}},
                     {1, 3, 5, 1, 3, 5, 6, 8, 10, 6, 8, 10});
    clReleaseCommandQueue(queue);
  }
  clReleaseContext(context);
  if (ok)
    std::cout
        << "OpenCL source fusion, matmul, views and event ordering passed\n";
  return ok ? 0 : 1;
}
