// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/GpuCompiler.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
struct RecordingRuntime : gc::gpu::GpuRuntime<int, int, void> {
  gc::gpu::DeviceInfo info;
  mlir::MLIRContext context{mlir::gc::getDialectRegistry()};
  unsigned kernels = 0, launches = 0, waits = 0;
  unsigned allocations = 0, frees = 0, copies = 0;
  bool valid = true;
  bool expectMath = true;

  RecordingRuntime() { context.loadAllAvailableDialects(); }

  const gc::gpu::DeviceInfo &getDeviceInfo() const override { return info; }
  Kernel *createKernel(const uint8_t *data, size_t size, const char *name,
                       uint32_t nArgs, gc::gpu::ArgType *types) override {
    llvm::SmallVector<uint32_t> words(size / sizeof(uint32_t));
    if (size % sizeof(uint32_t) || words.size() < 5) {
      valid = false;
      return new int(0);
    }
    std::memcpy(words.data(), data, size);
    auto module = mlir::spirv::deserialize(words, &context);
    valid &= module && mlir::succeeded(mlir::verify(*module));
    bool entry = false;
    bool math = false;
    if (module)
      module->walk(
          [&](mlir::spirv::EntryPointOp op) { entry |= op.getFn() == name; });
    if (module) module->walk([&](mlir::spirv::CLExpOp) { math = true; });
    valid &= entry && math == expectMath;
    for (unsigned i = 0; i < nArgs; ++i)
      valid &= types[i] == gc::gpu::ArgType::PTR ||
               types[i] == gc::gpu::ArgType::F32;
    ++kernels;
    return new int(nArgs);
  }
  static void destroyKernel(Kernel *kernel) { delete kernel; }
  Event *launch(Kernel *kernel, void **args, uint32_t nArgs, uint32_t gridX,
                uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                uint32_t blockY, uint32_t blockZ, Event *const *,
                uint32_t nDeps) override {
    valid &= nArgs == unsigned(*kernel) && gridX * gridY * gridZ == 4 &&
             blockX * blockY * blockZ == 1;
    if (launches) valid &= nDeps > 0;
    if (nArgs == 3)
      for (unsigned i = 0; i < nArgs; ++i)
        valid &= *static_cast<void **>(args[i]) != nullptr;
    ++launches;
    return new int(0);
  }
  void *allocDevice(size_t size, size_t) override {
    ++allocations;
    return std::malloc(size);
  }
  void *allocShared(size_t size, size_t alignment) override {
    return allocDevice(size, alignment);
  }
  void *allocHost(size_t size, size_t alignment) override {
    return allocDevice(size, alignment);
  }
  void free(void *ptr) override {
    ++frees;
    std::free(ptr);
  }
  Event *memcpy(const void *src, void *dst, size_t size, Event *const *,
                uint32_t) override {
    std::memcpy(dst, src, size);
    ++copies;
    return new int(0);
  }
  void wait(Event *const *events, uint32_t count) override {
    for (unsigned i = 0; i < count; ++i) delete events[i];
    waits += count;
  }
};
} // namespace

int main() {
  const char *elementwise = R"mlir(
func.func @entry(%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>) {
  %x = bufferization.to_tensor %a restrict writable : memref<4xf32> to tensor<4xf32>
  %y = bufferization.to_tensor %b restrict writable : memref<4xf32> to tensor<4xf32>
  %empty = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<4xf32>, tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%u: f32, %v: f32, %unused: f32):
    %sum = arith.addf %u, %v : f32
    %exp = math.exp %sum : f32
    linalg.yield %exp : f32
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
  for (auto ir : {elementwise, matmul}) {
    for (bool wait : {true, false}) {
      RecordingRuntime runtime;
      runtime.expectMath = ir == elementwise;
      unsigned expectedKernels = ir == elementwise ? 1 : 2;
      gc::gpu::CompileOptions options;
      options.spirv = true;
      options.wait = wait;
      auto program = gc::gpu::GpuCompiler::get()
                         .compile<RecordingRuntime>(ir, options)
                         .get();
      if (!program || (wait && program->eventNum())) return 1;
      float a[6]{}, b[6]{}, out[4]{};
      gc::gpu::SmallBuilder<> builder;
      int64_t shape = 4, stride = 1;
      if (ir == matmul) {
        int64_t lhs[] = {2, 3}, rhs[] = {3, 2}, dst[] = {2, 2};
        int64_t lhsStrides[] = {3, 1}, rhsStrides[] = {2, 1};
        builder.storeMemref(a, 2, lhs, lhsStrides);
        builder.storeMemref(b, 2, rhs, rhsStrides);
        builder.storeMemref(out, 2, dst, rhsStrides);
      } else
        for (void *ptr : {static_cast<void *>(a), static_cast<void *>(b),
                          static_cast<void *>(out)})
          builder.storeMemref(ptr, 1, &shape, &stride);
      auto events = program->main(builder.build(*program), &runtime);
      if (events) runtime.wait(events, program->eventNum());
      if (!runtime.valid || runtime.kernels != expectedKernels ||
          runtime.launches != expectedKernels ||
          runtime.waits != runtime.launches + runtime.copies ||
          runtime.allocations != runtime.frees)
        return 1;
#if !GC_ENABLE_IGC
      for (auto dialect : runtime.context.getAvailableDialects())
        if (dialect == "xegpu" || dialect == "xevm") return 1;
#endif
    }
  }
  std::cout << "SPIR-V graph compilation and JIT dispatch passed\n";
}
