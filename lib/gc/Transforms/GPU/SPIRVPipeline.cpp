// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/Transforms/Passes.h"
#include "gc/Utils/Transform.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::gc {
namespace {
struct PrepareSPIRV : PassWrapper<PrepareSPIRV, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareSPIRV)

  void runOnOperation() override {
    IRRewriter rw(&getContext());
    auto mod = getOperation();
    mod.walk([&](memref::AllocOp alloc) {
      if (alloc->getParentOfType<gpu::GPUModuleOp>()) return;
      rw.setInsertionPoint(alloc);
      auto buffer = gpu::AllocOp::create(
          rw, alloc.getLoc(), alloc.getType(), Type(), ValueRange{},
          alloc.getDynamicSizes(), alloc.getSymbolOperands(), false);
      rw.replaceOp(alloc, buffer.getMemref());
    });
    mod.walk([&](memref::DeallocOp dealloc) {
      if (dealloc->getParentOfType<gpu::GPUModuleOp>()) return;
      rw.setInsertionPoint(dealloc);
      gpu::DeallocOp::create(rw, dealloc.getLoc(), Type(), ValueRange{},
                             dealloc.getMemref());
      rw.eraseOp(dealloc);
    });
    mod.walk([&](gpu::GPUFuncOp fn) {
      fn->setAttr(spirv::getEntryPointABIAttrName(),
                  spirv::getEntryPointABIAttr(&getContext(), {}));
    });
    auto env = spirv::TargetEnvAttr::get(
        spirv::VerCapExtAttr::get(
            spirv::Version::V_1_0,
            {spirv::Capability::Kernel, spirv::Capability::Addresses,
             spirv::Capability::Int8, spirv::Capability::Int16,
             spirv::Capability::Int64, spirv::Capability::Float16,
             spirv::Capability::Float16Buffer, spirv::Capability::Float64},
            {}, &getContext()),
        spirv::getDefaultResourceLimits(&getContext()));
    mod->setAttr(spirv::getTargetEnvAttrName(), env);
    mod.walk([&](gpu::GPUModuleOp kernel) {
      kernel.setTargetsAttr(rw.getArrayAttr({env}));
    });
    SmallVector<StringRef> previous;
    mod.walk([&](gpu::LaunchFuncOp launch) {
      auto name = launch.getKernelModuleName().getValue();
      KernelAttrs(mod, name).setDepends(previous);
      previous.push_back(name);
    });
  }
};
} // namespace

void populateSPIRVPipeline(OpPassManager &pm,
                           const GPUPipelineOptions &options) {
  populateOclSourcePipeline(pm);
  bufferization::buildBufferDeallocationPipeline(pm);
  pm.addPass(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertParallelLoopToGpuPass());
  pm.addNestedPass<func::FuncOp>(createGpuLaunchSinkIndexComputationsPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(std::make_unique<PrepareSPIRV>());
  pm.addPass(createMemrefCopyToGpu());
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToSPIRVPass());
  auto convert = createConvertGPUToSPIRVPass();
  if (failed(convert->initializeOptions(
          "use-64bit-index=true", [](const Twine &) { return failure(); })))
    llvm_unreachable("Invalid SPIR-V conversion options");
  pm.addPass(std::move(convert));
  auto &spv = pm.nest<gpu::GPUModuleOp>().nest<spirv::ModuleOp>();
  spv.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spv.addPass(spirv::createSPIRVUpdateVCEPass());
  pm.addPass(createGpuModuleToBinaryPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  GpuToLLVMConversionPassOptions host;
  host.kernelBarePtrCallConv = true;
  pm.addPass(createGpuToLLVMConversionPass(host));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createGpuToGcGpu({options.wait}));
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}
} // namespace mlir::gc
