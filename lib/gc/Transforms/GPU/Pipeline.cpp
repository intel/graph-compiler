//===- Pipeline.cpp - Graph Compiler GPU pipeline ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "gc/Transforms/Passes.h"
#include "gc/Utils/Transform.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir::gc {

DialectRegistry &getDialectRegistry() {
  static mlir::DialectRegistry registry = []() {
    mlir::registerAllPasses();
    mlir::gc::registerGraphCompilerPasses();
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerConvertXeVMToLLVMInterface(registry);
    mlir::registerXeVMDialectTranslation(registry);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return registry;
  }();
  return registry;
}

void populateGPUPipeline(OpPassManager &pm,
                         const GPUPipelineOptions &pipelineOpts) {
  auto phase = [&pm, &pipelineOpts](const char *name,
                                    std::function<void()> func) {
    func();
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (pipelineOpts.dump)
      pm.addPass(createPrintIRPass({name}));
  };

  GpuDevicePropsOptions deviceProps;
  if (pipelineOpts.deviceProps) {
    deviceProps = *pipelineOpts.deviceProps;
  }
  phase("Initial", [&]() { pm.addPass(createGpuDeviceProps(deviceProps)); });
  phase("Tiling", [&]() {
    pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
    pm.addNestedPass<func::FuncOp>(createGpuTilingAndFusion());
  });

  phase("Vectorization", [&]() {
    pm.addNestedPass<func::FuncOp>(createVectorize());
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createLoopInvariantSubsetHoistingPass());
  });

  // Bufferization
  phase("Bufferization", [&]() {
    bufferization::OneShotBufferizePassOptions opts;
    opts.allowReturnAllocsFromLoops = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(bufferization::createOneShotBufferizePass(opts));
    opts.allowReturnAllocsFromLoops = false;
    pm.addPass(bufferization::createOneShotBufferizePass(opts));

    pm.addPass(bufferization::createEmptyTensorEliminationPass());
    pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
    pm.addPass(bufferization::createBufferResultsToOutParamsPass(
        {true, true, true, true}));
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
  });

  phase("KernelOutlining", [&]() {
    pm.addPass(createGpuKernelOutline());
    pm.addNestedPass<func::FuncOp>(createAddContextArg());
  });

  phase("VectorToXegpu", [&]() {
    pm.addPass(createConvertVectorToXeGPU());
    pm.addPass(memref::createExpandStridedMetadataPass());
  });

  phase("XeGpu", [&]() {
    gpu::GPUToXeVMPipelineOptions opts;
    opts.use64bitIndex = true;
    opts.binaryFormat = "binary";
    opts.zebinChip = deviceProps.arch;
    opts.optLevel = 3;
    gpu::buildLowerToXeVMPassPipeline(pm, opts);
  });

  phase("GpuToGpuOcl",
        [&]() { pm.addPass(createGpuToGpuOcl({pipelineOpts.callFinish})); });

  // phase("XeGpu", [&]() {
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUWgToSgDistribute());
  //   pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  //   pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());
  //   pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());

  //   {
  //     xegpu::XeGPUPropagateLayoutOptions opts;
  //     opts.layoutKind = "inst";
  //     pm.addNestedPass<gpu::GPUModuleOp>(
  //         xegpu::createXeGPUPropagateLayout(opts));
  //   }
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUBlocking());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());

  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUPropagateLayout());
  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUPeepHoleOptimizer());
  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUPropagateLayout());

  //   // pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUFoldAliasOps());
  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUSubgroupDistribute());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());

  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(createLoopInvariantCodeMotionPass());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(xegpu::createXeGPUVectorLinearize());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToXeVM());
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createConvertXeGPUToXeVMPass());
  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToLLVMSPVOps({true}));
  //   // pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  //   //
  //   pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
  // });
}

void registerGPUPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>(
      "gc-gpu-pipeline", "Graph Compiler GPU pipeline", populateGPUPipeline);
}

} // namespace mlir::gc
