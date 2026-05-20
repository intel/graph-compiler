//===- Pipeline.cpp - Graph Compiler GPU pipeline ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

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

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Transform.h"

namespace mlir::gc {

DialectRegistry &getDialectRegistry() {
  static mlir::DialectRegistry registry = []() {
    mlir::registerAllPasses();
    mlir::gc::registerGraphCompilerPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::linalgx::LinalgxDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerXeVMDialectTranslation(registry);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return registry;
  }();
  return registry;
}

static void
addAttentionOptimizationPasses(OpPassManager &pm,
                               const GPUPipelineOptions &pipelineOpts) {
  pm.addPass(createHoistAttentionVLoad());
  if (pipelineOpts.enableAttentionPrefetch)
    pm.addPass(createSetAttentionPrefetch());
  pm.addPass(createSetAttentionLayouts());
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
    pm.addNestedPass<func::FuncOp>(createTileContraction());
    pm.addNestedPass<func::FuncOp>(createTileAttention());
    pm.addNestedPass<func::FuncOp>(createTileParallel());
  });

  phase("Decomposition",
        [&]() { pm.addNestedPass<func::FuncOp>(createDecomposition()); });

  phase("Vectorization", [&]() {
    pm.addNestedPass<func::FuncOp>(createVectorize());
    pm.addNestedPass<func::FuncOp>(createHoistForLoopTransferRead());
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
    pm.addNestedPass<func::FuncOp>(createRemoveAllocs());
  });

  phase("KernelOutlining", [&]() {
    pm.addPass(createGpuKernelOutline());
    pm.addNestedPass<func::FuncOp>(createAddContextArg());
  });

  phase("VectorToXegpu", [&]() {
    pm.addPass(createConvertVectorToXeGPU());
    pm.addPass(createSetGpuFastMath());
    pm.addPass(memref::createExpandStridedMetadataPass());
    addAttentionOptimizationPasses(pm, pipelineOpts);
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createLoopInvariantSubsetHoistingPass());
  });

  phase("XeGpu", [&]() {
    gpu::GPUToXeVMPipelineOptions opts;
    opts.use64bitIndex = true;
    opts.binaryFormat = "binary";
    opts.zebinChip = deviceProps.arch;
    opts.cmdOptions = pipelineOpts.igcCmdOptions;
    opts.optLevel = 3;
    gpu::buildLowerToXeVMPassPipeline(pm, opts);
  });

  phase("GpuToGpuOcl",
        [&]() { pm.addPass(createGpuToGpuOcl({pipelineOpts.callFinish})); });
}

void registerGPUPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>(
      "gc-gpu-pipeline", "Graph Compiler GPU pipeline", populateGPUPipeline);
}

} // namespace mlir::gc
