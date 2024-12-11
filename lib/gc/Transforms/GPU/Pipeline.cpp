//===- Pipeline.cpp - Graph Compiler GPU pipeline ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "gc/Transforms/Passes.h"

#include "imex/Conversion/Passes.h"
#include "imex/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::gc {

void populateGPUPipeline(OpPassManager &pm,
                         const GPUPipelineOptions &pipelineOpts) {
  if (pipelineOpts.useGpuRuntime) {
    // Add an argument for the GPU context
    pm.addNestedPass<func::FuncOp>(createAddContextArg());
  }

  pm.addPass(createDecomposeTensorOperation());
  pm.addNestedPass<func::FuncOp>(createGpuTilingAndFusion());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(options));

  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createFinalizingBufferizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addPass(memref::createExpandReallocPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(bufferization::createLowerDeallocationsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createBufferizationToMemRefPass());

  pm.addNestedPass<func::FuncOp>(createForallToParallelLoopPass());
  pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createAllocsToSLM());
  pm.addNestedPass<func::FuncOp>(createLinalgToXeGPU(
      {/*kTile=*/16, /*stages=*/1, /*dpasTiles=*/{8, 16, 16}}));
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(xegpu::createXeGPUFoldAliasOps());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());

  imex::InsertGPUAllocsOptions insertGPUAllocsOption{
      /*clientAPI*/ "opencl", /*inRegions*/ false,
      /*isUsmArgs*/ pipelineOpts.isUsmArgs};
  pm.addNestedPass<func::FuncOp>(
      imex::createInsertGPUAllocsPass(insertGPUAllocsOption));
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(imex::createSetSPIRVCapabilitiesPass());
  pm.addNestedPass<gpu::GPUModuleOp>(
      imex::createSetSPIRVAbiAttributePass("opencl"));
  pm.addPass(createLowerAffinePass());
  pm.addPass(imex::createVectorLinearizePass());
  pm.addNestedPass<gpu::GPUModuleOp>(imex::createConvertXeGPUToVCPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(imex::createBF16ToGPUPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertFuncToSPIRVPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToSPIRVPass());
  pm.addPass(imex::createConvertGPUXToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
  pm.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());
  pm.addPass(imex::createSerializeSPIRVPass());
  pm.addPass(createConvertVectorToSCFPass());

  if (!pipelineOpts.useGpuRuntime) {
    pm.addPass(imex::createConvertGPUToGPUXPass());
  }

  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());

  if (pipelineOpts.useGpuRuntime) {
    pm.addPass(createGpuToGpuOcl({pipelineOpts.callFinish}));
  } else {
    pm.addPass(imex::createConvertGPUXToLLVMPass());
  }

  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void registerGPUPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>(
      "gc-gpu-pipeline", "The GPU pipeline for Graph Compiler with IMEX",
      populateGPUPipeline);
}

} // namespace mlir::gc
