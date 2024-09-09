#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include <imex/Transforms/Passes.h>
#include <imex/Conversion/Passes.h>

#include <string>

#include "gc/Transforms/Passes.h"

namespace mlir::gc {

void populateGPUPipeline(mlir::OpPassManager &pm) {
  IterativeTilingAndFusionOptions tilingOpts;
  std::string tilingSizes = "matmul:{16,16}";
  tilingOpts.defaultTileSize = tilingSizes;
  pm.addNestedPass<func::FuncOp>(createIterativeTilingAndFusion(tilingOpts));

  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(options));

  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());
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
  pm.addNestedPass<func::FuncOp>(
    createLinalgToXeGPU({/*kTile=*/16, /*stages=*/1, /*dpasTiles=*/{8, 16, 16}}));

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(xegpu::createXeGPUFoldAliasOps());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());

  pm.addNestedPass<func::FuncOp>(imex::createInsertGPUAllocsPass("opencl"));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(memref::createNormalizeMemRefsPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(imex::createSetSPIRVCapabilitiesPass());
  pm.addNestedPass<gpu::GPUModuleOp>(imex::createSetSPIRVAbiAttributePass("opencl"));
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
  pm.addPass(imex::createConvertGPUToGPUXPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(imex::createConvertGPUXToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

}

void registerGPUPipeline() {
  PassPipelineRegistration<>("gc-gpu-pipeline",
                             "The GPU pipeline for Graph Compiler with IMEX",
                             populateGPUPipeline);
}

} // namespace mlir::gc
