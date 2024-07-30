//===- Pipeline.cpp - Graph Compiler all-in-one pipeline --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Transforms/Passes.h"

namespace mlir::gc {

// linalg + linalgX + tensor
void populateFrontendPasses(mlir::OpPassManager &pm) {
  pm.addPass(createConvertOneDNNGraphToLinalg());
}

// scf + arith + math + vector + tensor + linalg.brgemm + tensor.pack/unpack
void populateTensorPasses(mlir::OpPassManager &pm) {
  // todo: padding propagation pass
  // todo: layout propagation pass
  // todo: tensor constant propagation pass
  // todo: linalg.matmul lowering to (scf.loop + linalg.brgemm) pass
  // Fine-grain fusion pass
  pm.addNestedPass<func::FuncOp>(createIterativeTilingAndFusion());
  // todo: lower linalg to arith/math on virtual vector pass

  // REMOVE this pass after the above passes are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
}

// scf + arith + math + vector + tensor + linalg.brgemm
void populateVectorPasses(mlir::OpPassManager &pm) {
  // Do promotion for math / arith ops
  pm.addNestedPass<func::FuncOp>(math::createMathLegalizeToF32());
  // sourceTypeStrs can be extended
  arith::ArithEmulateUnsupportedFloatsOptions options;
  std::array<std::string, 1> typeStr = {"bf16"};
  options.sourceTypeStrs = typeStr;
  options.targetTypeStr = "f32";
  pm.addNestedPass<func::FuncOp>(
      arith::createArithEmulateUnsupportedFloats(options));
  // Bf16 cast elimilation pass
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  // oneDNN graph spec
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  // todo: lower to physical vector pass, device dependent pass
}

// scf + arith + math + vector + memref + linalg.brgemm
void populateBufferizationPasses(mlir::OpPassManager &pm) {
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(createCSEPass());
  bufferization::BufferResultsToOutParamsOpts opt{};
  opt.hoistStaticAllocs = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(opt));
  // todo: buffer schedule pass
  // todo: Need to improve this pass to support nested parallel.
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferDeallocationPass());
  pm.addPass(createBufferizationToMemRefPass());
}

// scf + arith + math + vector + memref + func/microkernel
void populateMicroKernelPasses(mlir::OpPassManager &pm) {
  // todo: ConvertLinalgToMicrokernel pass
  // todo: CleanupInvalidMicrokernel pass
  // todo: InvariantMicrokernelMotion pass
  // todo: ConvertMicrokernelToDnnlFunc to lower brgemm to dnnl call
  // todo: ConvertMicrokernelToXsmm, to lower brgemm to libxsmm call
  // todo: LowerMicrokernel pass
  // todo: DispatchMicrokernel
}

void populateCPURuntimePasses(mlir::OpPassManager &pm) {
  // todo: flatten nested parallel pass to support coarse-grain usion
  // remove this pass after we add FlattenNestedParallel
  pm.addPass(createConvertSCFToOpenMPPass());
}

void populateLoweringToLLVMPasses(mlir::OpPassManager &pm) {
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(cpuruntime::createCPURuntimeToLLVM());
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createSymbolDCEPass());
}

void populateLLVMPasses(mlir::OpPassManager &pm) {
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  populateLoweringToLLVMPasses(pm);
}

void populateCPUPipeline(mlir::OpPassManager &pm) {
  // front-end, oneDNN graph dialect
  populateFrontendPasses(pm);
  // middle-end, LinalgX/Linalg/tensor dialects
  populateTensorPasses(pm);
  // middle-end, arith/math/vector dialects
  populateVectorPasses(pm);
  // back-end, arith/math/vector/memref dialects
  populateBufferizationPasses(pm);
  // REMOVE this pass after the TensorPasses are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
  populateMicroKernelPasses(pm);
  populateCPURuntimePasses(pm);
  // // back-end, llvm dialect
  populateLLVMPasses(pm);
}

void registerCPUPipeline() {
  PassPipelineRegistration<>("gc-cpu-pipeline",
                             "The CPU pipeline for Graph Compiler",
                             populateCPUPipeline);
}

} // namespace mlir::gc
