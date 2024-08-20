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
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Passes.h"
#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#endif

namespace mlir::gc {

void populateCleanUpPasses(mlir::OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  // pm.addPass(createLoopInvariantSubsetHoistingPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
}

// linalg + linalgX + tensor
void populateFrontendPasses(mlir::OpPassManager &pm) {
#ifdef GC_HAS_ONEDNN_DIALECT
  pm.addPass(createConvertOneDNNGraphToLinalg());
#endif
}

// scf + arith + math + vector + tensor + linalg.brgemm + tensor.pack/unpack
void populateTensorPasses(mlir::OpPassManager &pm) {
  // todo: padding propagation pass
  // todo: layout propagation pass
  pm.addPass(createPropagateLayoutOnNamedOps());
  pm.addPass(createPostProcessPackUnpack());
  // todo: tensor constant propagation pass
  // linalg.matmul lowering to (scf.loop + linalg.brgemm) pass
  pm.addNestedPass<func::FuncOp>(createDeepTileContractionNamedOp());

  // Fine-grain fusion pass
  pm.addNestedPass<func::FuncOp>(createIterativeTilingAndFusion());
  // todo: fine-grain fusion pass
  // todo: lower linalg to arith/math on virtual vector pass

  // REMOVE this pass after the above passes are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createControlFlowSinkPass());
  // TODO(yifei): remove lower pack here
  pm.addPass(createLowerPackUnpack());
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"Tensor passes result"};
  pm.addPass(createPrintIRPass(option));
}

// scf + arith + math + vector + tensor + linalg.brgemm
void populateVectorPasses(mlir::OpPassManager &pm) {
  // todo: bf16 promotion pass, device dependent pass
  // todo: bf16 cast elimilation pass, fast-math kind pass, designed to support
  // oneDNN graph spec
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  // todo: lower to physical vector pass, device dependent pass
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"Vector passes result"};
  pm.addPass(createPrintIRPass(option));
}

// scf + arith + math + vector + memref + linalg.brgemm
void populateBufferizationPasses(mlir::OpPassManager &pm) {
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.enforceAliasingInvariants = false;
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
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"Bufferization passes result"};
  pm.addPass(createPrintIRPass(option));
}

// scf + arith + math + vector + memref + func/microkernel
void populateMicroKernelPasses(mlir::OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      mlir::microkernel::createConvertLinalgToMicrokernel());
  pm.addPass(mlir::microkernel::createEarlyDispatchMicrokernel());
  pm.addPass(mlir::microkernel::createConvertMicrokernelToDnnlFunc());
  pm.addPass(mlir::microkernel::createMergeBranchMicrokernelContext());
  pm.addPass(mlir::microkernel::createMicrokernelInvariantCodeMotion());
  // pm.addPass(createRemoveDeadValuesPass());
  // pm.addPass(createInlinerPass());
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"MicroKernel passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateCPURuntimePasses(mlir::OpPassManager &pm) {
  // todo: flatten nested parallel pass to support coarse-grain usion
  // remove this pass after we add FlattenNestedParallel
  pm.addPass(createSinkOpIntoInnerLoop());
  pm.addPass(createMergeNestedForall());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createControlFlowSinkPass());
  pm.addPass(createForallToParallelLoopPass());
  pm.addPass(createParallelLoopFusionPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createConvertSCFToOpenMPPass());
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"CPURuntime passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateLoweringToLLVMPasses(mlir::OpPassManager &pm) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertVectorToLLVMPass());
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
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"LoweringToLLVM passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateLLVMPasses(mlir::OpPassManager &pm) {
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  populateLoweringToLLVMPasses(pm);
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"LLVM passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateCPUPipeline(mlir::OpPassManager &pm) {
  // verify the target description attribute
  pm.addPass(createVerifyTargetDescription());
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
  populateMicroKernelPasses(pm);
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  
  populateCPURuntimePasses(pm);
  // back-end, llvm dialect
  populateLLVMPasses(pm);
}

void registerCPUPipeline() {
  PassPipelineRegistration<>("gc-cpu-pipeline",
                             "The CPU pipeline for Graph Compiler",
                             populateCPUPipeline);
}

} // namespace mlir::gc
