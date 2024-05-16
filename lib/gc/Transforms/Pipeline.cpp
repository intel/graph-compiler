//===- Pipeline.cpp - Graph Compiler all-in-one pipeline --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Transforms/Passes.h"

namespace mlir::gc {

void populateFrontendPasses(mlir::PassManager &pm) {
  // pm.addPass(onednn_graph::createConvertOneDNNGraphToLinalg());
}
// linalg + linalgX + tensor ==> GC V1 GIR

void populateTensorPasses(mlir::PassManager &pm) {
  // + padding propagation pass, upstream-able 127x127 -> tilling size:32
  // ->padding to 128x128
  // + layout propagation pass, upstream-able 4x32x4x32 ->
  // tensor.pack/tensor.unpack
  // + tensor constant propagation pass, down-stream pass, designed to support
  // oneDNN graph spec
  // + linalg.matmul lowering to (scf.loop + linalg.brgemm) pass, upstream-able
  // + fine-grain fusion pass, upstream-able -> scf.for + linalgx.mask
  // + lower linalg to arith/math on virtual vector pass, up-streamable

  // REMOVE this pass after the above passes are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
}
// scf + arith + math + vector + tensor + linalg.brgemm + tensor.pack/unpack ==>
// GC V1 TIR

void populateVectorPasses(mlir::PassManager &pm) {
  // + bf16 promotion pass, down-stream pass, device dependent pass, maybe can
  // upstream
  // + bf16 cast elimilation pass, down-stream pass, fast-math kind pass,
  // designed to support oneDNN graph spec
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  // + lower to physical vector pass, down-stream pass, device dependent pass,
  // maybe can upstream
}
// scf + arith + math + vector + tensor + linalg.brgemm

void populateBufferizationPasses(mlir::PassManager &pm) {
  bufferization::OneShotBufferizationOptions options;
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(createCSEPass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createBufferizationBufferizePass());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createFinalizingBufferizePass());
  bufferization::BufferResultsToOutParamsOpts opt{};
  opt.hoistStaticAllocs = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(opt));
  // + buffer schedule pass, down-stream pass, to migrate buffer reschedule pass
  // from GC V1.
  pm.addNestedPass<func::FuncOp>(
      bufferization::createBufferHoistingPass()); // Need to improve this pass
                                                  // to support thread-local
                                                  // allocator.
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferDeallocationPass());
  pm.addPass(createBufferizationToMemRefPass());
}
// scf + arith + math + vector + memref + linalg.brgemm

void populateMicroKernelPasses(mlir::PassManager &pm) {
  // + ConvertLinalgToMicrokernel pass, upstream-able,
  // + CleanupInvalidMicrokernel pass, upstream-able
  // + InvariantMicrokernelMotion pass, upstream-able
  // + ConvertMicrokernelToDnnlFunc, down-stream pass, to lower brgemm to dnnl
  // call
  // + ConvertMicrokernelToXsmm, down-stream pass, to lower brgemm to libxsmm
  // call
  // + LowerMicrokernel pass, upstream-able
  // + DispatchMicrokernel, down-stream pass
}
// scf + arith + math + vector + memref + func/microkernel

void populateCPURuntimePasses(mlir::PassManager &pm) {
  // + flatten nested parallel pass, down-stream pass, to support coarse-grain
  // fusion
  // remove this pass after we add FlattenNestedParallel
  pm.addPass(createConvertSCFToOpenMPPass());
}

void populateLoweringToLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(cpuruntime::createCPURuntimeToLLVM());
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createSymbolDCEPass());
}

void populateLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  populateLoweringToLLVMPasses(pm);
}

void populateCPUPipeline(mlir::PassManager &pm) {
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

#define GEN_PASS_DEF_GCCPUPIPELINE
#include "gc/Transforms/Passes.h.inc"
namespace {

class GCCPUPipeline : public impl::GCCPUPipelineBase<GCCPUPipeline> {
public:
  friend struct PassHelper;
  using impl::GCCPUPipelineBase<GCCPUPipeline>::GCCPUPipelineBase;
  void runOnOperation() final {
    auto op = getOperation();
    PassManager pm{op->getContext()};
    populateCPUPipeline(pm);
    if (failed(pm.run(op)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::gc
