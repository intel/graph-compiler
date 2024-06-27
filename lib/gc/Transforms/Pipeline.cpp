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
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
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
#include "gc/Dialect/Linalgx/IR/LinalgxDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Passes.h"

namespace mlir::gc {
#define GEN_PASS_DEF_LINALGLOWERTOLOOP
#include "gc/Transforms/Passes.h.inc"
struct LinalgLowerToLoop
    : public impl::LinalgLowerToLoopBase<LinalgLowerToLoop> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](linalg::LinalgOp linalgOp) {
      rewriter.setInsertionPoint(linalgOp);
      if (linalgOp->getParentOfType<scf::ForallOp>() ||
          linalgOp->getParentOfType<scf::ParallelOp>()) {
        auto loops = linalgOpToLoops(rewriter, linalgOp);
        if (failed(loops)) {
          llvm::outs() << "Failed to convert to parallel loops\n";
          return;
        }
        rewriter.eraseOp(linalgOp);
      } else {
        auto loops = linalgOpToParallelLoops(rewriter, linalgOp);
        if (failed(loops)) {
          llvm::outs() << "Failed to convert to parallel loops\n";
          return;
        }
        rewriter.eraseOp(linalgOp);
      }
    });
  }
};
#undef GEN_PASS_DEF_LINALGLOWERTOLOOP

void populateCleanUpPasses(mlir::PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  // pm.addPass(createLoopInvariantSubsetHoistingPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
}

// linalg + linalgX + tensor
void populateFrontendPasses(mlir::PassManager &pm) {
  pm.addPass(createConvertOneDNNGraphToLinalg());
  PrintIRPassOptions option{"Frontend passes result"};
  pm.addPass(createPrintIRPass(option));
  populateCleanUpPasses(pm);
}

// scf + arith + math + vector + tensor + linalg.brgemm + tensor.pack/unpack
void populateTensorPasses(mlir::PassManager &pm) {
  // todo: padding propagation pass
  // todo: layout propagation pass
  // todo: tensor constant propagation pass
  pm.addNestedPass<func::FuncOp>(createDeepTileContractionNamedOp());
  // todo: fine-grain fusion pass
  // todo: lower linalg to arith/math on virtual vector pass

  // REMOVE this pass after the above passes are added. Currently we add this
  // pass to make the pipeline work properly
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"Tensor passes result"};
  pm.addPass(createPrintIRPass(option));
}

// scf + arith + math + vector + tensor + linalg.brgemm
void populateVectorPasses(mlir::PassManager &pm) {
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
void populateBufferizationPasses(mlir::PassManager &pm) {
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
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"Bufferization passes result"};
  pm.addPass(createPrintIRPass(option));
}

// scf + arith + math + vector + memref + func/microkernel
void populateMicroKernelPasses(mlir::PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      mlir::microkernel::createConvertLinalgToMicrokernel());
  pm.addPass(mlir::microkernel::createEarlyDispatchMicrokernel());
  pm.addPass(mlir::microkernel::createConvertMicrokernelToDnnlFunc());
  pm.addPass(mlir::microkernel::createMergeBranchMicrokernelContext());
  pm.addPass(mlir::microkernel::createMicrokernelInvariantCodeMotion());
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"MicroKernel passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateCPURuntimePasses(mlir::PassManager &pm) {
  // todo: flatten nested parallel pass to support coarse-grain usion
  // remove this pass after we add FlattenNestedParallel
  pm.addPass(createForallToParallelLoopPass());
  pm.addPass(createConvertSCFToOpenMPPass());
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"CPURuntime passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateLoweringToLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(createLowerAffinePass());
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
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"LoweringToLLVM passes result"};
  pm.addPass(createPrintIRPass(option));
}

void populateLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  populateLoweringToLLVMPasses(pm);
  populateCleanUpPasses(pm);
  PrintIRPassOptions option{"LLVM passes result"};
  pm.addPass(createPrintIRPass(option));
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
  populateMicroKernelPasses(pm);
  // REMOVE this pass after the TensorPasses are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addPass(createLinalgLowerToLoop());
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
    // TODO(longsheng): add a option to
    // disable threading and enable pm.enableIRPrinting();
    if (failed(pm.run(op)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::gc
