//===- Pipeline.cpp - Graph Compiler GPU pipeline ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Extensions/AllExtensions.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/IndexedAccessOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/IndexedAccessOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/XeVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Transforms/Passes.h"
#include "gc/Transforms/TensorMaskingOpInterface.h"
#include "gc/Utils/Misc.h"

namespace mlir::gc {

namespace {
struct TruncatingPrintIRPass
    : PassWrapper<TruncatingPrintIRPass, OperationPass<>> {
  std::string label;
  llvm::raw_ostream &stream;
  bool truncate;
  TruncatingPrintIRPass(StringRef label, llvm::raw_ostream &stream,
                        bool truncate)
      : label(label.str()), stream(stream), truncate(truncate) {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TruncatingPrintIRPass)

  void runOnOperation() override {
    stream << "// -----// IR Dump " << label << " //----- //\n";

    if (truncate) {
      std::string buf;
      llvm::raw_string_ostream os(buf);
      getOperation()->print(os);

      constexpr size_t kKeep = 32;
      auto tr = [&](StringRef marker, bool closeParen) {
        std::string result;
        size_t pos = 0;
        while (pos < buf.size()) {
          size_t found = buf.find(marker, pos);
          if (found == std::string::npos) {
            result += buf.substr(pos);
            buf = result;
            return;
          }
          result += buf.substr(pos, found - pos);
          size_t start = found + marker.size();
          size_t end = buf.find('"', start);
          size_t len = end - start;
          result += marker;
          result += buf.substr(start, std::min(len, kKeep));
          if (len > kKeep)
            result += "...<" + std::to_string(len) + " chars>...";
          result += '"';
          if (closeParen) result += ')';
          pos = end + (closeParen ? 2 : 1); // skip closing `"` [and `)`]
        }
        buf = result;
      };
      tr("bin = \"", false);
      tr("_SPIRV(\"", true);
      tr("_Bin(\"", true);
      stream << buf << "\n";
    } else {
      getOperation()->print(stream);
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

DialectRegistry &getDialectRegistry() {
  static mlir::DialectRegistry registry = []() {
    mlir::gc::registerGraphCompilerPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::cf::ControlFlowDialect, mlir::DLTIDialect,
                    mlir::func::FuncDialect, mlir::gpu::GPUDialect,
                    mlir::index::IndexDialect, mlir::linalg::LinalgDialect,
                    mlir::linalgx::LinalgxDialect, mlir::LLVM::LLVMDialect,
                    mlir::math::MathDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::ub::UBDialect, mlir::vector::VectorDialect,
                    mlir::xegpu::XeGPUDialect, mlir::xevm::XeVMDialect>();
    mlir::LLVM::registerInlinerInterface(registry);
    mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::bufferization::registerAllExtensions(registry);
    mlir::bufferization::registerBufferizableOpInterfaceExternalModels(
        registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::func::registerAllExtensions(registry);
    mlir::tensor::registerAllExtensions(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::registerConvertComplexToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::gpu::registerConvertGpuToLLVMInterface(registry);
    mlir::index::registerConvertIndexToLLVMInterface(registry);
    mlir::registerConvertMathToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    mlir::ub::registerConvertUBToLLVMInterface(registry);
    mlir::vector::registerConvertVectorToLLVMInterface(registry);

    mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::gpu::registerIndexedAccessOpInterfaceExternalModels(registry);
    mlir::gpu::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::linalg::registerAllDialectInterfaceImplementations(registry);
    mlir::linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
    mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
    mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::memref::registerDestructurableTypeExternalModels(registry);
    mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
    mlir::tensor::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
    mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerIndexedAccessOpInterfaceExternalModels(registry);
    mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
    mlir::vector::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::xevm::registerXeVMTargetInterfaceExternalModels(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerConvertComplexToLLVMInterface(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerGPUDialectTranslation(registry);

    mlir::gc::registerTensorMaskingOpInterfaceForLinalg(registry);
    mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(
        registry);
    mlir::registerXeVMDialectTranslation(registry);

    // Collect every dialect declared by the pipeline's passes before any
    // asynchronous compilation can start. The context loads this registry
    // synchronously during construction.
    mlir::OpPassManager pm;
    GPUPipelineOptions pipelineOpts;
    populateGPUPipeline(pm, pipelineOpts);
    pm.getDependentDialects(registry);

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
  static bool dumpIr = gcGetEnv("GC_DUMP_IR", false);
  llvm::raw_ostream &dumpStream =
      pipelineOpts.dumpStream ? *pipelineOpts.dumpStream : llvm::dbgs();
  bool dump = pipelineOpts.dump || dumpIr;
  bool truncate = false;
  auto createPrintPass = [&](const char *name) {
    return std::make_unique<TruncatingPrintIRPass>(name, dumpStream, truncate);
  };
  auto phase = [&](const char *name, std::function<void()> func) {
    func();
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (!dump) return;
    pm.addPass(createPrintPass(name));
  };

  GpuDevicePropsOptions deviceProps;
  if (pipelineOpts.deviceProps) {
    deviceProps = *pipelineOpts.deviceProps;
  }
  // Command line options for the IGC compiler. They are stored in the xevm
  // target attached to the kernels and passed to the gpu-module-to-binary pass.
  std::string igcCmdOptions =
      pipelineOpts.igcCmdOptions + " -ze-opt-large-register-file";

  pm.addPass(createCanonicalizerPass());
  if (dump) pm.addPass(createPrintPass("Initial"));

  phase("Preprocess", [&]() {
    pm.addPass(createGpuDeviceProps(deviceProps));
    pm.addNestedPass<func::FuncOp>(createTensorConcatToLinalg());
    pm.addNestedPass<func::FuncOp>(createTensorReshapeToLinalg());
  });

  phase("Tiling", [&]() {
    pm.addNestedPass<func::FuncOp>(createTileContraction());
    pm.addNestedPass<func::FuncOp>(createTileGroupedMatmul());
    pm.addNestedPass<func::FuncOp>(createTileAttention());
    pm.addNestedPass<func::FuncOp>(createTileParallel());
    pm.addNestedPass<func::FuncOp>(createSetKernelDepends());
  });

  phase("Decomposition", [&]() {
    pm.addNestedPass<func::FuncOp>(createDecomposition());
    pm.addPass(createSCFForLoopPeeling());
    pm.addPass(createCanonicalizerPass());
  });

  phase("Padding", [&]() {
    pm.addNestedPass<func::FuncOp>(createApplyPaddingLevel());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
    pm.addNestedPass<func::FuncOp>(createApplyPaddingLevel());
    pm.addNestedPass<func::FuncOp>(createLinalgSpecializeGenericOpsPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    pm.addNestedPass<func::FuncOp>(createHoistStaticForInit());
    pm.addNestedPass<func::FuncOp>(createFoldPadRoundtrip());
  });

  phase("Vectorization", [&]() {
    pm.addNestedPass<func::FuncOp>(createVectorize());
    pm.addNestedPass<func::FuncOp>(createHoistForLoopTransferRead());
    pm.addNestedPass<func::FuncOp>(createPeeledForToIf());
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
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createRemoveAllocs());
    pm.addNestedPass<func::FuncOp>(createDropUnitDims());
  });

  phase("KernelOutlining", [&]() {
    GpuKernelOutlineOptions opts;
    opts.cmdOptions = igcCmdOptions;
    pm.addPass(createGpuKernelOutline(std::move(opts)));
  });

  phase("VectorToXegpu", [&]() {
    pm.addPass(createConvertVectorToXeGPU());
    pm.addPass(createSetGpuFastMath());
    pm.addPass(memref::createExpandStridedMetadataPass());
    addAttentionOptimizationPasses(pm, pipelineOpts);
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createLoopInvariantSubsetHoistingPass());
    pm.addPass(createMemrefCopyToGpu());
    pm.addPass(createSetLayouts());
  });

  truncate = pipelineOpts.truncate;
  phase("XeGpu", [&]() {
    gpu::GPUToXeVMPipelineOptions opts;
    opts.use64bitIndex = true;
    opts.binaryFormat = "binary";
    opts.cmdOptions = igcCmdOptions;
    // The xevm target is attached by the GpuKernelOutline pass, with the chip
    // taken from the device attributes. The upstream pipeline would attach
    // another target with a statically configured chip and, since the two are
    // not identical, each kernel would be compiled once per target. Disable
    // the upstream attach by matching no module name.
    opts.xevmModuleMatcher = "^$";
    gpu::buildLowerToXeVMPassPipeline(pm, opts);
  });

  phase("GpuToGcGpu",
        [&]() { pm.addPass(createGpuToGcGpu({pipelineOpts.wait})); });
}

void registerGPUPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>(
      "gc-gpu-pipeline", "Graph Compiler GPU pipeline", populateGPUPipeline);
}

} // namespace mlir::gc
