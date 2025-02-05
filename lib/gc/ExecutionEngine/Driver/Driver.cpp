//===-- Driver.cpp - Top-level MLIR compiler driver -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#endif
#include "gc/Conversion/Passes.h"
#include "gc/Target/LLVM/XeVM/Target.h"
#include "gc/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "gc/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "string.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir {
namespace gc {

static DialectRegistry initDialects() {
  mlir::registerAllPasses();
  mlir::gc::registerGraphCompilerPasses();
  mlir::registerGCConversionPasses();
  mlir::cpuruntime::registerCPURuntimePasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::cpuruntime::CPURuntimeDialect>();
  mlir::registerAllDialects(registry);
  mlir::cpuruntime::registerConvertCPURuntimeToLLVMInterface(registry);
  mlir::registerAllExtensions(registry);
  // Adds missing `LLVMTranslationDialectInterface` registration for dialect for
  // gpu.module op
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerConvertXeVMToLLVMInterface(registry);
  mlir::registerXeVMDialectTranslation(registry);
  mlir::xevm::registerXeVMTargetInterfaceExternalModels(registry);
#ifdef GC_HAS_ONEDNN_DIALECT
  registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
#endif
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  return registry;
}

DialectRegistry &initCompilerAndGetDialects() {
  static DialectRegistry reg = initDialects();
  return reg;
}

static const char defaultComputeName[] = "_mlir_ciface_compute";

llvm::Expected<std::shared_ptr<JitModule>>
JitModule::create(Operation *op, const DriverOptions &options) {
  if (options.runTransforms) {
    mlir::PassManager pm{op->getContext()};
    populateCPUPipeline(pm);
    if (auto result = pm.run(op); failed(result)) {
      return llvm::make_error<llvm::StringError>(
          "MLIR pass error", llvm::inconvertibleErrorCode());
    }
  }
  ExecutionEngineOptions exeOptions;
  exeOptions.jitCodeGenOptLevel = options.jitCodeGenOptLevel;
  std::unique_ptr<llvm::TargetMachine> tm = nullptr;
  auto exec = ExecutionEngine::create(op, exeOptions, std::move(tm));
  if (!exec) {
    return exec.takeError();
  }
  auto &engine = *exec;
  JitModuleFuncT compute;
  {
    auto expectCompute = engine->lookupPacked(defaultComputeName);
    if (!expectCompute) {
      return expectCompute.takeError();
    }
    compute = *expectCompute;
  }
  return std::make_shared<JitModule>(std::move(engine), compute);
}

JitModule::JitModule(std::unique_ptr<ExecutionEngine> engine,
                     JitModuleFuncT compute)
    : engine{std::move(engine)}, compute{compute} {}
JitModule::~JitModule() = default;

} // namespace gc
} // namespace mlir