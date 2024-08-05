
/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#include "gc/Dialect/LLVMIR/GENDialect.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#endif
#include "gc/Target/LLVM/GEN/Target.h"
#include "gc/Target/LLVMIR/Dialect/GEN/GENToLLVMIRTranslation.h"
#include "gc/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef GC_USE_IMEX
#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>
#endif

namespace mlir::gc {
void registerCPUPipeline();
void registerGPUPipeline();
} // namespace mlir::gc

int main(int argc, char *argv[]) {
#ifdef GC_USE_IMEX
  imex::registerTransformsPasses();
  // Conversion passes
  imex::registerConvertGPUToGPUX();
  imex::registerConvertGPUXToLLVM();
  imex::registerConvertGPUXToSPIRV();
  imex::registerConvertXeGPUToVC();
  imex::registerConvertXeTileToXeGPU();
#endif
  mlir::registerAllPasses();
  mlir::gc::registerCPUPipeline();
  mlir::gc::registerGPUPipeline();
  mlir::gc::registerGraphCompilerPasses();
  mlir::cpuruntime::registerCPURuntimePasses();
  mlir::DialectRegistry registry;
#ifdef GC_HAS_ONEDNN_DIALECT
  registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
#endif
  registry.insert<mlir::cpuruntime::CPURuntimeDialect>();
  registry.insert<mlir::linalgx::LinalgxDialect>();
  registry.insert<mlir::microkernel::MicrokernelDialect>();
  registry.insert<mlir::gen::GENDialect>();
  mlir::registerAllDialects(registry);
  // covers lowerings for weird dialects like ub
  // TODO: avoid `registerALL` to remove this
  mlir::registerAllExtensions(registry);
  // Adds missing `LLVMTranslationDialectInterface` registration for dialect for
  // gpu.module op
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::gen::registerGenTargetInterfaceExternalModels(registry);
  mlir::registerGENDialectTranslation(registry);
#ifdef GC_USE_GPU
  registry.insert<::imex::xetile::XeTileDialect, ::imex::gpux::GPUXDialect>();
#endif
  mlir::cpuruntime::registerConvertCPURuntimeToLLVMInterface(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Graph Compiler modular optimizer driver\n", registry));
}
