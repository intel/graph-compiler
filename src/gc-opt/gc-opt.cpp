
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
#include "gc/Dialect/LLVMIR/XeVMDialect.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#endif
#include "gc/Conversion/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef GC_USE_IMEX
#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>
#endif

namespace mlir::gc {
void registerCPUPipeline();
#ifdef GC_USE_IMEX
void registerGPUPipeline();
#endif
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
  mlir::gc::registerGPUPipeline();
#endif
  mlir::registerAllPasses();
  mlir::gc::registerCPUPipeline();
  mlir::gc::registerGraphCompilerPasses();
  mlir::registerGCConversionPasses();
  mlir::cpuruntime::registerCPURuntimePasses();
  mlir::microkernel::registerMicrokernelPasses();

  mlir::DialectRegistry registry;
#ifdef GC_HAS_ONEDNN_DIALECT
  registry.insert<mlir::onednn_graph::OneDNNGraphDialect>();
#endif
  registry.insert<mlir::cpuruntime::CPURuntimeDialect>();
  registry.insert<mlir::linalgx::LinalgxDialect>();
  registry.insert<mlir::microkernel::MicrokernelDialect>();
  registry.insert<mlir::xevm::XeVMDialect>();
  mlir::registerAllDialects(registry);
#ifdef GC_USE_IMEX
  registry.insert<::imex::xetile::XeTileDialect, ::imex::gpux::GPUXDialect>();
#endif
  mlir::cpuruntime::registerConvertCPURuntimeToLLVMInterface(registry);
  mlir::registerAllExtensions(registry); // TODO: cleanup
  // Adds missing `LLVMTranslationDialectInterface` registration for dialect for
  // gpu.module op
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerConvertXeVMToLLVMInterface(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Graph Compiler modular optimizer driver\n", registry));
}
