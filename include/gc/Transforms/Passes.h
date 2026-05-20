//===- Passes.h - Graph Compiler passes ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_PASSES_H
#define GC_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
class OpBuilder;
class ModuleOp;

namespace func {
class FuncOp;
}

class OpPassManager;

namespace gc {
struct GpuDevicePropsOptions;
struct GPUPipelineOptions : PassPipelineOptions<GPUPipelineOptions> {
  Option<bool> dump{*this, "dump",
                    llvm::cl::desc("Dump the IR after each phase."),
                    llvm::cl::init(false)};
  Option<bool> isUsmArgs{
      *this, "is-usm-args",
      llvm::cl::desc("Whether to use USM(unified shared memory) func args, in "
                     "which the host and device could access the same buffer "
                     "and there is no need to add memcpy explicitly."),
      llvm::cl::init(true)};
  Option<bool> callFinish{
      *this, "call-finish",
      llvm::cl::desc(
          "Call finish() after each GPU kernel launch. This option is passed "
          "to the GpuToGpuOcl path, if use-gpu-ocl is true."),
      llvm::cl::init(false)};
  Option<bool> enableAttentionPrefetch{
      *this, "enable-attention-prefetch",
      llvm::cl::desc("Enable the SetAttentionPrefetch pass."),
      llvm::cl::init(false)};
  Option<std::string> igcCmdOptions{
      *this, "igc-cmd-options",
      llvm::cl::desc("Command options to pass to IGC compiler."),
      llvm::cl::init("")};
  const GpuDevicePropsOptions *deviceProps = nullptr;
};

DialectRegistry &getDialectRegistry();

void populateGPUPipeline(mlir::OpPassManager &, const GPUPipelineOptions &);

#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
