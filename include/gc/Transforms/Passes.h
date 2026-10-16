//===- Passes.h - Graph Compiler passes ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_PASSES_H
#define GC_PASSES_H

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassOptions.h>

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
  Option<bool> truncate{
      *this, "truncate-bin",
      llvm::cl::desc("Truncate large binary blobs in the dumps."),
      llvm::cl::init(true)};
  Option<bool> enableAttentionPrefetch{
      *this, "enable-attention-prefetch",
      llvm::cl::desc("Enable the SetAttentionPrefetch pass."),
      llvm::cl::init(false)};
  Option<std::string> igcCmdOptions{
      *this, "igc-cmd-options",
      llvm::cl::desc("Commandline options to pass to IGC compiler."),
      llvm::cl::init("")};
  Option<bool> wait{
      *this, "wait",
      llvm::cl::desc("Wait for all the GPU events at the end of the main "
                     "function instead of returning them to the caller."),
      llvm::cl::init(true)};
  Option<bool> dump{*this, "dump",
                    llvm::cl::desc("Dump the IR after each phase."),
                    llvm::cl::init(false)};
  llvm::raw_ostream *dumpStream = nullptr;
  const GpuDevicePropsOptions *deviceProps = nullptr;
};

DialectRegistry &getDialectRegistry();

void populateGPUPipeline(mlir::OpPassManager &, const GPUPipelineOptions &);

void registerGPUPipeline();

#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
