/*
 * Copyright (C) 2025 Intel Corporation
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

#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Error.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

namespace {
struct Options {
  llvm::cl::OptionCategory runnerCategory{"GPU runner options"};
  llvm::cl::opt<std::string> inputFilename{
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(runnerCategory)};
  llvm::cl::opt<std::string> mainFuncName{
      "e",
      llvm::cl::desc("The function to be executed. If not specified, the "
                     "first matching function in the module to be used."),
      llvm::cl::value_desc("function name"), llvm::cl::cat(runnerCategory)};
  llvm::cl::opt<bool> skipPipeline{
      "skip-pipeline",
      llvm::cl::desc("Skip the GPU pipeline. It's expected, that the input is "
                     "already lowered with 'gc-op --gc-gpu-pipeline'."),
      llvm::cl::init(false), llvm::cl::cat(runnerCategory)};
  llvm::cl::list<std::string> sharedLibs{
      "shared-libs",
      llvm::cl::desc("Comma separated library paths to link dynamically."),
      llvm::cl::MiscFlags::CommaSeparated, llvm::cl::desc("<lib1,lib2,...>"),
      llvm::cl::cat(runnerCategory)};
  llvm::cl::opt<bool> printIr{
      "print-ir",
      llvm::cl::desc("Print the resulting IR before the execution."),
      llvm::cl::init(false), llvm::cl::cat(runnerCategory)};
  llvm::cl::opt<bool> dumpSpirv{
      "dump-spirv", llvm::cl::desc("Dump spirv for generated kernels."),
      llvm::cl::init(false), llvm::cl::cat(runnerCategory)};
  llvm::cl::opt<std::string> objDumpFile{
      "obj-dump-file",
      llvm::cl::desc("Dump the compiled object to the specified file."),
      llvm::cl::value_desc("file path"), llvm::cl::cat(runnerCategory)};
};
} // namespace

void findFunc(Options &opts, ModuleOp mod) {
  bool (*matcher)(ArrayRef<Type>, ModuleOp &);

  if (opts.skipPipeline) {
    matcher = [](ArrayRef<Type> args, ModuleOp &mod) {
      if (args.size() != 3)
        return false;
      auto ctx = mod.getContext();
      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      return args[0] == ptrType && args[1] == ptrType &&
             args[2] == IntegerType::get(ctx, 64);
    };
  } else {
    matcher = [](ArrayRef<Type> args, ModuleOp &) { return args.empty(); };
  }

  if (opts.mainFuncName.empty()) {
    auto setFuncName = [&](auto funcOp) {
      if (funcOp && !funcOp.isExternal() && funcOp.isPublic() &&
          matcher(funcOp.getArgumentTypes(), mod)) {
        opts.mainFuncName = funcOp.getName().str();
        return true;
      }
      return false;
    };

    for (auto &op : mod.getBody()->getOperations()) {
      if (setFuncName(dyn_cast<LLVM::LLVMFuncOp>(op)) ||
          setFuncName(dyn_cast<func::FuncOp>(op))) {
        return;
      }
    }
    gcReportErr("No matching function found.");
  }

  ArrayRef<Type> args;
  if (auto llvmFunc = mod.lookupSymbol<LLVM::LLVMFuncOp>(opts.mainFuncName)) {
    args = llvmFunc.getArgumentTypes();
  } else if (auto func = mod.lookupSymbol<func::FuncOp>(opts.mainFuncName)) {
    args = func.getArgumentTypes();
  } else {
    gcReportErr("The function '", opts.mainFuncName.c_str(), "' not found.");
  }

  if (!matcher(args, mod)) {
    if (opts.skipPipeline) {
      gcReportErr("The function '", opts.mainFuncName.c_str(),
                  "' signature does not match (!llvm.ptr, !llvm.ptr, i64).");
    }
    gcReportErr("The function '", opts.mainFuncName.c_str(),
                "' must have no arguments.");
  }
}

int main(int argc, char **argv) {
  Options opts;
  llvm::cl::ParseCommandLineOptions(argc, argv, "GraphCompiler GPU runner\n");

  std::string errMsg;
  auto file = openInputFile(opts.inputFilename, &errMsg);
  if (!file) {
    gcReportErr("Failed to read input IR: ", errMsg.c_str());
  }

  auto srcMgr = std::make_shared<llvm::SourceMgr>();
  srcMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  MLIRContext mlirCtx{gc::initCompilerAndGetDialects()};
  auto mlirMod = parseSourceFile<ModuleOp>(srcMgr, {&mlirCtx});
  findFunc(opts, *mlirMod);

  gc::gpu::OclModuleBuilderOpts builderOpts;
  SmallVector<StringRef, 4> sharedLibs(opts.sharedLibs.begin(),
                                       opts.sharedLibs.end());
  builderOpts.funcName = opts.mainFuncName;
  builderOpts.printIr = opts.printIr;
  builderOpts.spirvDump = opts.dumpSpirv;
  builderOpts.enableObjectDump = !opts.objDumpFile.getValue().empty();
  builderOpts.sharedLibPaths = sharedLibs;
  builderOpts.pipeline =
      opts.skipPipeline ? [](OpPassManager &) {} : [](OpPassManager &pm) {
        gc::GPUPipelineOptions pipelineOpts;
        pipelineOpts.isUsmArgs = false;
        pipelineOpts.callFinish = true;
#ifdef GC_USE_IMEX
        populateIMEXPipeline(pm, pipelineOpts);
#else
        populateGPUPipeline(pm, pipelineOpts);
#endif
      };

  gc::gpu::OclModuleBuilder builder{mlirMod, builderOpts};
  auto runtime = gcGetOrReport(gc::gpu::OclRuntime::get());
  auto oclMod = gcGetOrReport(builder.build(runtime));
  assert(oclMod->isStatic);

  if (!opts.objDumpFile.getValue().empty()) {
    gcLogD("Dumping the compiled object to ", opts.objDumpFile.getValue());
    oclMod->dumpToObjectFile(opts.objDumpFile.getValue());
  }

  auto queue = gcGetOrReport(runtime.createQueue());
  gc::gpu::OclContext ctx{runtime, queue};
  gc::gpu::StaticExecutor<0> exec{oclMod};
  gcLogD("Executing function ", opts.mainFuncName.c_str(), "()");
  exec(ctx);
  gcGetOrReport(ctx.finish());
  gcGetOrReport(runtime.releaseQueue(queue));
  return 0;
}
