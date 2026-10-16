//===-- GpuCompiler.cpp - Native GPU binary compiler --------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <string>
#include <utility>

#include <llvm/Support/Error.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include "gc/ExecutionEngine/JitEngine.h"
#include "gc/GpuCompiler.h"
#include "gc/Runtime/GpuRuntime.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"

namespace gc::gpu {

GpuCompiler &GpuCompiler::get() {
  static GpuCompiler instance;
  return instance;
}

FutureProgram GpuCompiler::compile(mlir::OwningOpRef<mlir::ModuleOp> src,
                                   Options opts, const RtSymbols &syms) {
  // TODO: Compile asynchronously.
  const auto ready = [](std::shared_ptr<Program> prog) -> FutureProgram {
    std::promise<std::shared_ptr<Program>> promise;
    promise.set_value(std::move(prog));
    return promise.get_future();
  };
  if (!src) {
    gcLogE("Invalid MLIR module.");
    return ready(nullptr);
  }

  static const Options defaults = [] {
    Options o(mlir::gc::GpuDevicePropsOptions().arch);
    o.igcOpts = mlir::gc::GPUPipelineOptions().igcCmdOptions;
    return o;
  }();
  if (opts.arch.empty()) opts.arch = defaults.arch;
  if (opts.igcOpts.empty()) opts.igcOpts = defaults.igcOpts;

  {
    mlir::gc::GpuDevicePropsOptions deviceProps;
    deviceProps.arch = opts.arch;
    mlir::gc::GPUPipelineOptions pipelineOpts;
    pipelineOpts.deviceProps = &deviceProps;
    pipelineOpts.dump = opts.dump;
    pipelineOpts.igcCmdOptions = opts.igcOpts;
    pipelineOpts.wait = opts.wait;
    mlir::PassManager pm(src->getContext());
    mlir::gc::populateGPUPipeline(pm, pipelineOpts);
    if (mlir::failed(pm.run(*src))) {
      gcLogE("GPU pipeline failed.");
      return ready(nullptr);
    }
  }

  mlir::ExecutionEngineOptions engOpts;
  engOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  auto eng = mlir::ExecutionEngine::create(src.get(), engOpts);
  if (!eng) {
    gcLogE("Failed to create execution engine: ",
           llvm::toString(eng.takeError()));
    return ready(nullptr);
  }

  auto jit = std::make_unique<JitEngine>(std::move(*eng));
  jit->registerSymbols(syms);
  return ready(std::make_shared<Program>(std::move(jit)));
}

FutureProgram GpuCompiler::compile(const std::string_view src, Options opts,
                                   const RtSymbols &syms) {
  static mlir::MLIRContext *ctx = [] {
    auto *c = new mlir::MLIRContext(mlir::gc::getDialectRegistry());
    c->loadAllAvailableDialects();
    return c;
  }();
  return compile(mlir::parseSourceString<mlir::ModuleOp>(src, ctx),
                 std::move(opts), syms);
}

Program::Program(std::unique_ptr<::gc::JitEngine> eng)
    : sym{
          gcGetOrReport(eng->lookupPacked(Program::Symbols::mainName)),
          reinterpret_cast<Program::Symbols::eventNumType>(
              gcGetOrReport(eng->lookup(Program::Symbols::eventNumName))),
          reinterpret_cast<Program::Symbols::destructorType>(
              gcGetOrReport(eng->lookup(Program::Symbols::destructorName))),
      },
      evtNum(sym.eventNum()), jit(std::move(eng)) {}
Program::~Program() { sym.destructor(); }

// TODO: Implement a parallel compilation.
GpuCompiler::GpuCompiler(unsigned) {}
GpuCompiler::~GpuCompiler() = default;
} // namespace gc::gpu
