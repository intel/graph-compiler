//===-- GpuCompiler.cpp - Native GPU binary compiler --------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include "gc/ExecutionEngine/JitEngine.h"
#include "gc/GpuCompiler.h"
#include "gc/Runtime/GpuRuntime.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Cache.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"

namespace gc::gpu {

constexpr auto CACHE_FILE_BIN = "program.bin";
constexpr auto CACHE_FILE_INPUT = "input.mlir";
constexpr auto CACHE_FILE_OUTPUT = "output.mlir";
constexpr auto CACHE_FILE_DUMP = "pipeline.dump";

// With concurrency 1, the compilation runs in the caller's thread and a module
// source is compiled in its own context. Otherwise, the source is compiled on
// the cache's thread pool in the internal context.
//
// The internal context is used concurrently without locks. This is safe only
// while its dialect registry and loaded dialects do not change, so all of them,
// including the pipeline's dependent dialects that PassManager::run() would
// append, are loaded here.
struct GpuCompiler::Impl {
  explicit Impl(unsigned concurrency)
      : ctx(std::make_unique<mlir::MLIRContext>(
            mlir::gc::getDialectRegistry())),
        cache(concurrency) {
    mlir::gc::GpuDevicePropsOptions deviceProps;
    mlir::gc::GPUPipelineOptions pipelineOpts;
    pipelineOpts.deviceProps = &deviceProps;
    pipelineOpts.enableAttentionPrefetch = true;
    mlir::PassManager pm(ctx.get());
    mlir::gc::populateGPUPipeline(pm, pipelineOpts);
    mlir::DialectRegistry deps;
    pm.getDependentDialects(deps);
    ctx->appendDialectRegistry(deps);
    ctx->loadAllAvailableDialects();
  }

  // Either mod or text is the source.
  FutureProgram compile(mlir::OwningOpRef<mlir::ModuleOp> mod,
                        std::string_view text, Options opts,
                        const RtSymbols &syms) {
    static const Options defaults = [] {
      Options o(mlir::gc::GpuDevicePropsOptions().arch);
      o.igcOpts = mlir::gc::GPUPipelineOptions().igcCmdOptions;
      return o;
    }();
    if (opts.arch.empty()) opts.arch = defaults.arch;
    if (opts.igcOpts.empty()) opts.igcOpts = defaults.igcOpts;

    std::string input;
    if (mod) {
      llvm::raw_string_ostream os(input);
      mod->print(os);
      text = input;
    }
    gc::cache::Key key({
        text,
        opts.arch,
        opts.igcOpts,
        opts.wait ? "" : "!wait",
    });
    return cache.get(key, [&] {
      // The async task owns the text and parses it into the internal context.
      // The sync one runs before return, viewing the caller's text.
      const bool async = cache.async();
      if (async) {
        if (!mod) input = text;
        text = {};
        mod = nullptr;
      }
      return [this, mod = std::move(mod),
              input = async ? std::move(input) : std::string(), text,
              opts = std::move(opts), &syms](Cache::Accessor &acc) mutable {
        build(acc, std::move(mod), text.empty() ? input : text, opts, syms);
      };
    });
  }

private:
  using Cache = gc::cache::FileCache<Program>;

  // A null mod is parsed from input into the internal context. On failure,
  // put() is not called, so waiters get nullptr.
  void build(Cache::Accessor &acc, mlir::OwningOpRef<mlir::ModuleOp> mod,
             llvm::StringRef input, const Options &opts,
             const RtSymbols &syms) {
    if (auto eng = JitEngine::load(acc.read(CACHE_FILE_BIN, true))) {
      gcLogD("Returning program from file cache: ", acc.key.hex());
      eng->registerSymbols(syms);
      acc.put(std::make_shared<Program>(std::move(eng)));
      return;
    }

    if (!mod) {
      mod = mlir::parseSourceString<mlir::ModuleOp>(input, ctx.get());
      if (!mod) {
        gcLogE("Failed to parse MLIR module:\n", input);
        return;
      }
    }

    static bool dumpIr = gcGetEnv("GC_DUMP_IR", false);
    int dumpMode =
#ifndef NDEBUG
        dumpIr || opts.dump ? 3 : 2 // Dump also into CACHE_FILE_DUMP
#else
        dumpIr || opts.dump ? 1 : 0
#endif
        ;
    mlir::gc::GpuDevicePropsOptions deviceProps;
    deviceProps.arch = opts.arch;
    mlir::gc::GPUPipelineOptions pipelineOpts;
    pipelineOpts.deviceProps = &deviceProps;
    pipelineOpts.dump = dumpMode > 0;
    pipelineOpts.igcCmdOptions = opts.igcOpts;
    pipelineOpts.wait = opts.wait;

    std::string dump;
    {
      llvm::raw_string_ostream dumpStream(dump);
      if (dumpMode > 1) pipelineOpts.dumpStream = &dumpStream;
      mlir::PassManager pm(mod->getContext());
      mlir::gc::populateGPUPipeline(pm, pipelineOpts);
      if (mlir::failed(pm.run(*mod))) {
        gcLogE("GPU pipeline failed.");
        return;
      }
    }
    if (dumpMode == 3) {
      static std::mutex dbgsMux;
      std::lock_guard<std::mutex> lock(dbgsMux);
      llvm::dbgs() << dump;
    }

    mlir::ExecutionEngineOptions engOpts;
    engOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    engOpts.enableObjectDump = true;
    auto eng = mlir::ExecutionEngine::create(mod.get(), engOpts);
    if (!eng) {
      gcLogE("Failed to create execution engine: ",
             llvm::toString(eng.takeError()));
      return;
    }

    auto jit = std::make_unique<JitEngine>(std::move(*eng));
    jit->registerSymbols(syms);
    auto prog = std::make_shared<Program>(std::move(jit));
    gcLogD("Built new program: ", acc.key.hex());
    acc.put(prog);
    if (!acc.write(
            CACHE_FILE_BIN,
            [&](llvm::StringRef tmp) { return prog->jit->saveToFile(tmp); },
            true))
      return;

    std::string output;
    llvm::raw_string_ostream(output) << *mod;
    llvm::StringRef names[] = {CACHE_FILE_INPUT, CACHE_FILE_OUTPUT,
                               CACHE_FILE_DUMP};
    llvm::StringRef contents[] = {input, output, dump};
    for (auto [name, content] : llvm::zip(names, contents))
      if (!content.empty()) acc.write(name, content);
  }

  std::unique_ptr<mlir::MLIRContext> ctx;
  // Must be last: destroyed first, draining the tasks using the members above.
  Cache cache;
};

GpuCompiler &GpuCompiler::get() {
  static GpuCompiler instance(gcGetEnv("GC_THREADS", 0u));
  return instance;
}

FutureProgram GpuCompiler::compile(mlir::OwningOpRef<mlir::ModuleOp> src,
                                   Options opts, const RtSymbols &syms) {
  if (!src) {
    gcLogE("Invalid MLIR module.");
    std::promise<std::shared_ptr<Program>> promise;
    promise.set_value(nullptr);
    return promise.get_future();
  }
  return impl->compile(std::move(src), {}, std::move(opts), syms);
}

FutureProgram GpuCompiler::compile(const std::string_view src, Options opts,
                                   const RtSymbols &syms) {
  return impl->compile(nullptr, src, std::move(opts), syms);
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

GpuCompiler::GpuCompiler(unsigned concurrency)
    : impl(std::make_unique<Impl>(concurrency)) {}
GpuCompiler::~GpuCompiler() = default;
} // namespace gc::gpu
