//===- Target.cpp - MLIR LLVM GEN target compilation ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines GEN target related functions including registration
// calls for the `#gen.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "gc/Target/LLVM/GEN/Target.h"

#include "gc/Dialect/LLVMIR/GENDialect.h"
#include "gc/Target/LLVM/GEN/Utils.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;
using namespace mlir::gen;

namespace {
// Gen implementation of the gpu:TargetAttrInterface.
class GenTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<GenTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

void mlir::gen::registerGenTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gen::GENDialect *dialect) {
    GenTargetAttr::attachInterface<GenTargetAttrImpl>(*ctx);
  });
}

void mlir::gen::registerGenTargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  registerGenTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

StringRef mlir::gen::getONEAPIToolkitPath() {
  if (const char *var = std::getenv("ONEAPI_ROOT"))
    return var;
  return "/usr/";
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, GenTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(), {},
                     target.getO()),
      target(target), toolkitPath(targetOptions.getToolkitPath()) {
  if (toolkitPath.empty())
    toolkitPath = getONEAPIToolkitPath();
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
#if LLVM_HAS_SPIRV_TARGET
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
#endif
  });
}

GenTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

StringRef SerializeGPUModuleBase::getToolkitPath() const { return toolkitPath; }

namespace {
class GenSerializer : public SerializeGPUModuleBase {
public:
  GenSerializer(Operation &module, GenTargetAttr target,
                const gpu::TargetOptions &targetOptions);

  gpu::GPUModuleOp getOperation();

  std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &serializedSPV);

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

  std::optional<std::string> findTool(StringRef tool);

private:
  using TmpFile = std::pair<llvm::SmallString<128>, llvm::FileRemover>;
  std::optional<TmpFile> createTemp(StringRef name, StringRef suffix);

  std::optional<std::string>
  translateToSPIRVBinary(llvm::Module &llvmModule,
                         llvm::TargetMachine &targetMachine);
  gpu::TargetOptions targetOptions;
};
} // namespace

GenSerializer::GenSerializer(Operation &module, GenTargetAttr target,
                             const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions) {}

gpu::GPUModuleOp GenSerializer::getOperation() {
  return dyn_cast<gpu::GPUModuleOp>(&SerializeGPUModuleBase::getOperation());
}

std::optional<GenSerializer::TmpFile>
GenSerializer::createTemp(StringRef name, StringRef suffix) {
  llvm::SmallString<128> filename;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(name, suffix, filename);
  if (ec) {
    getOperation().emitError() << "Couldn't create the temp file: `" << filename
                               << "`, error message: " << ec.message();
    return std::nullopt;
  }
  return TmpFile(filename, llvm::FileRemover(filename.c_str()));
}

std::optional<std::string> GenSerializer::findTool(StringRef tool) {
  if (std::optional<std::string> toolPath =
          llvm::sys::Process::FindInEnvPath("PATH", tool))
    return *toolPath;
  getOperation().emitError()
      << "Couldn't find the `" << tool
      << "` binary. Please specify the toolkit "
         "path, add the compiler to $PATH, or set one of the environment "
         "variables in `gen::getGENToolkitPath()`.";
  return std::nullopt;
}

std::optional<SmallVector<char, 0>>
GenSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is `offload`.
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

#if !LLVM_HAS_SPIRV_TARGET
  getOperation()->emitError(
      "The `SPIRV` target was not built. Please enable it when building LLVM.");
  return std::nullopt;
#endif // LLVM_HAS_SPIRV_TARGET

  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "Target Machine unavailable for triple "
                               << triple << ", can't compile with LLVM\n";
    return std::nullopt;
  }

  // Return SPIRV if the compilation target is `assembly`.
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
    std::optional<std::string> serializedISA =
        translateToISA(llvmModule, **targetMachine);
    if (!serializedISA) {
      getOperation().emitError() << "Failed translating the module to ISA.";
      return std::nullopt;
    }
    // Make sure to include the null terminator.
    StringRef bin(serializedISA->c_str(), serializedISA->size() + 1);
    return SmallVector<char, 0>(bin.begin(), bin.end());
  }

  std::optional<std::string> serializedSPIRVBinary =
      translateToSPIRVBinary(llvmModule, **targetMachine);
  if (!serializedSPIRVBinary) {
    getOperation().emitError() << "Failed translating the module to Binary.";
    return std::nullopt;
  }

  StringRef bin(serializedSPIRVBinary->c_str(),
                serializedSPIRVBinary->size() + 1);
  return SmallVector<char, 0>(bin.begin(), bin.end());
  return compileToBinary(*serializedSPIRVBinary);
}

std::optional<SmallVector<char, 0>>
GenSerializer::compileToBinary(const std::string &serializedSPV) {
  std::optional<std::string> ocloc = findTool("ocloc");
  if (!ocloc)
    return std::nullopt;

  std::string basename =
      llvm::formatv("mlir-{0}-{1}-{2}", getOperation().getNameAttr().getValue(),
                    getTarget().getTriple(), getTarget().getChip());

  std::optional<TmpFile> spvFile = createTemp(basename, "spv");
  if (!spvFile)
    return std::nullopt;
  std::optional<TmpFile> binaryFile = createTemp(basename, "bin");
  if (!binaryFile)
    return std::nullopt;

  Location loc = getOperation().getLoc();
  std::error_code ec;
  {
    llvm::raw_fd_ostream spvStream(spvFile->first, ec);
    if (ec) {
      emitError(loc) << "Couldn't open the file: `" << spvFile->first
                     << "`, error message: " << ec.message();
      return std::nullopt;
    }
    spvStream << serializedSPV;
    if (spvStream.has_error()) {
      emitError(loc) << "An error occurred while writing the SPIRV to: `"
                     << spvFile->first << "`.";
      return std::nullopt;
    }
    spvStream.flush();
  }

  SmallVector<StringRef, 12> oclocArgs(
      {StringRef("compile"), StringRef("-device"), getTarget().getChip(),
       StringRef("-qq"), StringRef("-spirv_input"), StringRef("-file"),
       StringRef(spvFile->first), StringRef("-o"),
       StringRef(binaryFile->first)});

  std::string message;
  if (llvm::sys::ExecuteAndWait(ocloc.value(), oclocArgs,
                                /*Env=*/std::nullopt,
                                /*Redirects=*/std::nullopt,
                                /*SecondsToWait=*/0,
                                /*MemoryLimit=*/0,
                                /*ErrMsg=*/&message)) {
    emitError(loc) << " ocloc invocation failed. Message:\n" << message;
    return std::nullopt;
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> binaryBuffer =
      llvm::MemoryBuffer::getFile(binaryFile->first);
  if (!binaryBuffer) {
    emitError(loc) << "Couldn't open the file: `" << binaryFile->first
                   << "`, error message: " << binaryBuffer.getError().message();
    return std::nullopt;
  }
  StringRef result = (*binaryBuffer)->getBuffer();
  return SmallVector<char, 0>(result.begin(), result.end());
}

std::optional<std::string>
GenSerializer::translateToSPIRVBinary(llvm::Module &llvmModule,
                                      llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile))
      return std::nullopt;

    codegenPasses.run(llvmModule);
  }
  return stream.str();
}

std::optional<SmallVector<char, 0>>
GenTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                     const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("expected to be a gpu.module op");
    return std::nullopt;
  }

  GenSerializer serializer(*module, cast<GenTargetAttr>(attribute), options);
  serializer.init();

  return serializer.run();
}

Attribute
GenTargetAttrImpl::createObject(Attribute attribute,
                                const SmallVector<char, 0> &object,
                                const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps);
}
