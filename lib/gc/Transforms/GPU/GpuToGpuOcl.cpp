//===-- GpuToGpuOcl.cpp - GpuToGpuOcl path ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <unordered_set>

#define GC_GPU_OCL_CONST_ONLY
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::gc::gpu;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTOGPUOCL
#define GEN_PASS_DEF_GPUTOGPUOCL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
LLVM::CallOp funcCall(OpBuilder &builder, const StringRef name,
                      const Type returnType, const ArrayRef<Type> argTypes,
                      const Location loc, const ArrayRef<Value> arguments,
                      bool isVarArg = false) {
  auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!function) {
    auto type = LLVM::LLVMFunctionType::get(returnType, argTypes, isVarArg);
    function = OpBuilder::atBlockEnd(module.getBody())
                   .create<LLVM::LLVMFuncOp>(loc, name, type);
  }
  return builder.create<LLVM::CallOp>(loc, function, arguments);
}

// Assuming that the pointer to the context is passed as the last argument
// of the current function of type memref<anyType> with zero dims. When lowering
// to LLVM, the memref arg is replaced with 3 args of types ptr, ptr, i64.
// Returning the first one.
Value getCtxPtr(const OpBuilder &rewriter) {
  auto func =
      rewriter.getBlock()->getParent()->getParentOfType<LLVM::LLVMFuncOp>();
  return func.getArgument(func.getNumArguments() - 3);
}

struct Helper final {
  LLVMTypeConverter &converter;
  Type voidType;
  Type ptrType;
  Type idxType;
  mutable std::unordered_set<std::string> kernelNames;

  explicit Helper(MLIRContext *ctx, LLVMTypeConverter &converter)
      : converter(converter), voidType(LLVM::LLVMVoidType::get(ctx)),
        ptrType(LLVM::LLVMPointerType::get(ctx)),
        idxType(IntegerType::get(ctx, converter.getPointerBitwidth())) {}

  Value idxConstant(OpBuilder &rewriter, const Location loc,
                    size_t value) const {
    return rewriter.create<LLVM::ConstantOp>(
        loc, idxType,
        rewriter.getIntegerAttr(idxType, static_cast<int64_t>(value)));
  }

  void destroyKernels(OpBuilder &rewriter, Location loc,
                      ArrayRef<Value> kernelPtrs) const {
    auto size = idxConstant(rewriter, loc, kernelPtrs.size());
    auto kernelPtrsArray =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, size);
    for (size_t i = 0, n = kernelPtrs.size(); i < n; i++) {
      auto elementPtr =
          rewriter.create<LLVM::GEPOp>(loc, ptrType, ptrType, kernelPtrsArray,
                                       idxConstant(rewriter, loc, i));
      rewriter.create<LLVM::StoreOp>(loc, kernelPtrs[i], elementPtr);
    }

    funcCall(rewriter, GPU_OCL_KERNEL_DESTROY, voidType, {idxType, ptrType},
             loc, {size, kernelPtrsArray});
  }
};

template <typename SourceOp>
struct ConvertOpPattern : ConvertOpToLLVMPattern<SourceOp> {
  const Helper &helper;

  explicit ConvertOpPattern(const Helper &helper)
      : ConvertOpToLLVMPattern<SourceOp>(helper.converter), helper(helper) {}
};

struct ConvertAlloc final : ConvertOpPattern<gpu::AllocOp> {
  explicit ConvertAlloc(const Helper &helper) : ConvertOpPattern(helper) {}

  LogicalResult
  matchAndRewrite(gpu::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();
    MemRefType type = allocOp.getType();
    auto shape = type.getShape();
    auto dynamics = adaptor.getDynamicSizes();

    if (shape.empty() || dynamics.empty()) {
      int64_t staticSize;
      if (shape.empty()) {
        staticSize = 0;
      } else {
        staticSize = type.getElementType().getIntOrFloatBitWidth() / 8;
        for (auto dim : shape) {
          assert(dim != ShapedType::kDynamic);
          staticSize *= dim;
        }
      }
      auto size = helper.idxConstant(rewriter, loc, staticSize);
      auto ptr = funcCall(rewriter, GPU_OCL_MALLOC, helper.ptrType,
                          {helper.ptrType, helper.idxType}, loc,
                          {getCtxPtr(rewriter), size})
                     .getResult();
      Value replacement = MemRefDescriptor::fromStaticShape(
          rewriter, loc, helper.converter, type, ptr, ptr);
      rewriter.replaceOp(allocOp, replacement);
      return success();
    }

    auto ndims = shape.size();
    SmallVector<Value> newShape;
    SmallVector<Value> newStrides(ndims);
    auto staticSize = type.getElementType().getIntOrFloatBitWidth() / 8;
    auto size = dynamics[0];

    auto idxMul = [&](Value x, Value y) -> Value {
      if (auto xConst = getConstantIntValue(x)) {
        if (auto yConst = getConstantIntValue(y)) {
          return helper.idxConstant(rewriter, loc,
                                    xConst.value() * yConst.value());
        }
      }
      return rewriter.create<LLVM::MulOp>(loc, x, y);
    };

    for (size_t i = 0, j = 0; i < ndims; i++) {
      auto dim = shape[i];
      if (dim == ShapedType::kDynamic) {
        auto dynSize = dynamics[j++];
        newShape.emplace_back(dynSize);
        if (j != 1) {
          size = idxMul(size, dynSize);
        }
      } else {
        staticSize *= dim;
        newShape.emplace_back(helper.idxConstant(rewriter, loc, dim));
      }
    }

    size = idxMul(size, helper.idxConstant(rewriter, loc, staticSize));
    auto ptr = funcCall(rewriter, GPU_OCL_MALLOC, helper.ptrType,
                        {helper.ptrType, helper.idxType}, loc,
                        {getCtxPtr(rewriter), size})
                   .getResult();

    newStrides[ndims - 1] = helper.idxConstant(rewriter, loc, 1);
    for (int i = static_cast<int>(ndims) - 2; i >= 0; i--) {
      newStrides[i] = idxMul(newStrides[i + 1], newShape[i]);
      ;
    }

    auto dsc = MemRefDescriptor::undef(rewriter, loc,
                                       helper.converter.convertType(type));
    dsc.setAllocatedPtr(rewriter, loc, ptr);
    dsc.setAlignedPtr(rewriter, loc, ptr);
    dsc.setOffset(rewriter, loc, helper.idxConstant(rewriter, loc, 0));

    for (unsigned i = 0, n = static_cast<unsigned>(ndims); i < n; i++) {
      dsc.setSize(rewriter, loc, i, newShape[i]);
      dsc.setStride(rewriter, loc, i, newStrides[i]);
    }

    rewriter.replaceOp(allocOp, static_cast<Value>(dsc));
    return success();
  }
};

struct ConvertDealloc final : ConvertOpPattern<gpu::DeallocOp> {
  explicit ConvertDealloc(const Helper &helper) : ConvertOpPattern(helper) {}

  LogicalResult
  matchAndRewrite(gpu::DeallocOp gpuDealloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gpuDealloc.getLoc();
    MemRefDescriptor dsc(adaptor.getMemref());
    auto ptr = dsc.allocatedPtr(rewriter, loc);
    auto oclDealloc = funcCall(rewriter, GPU_OCL_DEALLOC, helper.voidType,
                               {helper.ptrType, helper.ptrType}, loc,
                               {getCtxPtr(rewriter), ptr});
    rewriter.replaceOp(gpuDealloc, oclDealloc);
    return success();
  }
};

struct ConvertMemcpy final : ConvertOpPattern<gpu::MemcpyOp> {
  explicit ConvertMemcpy(const Helper &helper) : ConvertOpPattern(helper) {}

  LogicalResult
  matchAndRewrite(gpu::MemcpyOp gpuMemcpy, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gpuMemcpy.getLoc();
    auto srcType = gpuMemcpy.getSrc().getType();
    auto elementSize = srcType.getElementType().getIntOrFloatBitWidth() / 8;
    uint64_t numElements = 0;
    for (auto dim : srcType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        gpuMemcpy.emitOpError()
            << "dynamic shapes are not currently not supported";
        return failure();
      }
      numElements = numElements ? numElements * dim : dim;
    }

    MemRefDescriptor srcDsc(adaptor.getSrc());
    MemRefDescriptor dstDsc(adaptor.getDst());
    auto srcPtr = srcDsc.alignedPtr(rewriter, loc);
    auto dstPtr = dstDsc.alignedPtr(rewriter, loc);
    auto size = helper.idxConstant(rewriter, loc, elementSize * numElements);
    auto oclMemcpy = funcCall(
        rewriter, GPU_OCL_MEMCPY, helper.voidType,
        {helper.ptrType, helper.ptrType, helper.ptrType, helper.idxType}, loc,
        {getCtxPtr(rewriter), srcPtr, dstPtr, size});
    rewriter.replaceOp(gpuMemcpy, oclMemcpy);
    return success();
  }
};

struct ConvertLaunch final : ConvertOpPattern<gpu::LaunchFuncOp> {

  explicit ConvertLaunch(const Helper &helper) : ConvertOpPattern(helper) {}

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp gpuLaunch, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kernelPtr = getKernel(gpuLaunch, adaptor, rewriter);
    if (!kernelPtr) {
      return failure();
    }

    const Location loc = gpuLaunch.getLoc();
    auto kernelArgs = adaptor.getKernelOperands();
    SmallVector<Value> args;
    args.reserve(kernelArgs.size() + 2);
    args.emplace_back(getCtxPtr(rewriter));
    args.emplace_back(kernelPtr.value());

    int i = 0;
    for (auto arg : kernelArgs) {
      if (isa<MemRefType>(gpuLaunch.getKernelOperand(i++).getType())) {
        MemRefDescriptor desc(arg);
        args.emplace_back(desc.alignedPtr(rewriter, loc));
      } else {
        args.emplace_back(arg);
      }
    }

    const auto gpuOclLaunch =
        funcCall(rewriter, GPU_OCL_KERNEL_LAUNCH, helper.voidType,
                 {helper.ptrType, helper.ptrType}, loc, args, true);
    rewriter.replaceOp(gpuLaunch, gpuOclLaunch);
    return success();
  }

private:
  // Returns the kernel pointer stored in the global var ...name_Ptr.
  // If it's NULL, calls the createKernel() function.
  std::optional<Value> getKernel(gpu::LaunchFuncOp &gpuLaunch,
                                 OpAdaptor &adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    auto loc = gpuLaunch.getLoc();
    auto ctx = getCtxPtr(rewriter);
    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto kernelModName = gpuLaunch.getKernelModuleName();
    SmallString<128> getFuncName("getGcGpuOclKernel_");
    getFuncName.append(kernelModName);

    if (helper.kernelNames
            .insert(std::string(kernelModName.begin(), kernelModName.end()))
            .second) {
      auto insPoint = rewriter.saveInsertionPoint();
      SmallString<128> strBuf("gcGpuOclKernel_");
      strBuf.append(kernelModName);
      strBuf.append("_");
      auto strBufStart = strBuf.size();
      auto str = [&strBuf,
                  strBufStart](const char *chars) -> SmallString<128> & {
        strBuf.truncate(strBufStart);
        strBuf.append(chars);
        return strBuf;
      };

      SmallString<128> createFuncName("createGcGpuOclKernel_");
      createFuncName.append(kernelModName);
      if (!createKernel(gpuLaunch, adaptor, rewriter, loc, mod, createFuncName,
                        str)) {
        return std::nullopt;
      }

      auto function = rewriter.create<LLVM::LLVMFuncOp>(
          loc, getFuncName,
          LLVM::LLVMFunctionType::get(helper.ptrType, {helper.ptrType}),
          LLVM::Linkage::Internal);
      rewriter.setInsertionPointToStart(function.addEntryBlock(rewriter));

      auto ptr = mod.lookupSymbol<LLVM::GlobalOp>(str("Ptr"));
      assert(ptr);
      auto null = rewriter.create<LLVM::ZeroOp>(loc, helper.ptrType);
      auto ptrPtr = rewriter.create<LLVM::AddressOfOp>(loc, ptr);
      auto ptrVal = rewriter.create<LLVM::LoadOp>(loc, helper.ptrType, ptrPtr);
      auto cmp = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                               ptrVal, null);

      auto body = &function.getBody();
      auto thenBlock = rewriter.createBlock(body);
      auto elseBlock = rewriter.createBlock(body);
      rewriter.setInsertionPointToEnd(&body->front());
      rewriter.create<LLVM::CondBrOp>(loc, cmp, thenBlock, elseBlock);

      // Then block
      rewriter.setInsertionPointToStart(thenBlock);
      auto result = funcCall(rewriter, createFuncName, helper.ptrType,
                             {helper.ptrType}, loc, {function.getArgument(0)});
      rewriter.create<LLVM::ReturnOp>(loc, result.getResult());

      // Else block
      rewriter.setInsertionPointToStart(elseBlock);
      rewriter.create<LLVM::ReturnOp>(loc, ptrVal);

      rewriter.restoreInsertionPoint(insPoint);
    }

    auto kernelFunc = mod.lookupSymbol<LLVM::LLVMFuncOp>(getFuncName);
    if (!kernelFunc) {
      gpuLaunch.emitOpError() << "Function " << getFuncName << " not found!";
      return std::nullopt;
    }
    return rewriter.create<LLVM::CallOp>(loc, kernelFunc, ValueRange(ctx))
        .getResult();
  }

  // Create a new kernel and save the pointer to the global variable
  // ...name_Ptr.
  bool createKernel(
      gpu::LaunchFuncOp &gpuLaunch, OpAdaptor &adaptor,
      ConversionPatternRewriter &rewriter, Location &loc, ModuleOp &mod,
      StringRef funcName,
      const std::function<SmallString<128> &(const char *chars)> &str) const {
    auto kernelModName = gpuLaunch.getKernelModuleName();
    auto kernelMod = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
        gpuLaunch, kernelModName);
    if (!kernelMod) {
      gpuLaunch.emitOpError() << "Module " << kernelModName << " not found!";
      return false;
    }
    const auto binaryAttr = kernelMod->getAttrOfType<StringAttr>("gpu.binary");
    if (!binaryAttr) {
      kernelMod.emitOpError() << "missing 'gpu.binary' attribute";
      return false;
    }

    rewriter.setInsertionPointToStart(mod.getBody());
    // The kernel pointer is stored here
    rewriter.create<LLVM::GlobalOp>(loc, helper.ptrType, /*isConstant=*/false,
                                    LLVM::Linkage::Internal, str("Ptr"),
                                    rewriter.getZeroAttr(helper.ptrType));
    rewriter.eraseOp(kernelMod);

    auto function = rewriter.create<LLVM::LLVMFuncOp>(
        loc, funcName,
        LLVM::LLVMFunctionType::get(helper.ptrType, {helper.ptrType}),
        LLVM::Linkage::Internal);
    rewriter.setInsertionPointToStart(function.addEntryBlock(rewriter));

    auto ptr = mod.lookupSymbol<LLVM::GlobalOp>(str("Ptr"));
    assert(ptr);
    SmallVector<char> nameChars(kernelModName.getValue().begin(),
                                kernelModName.getValue().end());
    nameChars.emplace_back('\0');
    // Kernel name and SPIRV are stored as global strings
    auto name = LLVM::createGlobalString(
        loc, rewriter, str("Name"),
        StringRef(nameChars.data(), nameChars.size()), LLVM::Linkage::Internal);
    auto spirv = LLVM::createGlobalString(loc, rewriter, str("SPIRV"),
                                          binaryAttr.getValue(),
                                          LLVM::Linkage::Internal);
    auto spirvSize = rewriter.create<LLVM::ConstantOp>(
        loc, helper.idxType,
        IntegerAttr::get(helper.idxType,
                         static_cast<int64_t>(binaryAttr.size())));

    SmallVector<Value> gridSize;
    SmallVector<Value> blockSize;
    SmallVector<Value> argSize;
    gridSize.emplace_back(gpuLaunch.getGridSizeX());
    gridSize.emplace_back(gpuLaunch.getGridSizeY());
    gridSize.emplace_back(gpuLaunch.getGridSizeZ());
    blockSize.emplace_back(gpuLaunch.getBlockSizeX());
    blockSize.emplace_back(gpuLaunch.getBlockSizeY());
    blockSize.emplace_back(gpuLaunch.getBlockSizeZ());

    for (auto arg : adaptor.getKernelOperands()) {
      auto type = arg.getType();
      auto size = type.isIntOrFloat() ? type.getIntOrFloatBitWidth() / 8 : 0;
      argSize.emplace_back(helper.idxConstant(rewriter, loc, size));
    }

    auto array = [&](SmallVector<Value> &values) {
      auto size = helper.idxConstant(rewriter, loc, values.size());
      auto arrayPtr = rewriter.create<LLVM::AllocaOp>(loc, helper.ptrType,
                                                      helper.idxType, size);
      for (size_t i = 0, n = values.size(); i < n; i++) {
        auto elementPtr = rewriter.create<LLVM::GEPOp>(
            loc, helper.ptrType, helper.idxType, arrayPtr,
            helper.idxConstant(rewriter, loc, i));
        auto value = values[i];
        if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
          assert(getConstantIntValue(cast.getOperand(0)));
          value = helper.idxConstant(
              rewriter, loc, getConstantIntValue(cast.getOperand(0)).value());
        }
        rewriter.create<LLVM::StoreOp>(loc, value, elementPtr);
      }
      return arrayPtr.getResult();
    };

    auto ctx = function.getArgument(0);
    auto argNum =
        helper.idxConstant(rewriter, loc, adaptor.getKernelOperands().size());
    auto createKernelCall = funcCall(
        rewriter, GPU_OCL_KERNEL_CREATE, helper.ptrType,
        {helper.ptrType, helper.idxType, helper.ptrType, helper.ptrType,
         helper.ptrType, helper.ptrType, helper.idxType, helper.ptrType},
        loc,
        {ctx, spirvSize, spirv, name, array(gridSize), array(blockSize), argNum,
         array(argSize)});
    auto result = createKernelCall.getResult();

    // Save the kernel pointer to the global var using CAS
    auto null = rewriter.create<LLVM::ZeroOp>(loc, helper.ptrType);
    auto ptrPtr = rewriter.create<LLVM::AddressOfOp>(loc, ptr);
    auto casResult = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, ptrPtr, null, result, LLVM::AtomicOrdering::acq_rel,
        LLVM::AtomicOrdering::monotonic);
    auto casFlag = rewriter.create<LLVM::ExtractValueOp>(
        loc, rewriter.getI1Type(), casResult, 1);

    auto body = &function.getBody();
    auto thenBlock = rewriter.createBlock(body);
    auto elseBlock = rewriter.createBlock(body);
    rewriter.setInsertionPointToEnd(&body->front());
    rewriter.create<LLVM::CondBrOp>(loc, casFlag, thenBlock, elseBlock);

    // Then block
    rewriter.setInsertionPointToStart(thenBlock);
    rewriter.create<LLVM::ReturnOp>(loc, result);

    // Else block
    // The kernel has already been created by another thread, destroying this
    // one.
    rewriter.setInsertionPointToStart(elseBlock);
    helper.destroyKernels(rewriter, loc, result);
    result = rewriter.create<LLVM::ExtractValueOp>(loc, helper.ptrType,
                                                   casResult, 0);
    rewriter.create<LLVM::ReturnOp>(loc, result);

    rewriter.setInsertionPointAfter(function);
    return true;
  }
};

struct GpuToGpuOcl final : gc::impl::GpuToGpuOclBase<GpuToGpuOcl> {

  void runOnOperation() override {
    const auto ctx = &getContext();
    const LLVMConversionTarget target(getContext());
    LLVMTypeConverter converter(ctx);
    Helper helper(ctx, converter);
    RewritePatternSet patterns(ctx);

    populateGpuToLLVMConversionPatterns(converter, patterns);
    patterns.insert<ConvertAlloc, ConvertMemcpy, ConvertLaunch, ConvertDealloc>(
        helper);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Add gpuOclDestructor() function that destroys all the kernels
    auto mod = llvm::dyn_cast<ModuleOp>(getOperation());
    assert(mod);
    OpBuilder rewriter(mod.getBody(), mod.getBody()->end());
    auto destruct = rewriter.create<LLVM::LLVMFuncOp>(
        mod.getLoc(), GPU_OCL_MOD_DESTRUCTOR,
        LLVM::LLVMFunctionType::get(helper.voidType, {}),
        LLVM::Linkage::External);
    auto loc = destruct.getLoc();
    rewriter.setInsertionPointToStart(destruct.addEntryBlock(rewriter));
    // Add memory fence
    rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acquire);

    SmallVector<Value> kernelPtrs;
    SmallString<128> strBuf("gcGpuOclKernel_");
    auto strBufStart = strBuf.size();
    kernelPtrs.reserve(helper.kernelNames.size());
    for (auto &name : helper.kernelNames) {
      strBuf.truncate(strBufStart);
      strBuf.append(name);
      strBuf.append("_Ptr");
      auto ptr = mod.lookupSymbol<LLVM::GlobalOp>(strBuf);
      assert(ptr);
      auto ptrVal = rewriter.create<LLVM::LoadOp>(
          loc, helper.ptrType, rewriter.create<LLVM::AddressOfOp>(loc, ptr));
      kernelPtrs.emplace_back(ptrVal);
    }

    helper.destroyKernels(rewriter, loc, kernelPtrs);
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
};
} // namespace