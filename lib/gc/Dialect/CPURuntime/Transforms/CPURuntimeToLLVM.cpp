//===- CPURuntimeToLLVM.cpp - CPU Runtime To LLVM ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"

namespace mlir::cpuruntime {

void populateCPURuntimeToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns);

#define GEN_PASS_DEF_CPURUNTIMETOLLVM
#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h.inc"

namespace {
static const char formatStringPrefix[] = "cpuprintfFormat_";

static LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp,
                                            const Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

class PrintfRewriter : public ConvertOpToLLVMPattern<PrintfOp> {
public:
  using ConvertOpToLLVMPattern<PrintfOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(PrintfOp op, PrintfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    mlir::Type llvmI32 = typeConverter->convertType(rewriter.getI32Type());
    mlir::Type llvmI64 = typeConverter->convertType(rewriter.getI64Type());
    mlir::Type llvmI8 = typeConverter->convertType(rewriter.getI8Type());
    mlir::Type i8Ptr = LLVM::LLVMPointerType::get(op.getContext());
    auto printfFunc = getOrDefineFunction(
        moduleOp, loc, rewriter, "printf",
        LLVM::LLVMFunctionType::get(llvmI32, {i8Ptr}, /*isVarArg*/ true));

    unsigned stringNumber = 0;
    SmallString<16> stringConstName;
    do {
      stringConstName.clear();
      (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
    } while (moduleOp.lookupSymbol(stringConstName));

    llvm::SmallString<20> formatString(adaptor.getFormat());
    formatString.push_back('\0'); // Null terminate for C
    size_t formatStringSize = formatString.size_in_bytes();

    auto globalType = LLVM::LLVMArrayType::get(llvmI8, formatStringSize);
    LLVM::GlobalOp global;
    {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      global = rewriter.create<LLVM::GlobalOp>(
          loc, globalType,
          /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
          rewriter.getStringAttr(formatString));
    }
    Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
        loc,
        LLVM::LLVMPointerType::get(rewriter.getContext(),
                                   global.getAddrSpace()),
        global.getSymNameAttr());
    Value stringStart = rewriter.create<LLVM::GEPOp>(
        loc, i8Ptr, globalType, globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});
    SmallVector<Value, 5> appendFormatArgs = {stringStart};
    for (auto arg : adaptor.getArgs()) {
      if (auto floatType = dyn_cast<FloatType>(arg.getType())) {
        if (!floatType.isF64())
          arg = rewriter.create<LLVM::FPExtOp>(
              loc, typeConverter->convertType(rewriter.getF64Type()), arg);
      }
      if (arg.getType().getIntOrFloatBitWidth() != 64)
        arg = rewriter.create<LLVM::ZExtOp>(loc, llvmI64, arg);
      appendFormatArgs.push_back(arg);
    }
    rewriter.create<LLVM::CallOp>(loc, printfFunc, appendFormatArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

class CPURuntimeToLLVM : public impl::CPURuntimeToLLVMBase<CPURuntimeToLLVM> {
public:
  using Base::Base;
  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions options(&getContext());
    LLVMTypeConverter converter(&getContext(), options);
    populateCPURuntimeToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

/// Implement the interface to convert MemRef to LLVM.
struct CPURuntimeToDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateCPURuntimeToLLVMConversionPatterns(typeConverter, patterns);
  }
};

} // namespace

void populateCPURuntimeToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns) {
  patterns.add<PrintfRewriter>(converter);
}

void registerConvertCPURuntimeToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    dialect->addInterfaces<CPURuntimeToDialectInterface>();
  });
}

} // namespace mlir::cpuruntime
