//===-- XeVMToLLVM.cpp - XeVM to LLVM dialect conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Conversion/XeVMToLLVM/XeVMToLLVM.h"

#include "gc/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "xevm-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEVMTOLLVMPASS
#include "gc/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace xevm;

namespace {

enum CacheLevel { L1, L2, L3 };

// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_cache_controls.asciidoc#decorations
// Cache Control for a certain level can have different values, depending on
// architecture.
const static std::unordered_map<std::string,
                                std::unordered_map<CacheLevel, int32_t>>
    chipCacheControls{{"pvc", {{CacheLevel::L1, 0}, {CacheLevel::L3, 1}}}};

struct LLVMFuncAttributeOptions {
  bool isConvergent = false;
  bool isNoUnwind = false;
  bool isWillReturn = false;
  LLVM::MemoryEffectsAttr memEffectsAttr{};
};
static constexpr LLVMFuncAttributeOptions noUnwindAttrs = {
    false, true, false, {}};
static constexpr LLVMFuncAttributeOptions noUnwindWillReturnAttrs = {
    false, true, true, {}};

std::string getTypeMangling(Type ty, bool isUnsigned = false) {
  return TypeSwitch<Type, std::string>(ty)
      .Case([isUnsigned](VectorType ty) -> std::string {
        return "Dv" + std::to_string(ty.getNumElements()) + "_" +
               getTypeMangling(ty.getElementType(), isUnsigned);
      })
      .Case([](Float16Type) -> std::string { return "Dh"; })
      .Case([](Float32Type) -> std::string { return "f"; })
      .Case([](Float64Type) -> std::string { return "d"; })
      .Case([isUnsigned](IntegerType ty) -> std::string {
        switch (ty.getWidth()) {
        case 8:
          return isUnsigned ? "h" : "c";
        case 16:
          return isUnsigned ? "t" : "s";
        case 32:
          return isUnsigned ? "j" : "i";
        case 64:
          return isUnsigned ? "m" : "l";
        default:
          llvm_unreachable("unhandled integer type");
        }
      });
}

template <typename OpType>
static std::optional<ArrayAttr>
getCacheControlMetadata(ConversionPatternRewriter &rewriter, OpType op,
                        const bool isLoad, const std::string &chip) {
  if ((op.getL1CacheControlAttr() ==
           L1StoreCacheControlAttr::get(rewriter.getContext(),
                                        L1StoreCacheControl::DEFAULT) &&
       op.getL3CacheControlAttr() ==
           L3StoreCacheControlAttr::get(rewriter.getContext(),
                                        L3StoreCacheControl::DEFAULT)) ||

      (op.getL1CacheControlAttr() ==
           L1LoadCacheControlAttr::get(rewriter.getContext(),
                                       L1LoadCacheControl::DEFAULT) &&
       op.getL3CacheControlAttr() ==
           L3LoadCacheControlAttr::get(rewriter.getContext(),
                                       L3LoadCacheControl::DEFAULT))) {
    return {};
  }
  constexpr int32_t decorationCacheControlArity{4};
  constexpr int32_t loadCacheControlKey{6442};
  constexpr int32_t storeCacheControlKey{6443};
  const int32_t controlKey{isLoad ? loadCacheControlKey : storeCacheControlKey};
  SmallVector<int32_t, decorationCacheControlArity> decorationsL1{
      controlKey, chipCacheControls.at(chip).at(CacheLevel::L1),
      static_cast<int32_t>(op.getL1CacheControl()), 0};
  SmallVector<int32_t, decorationCacheControlArity> decorationsL3{
      controlKey, chipCacheControls.at(chip).at(CacheLevel::L3),
      static_cast<int32_t>(op.getL3CacheControl()), 0};
  auto arrayAttrL1 = rewriter.getI32ArrayAttr(decorationsL1);
  auto arrayAttrL3 = rewriter.getI32ArrayAttr(decorationsL3);

  SmallVector<Attribute, 2> combinedAttrs = {arrayAttrL1, arrayAttrL3};
  return rewriter.getArrayAttr(combinedAttrs);
}

static LLVM::CallOp
createDeviceFunctionCall(ConversionPatternRewriter &rewriter,
                         StringRef funcName, Type retType,
                         ArrayRef<Type> argTypes, ArrayRef<Value> args,
                         ArrayRef<std::pair<unsigned, StringRef>> paramAttrs,
                         LLVMFuncAttributeOptions funcAttributeOptions) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = UnknownLoc::get(ctx);

  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, retType);
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  funcOp.setConvergent(funcAttributeOptions.isConvergent);
  funcOp.setNoUnwind(funcAttributeOptions.isNoUnwind);
  funcOp.setWillReturn(funcAttributeOptions.isWillReturn);

  if (funcAttributeOptions.memEffectsAttr)
    funcOp.setMemoryEffectsAttr(funcAttributeOptions.memEffectsAttr);

  for (auto [idx, attrName] : paramAttrs)
    funcOp.setArgAttr(idx, attrName, rewriter.getUnitAttr());

  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  callOp->setAttrs(funcOp->getAttrs());

  return callOp;
}

template <typename OpType>
class LoadStorePrefetchToOCLPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isLoad = std::is_same_v<OpType, BlockLoad2dOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, BlockPrefetch2dOp>;

    auto loc = op.getLoc();
    VectorType vecType;
    bool vnni = false;
    bool transpose = false;
    if constexpr (isLoad) {
      vecType = op.getRes().getType();
      vnni = op.getVnniTransform();
      transpose = op.getTranspose();
    } else if constexpr (!isPrefetch) {
      vecType = op.getStoredVal().getType();
    }

    auto i32Type = rewriter.getI32Type();
    Value byteCoord =
        rewriter.create<LLVM::UndefOp>(loc, VectorType::get(2, i32Type));
    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(0));
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(1));
    byteCoord = rewriter.create<LLVM::InsertElementOp>(
        loc, VectorType::get(2, i32Type), byteCoord, op.getX(), zero);
    byteCoord = rewriter.create<LLVM::InsertElementOp>(
        loc, VectorType::get(2, i32Type), byteCoord, op.getY(), one);
    SmallVector<Value> args{op.getPtr(), op.getBaseWidth(), op.getBaseHeight(),
                            op.getBasePitch(), byteCoord};
    SmallVector<Type> retTypes;
    Value spvLoadDstPtr;
    std::string funcName{"intel_sub_group_2d_block_"};
    std::string bitWidthId;
    LLVMFuncAttributeOptions funcAttr{noUnwindWillReturnAttrs};
    SmallVector<std::pair<unsigned, StringRef>, 4> paramAttrs;
    if constexpr (isPrefetch) { // Prefetch
      funcName += "prefetch";
      paramAttrs = {std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName())};
      auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
          /*other=*/LLVM::ModRefInfo::NoModRef,
          /*argMem=*/LLVM::ModRefInfo::Ref,
          /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
      auto funcAttrs = noUnwindAttrs;
      funcAttrs.memEffectsAttr = memAttr;
    } else {
      auto vecElemType = vecType.getElementType();
      auto vecElemBitWidth = vecElemType.getIntOrFloatBitWidth();
      Value numElems = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, vecType.getNumElements());
      auto dstOrSrcPtr = rewriter.create<LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext()), vecElemType,
          numElems);
      args.push_back(dstOrSrcPtr);
      if constexpr (isLoad) { // Load
        funcName += "read";
        bitWidthId = getTypeMangling(vecElemType, /*isUnsigned=*/true);
        if (vnni)
          funcName += "_transform";
        else if (transpose)
          funcName += "_transpose";
        spvLoadDstPtr = dstOrSrcPtr;
        retTypes.push_back(vecType);
        paramAttrs = {
            std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(0, LLVM::LLVMDialect::getReadonlyAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getWriteOnlyAttrName()),
        };
      } else { // Store
        funcName += "write";
        bitWidthId = (vecElemBitWidth == 32)
                         ? "j"
                         : ((vecElemBitWidth == 16) ? "t" : "h");
        rewriter.create<LLVM::StoreOp>(loc, op.getStoredVal(), dstOrSrcPtr);
        paramAttrs = {
            std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(0, LLVM::LLVMDialect::getWriteOnlyAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getReadonlyAttrName()),
        };
      }
    }

    funcName =
        llvm::formatv("{0}_{1}b_{2}r{3}x{4}c", funcName, op.getElemSizeInBits(),
                      op.getTileHeight(), op.getTileWidth(), op.getVBlocks())
            .str();
    funcName = llvm::formatv("_Z{0}{1}PU3AS1viiiDv2_i{2}{3}", funcName.size(),
                             funcName, isPrefetch ? "" : "P", bitWidthId)
                   .str();
    SmallVector<Type> argTypes;
    for (auto arg : args) {
      argTypes.push_back(arg.getType());
    }
    LLVM::CallOp call = createDeviceFunctionCall(
        rewriter, funcName, LLVM::LLVMVoidType::get(rewriter.getContext()),
        argTypes, args, paramAttrs, funcAttr);
    // TODO: extract chip from the attached target
    const std::string chip{"pvc"};
    if (std::optional<ArrayAttr> optCacheControls =
            getCacheControlMetadata(rewriter, op, isLoad || isPrefetch, chip)) {
      call->setAttr(XeVMDialect::getCacheControlsAttrName(), *optCacheControls);
    }
    if constexpr (isLoad)
      rewriter.replaceOp(
          op, rewriter.create<LLVM::LoadOp>(loc, vecType, spvLoadDstPtr));
    else
      rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertXeVMToLLVMPass
    : public impl::ConvertXeVMToLLVMPassBase<ConvertXeVMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, XeVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<XeVMDialect>();
    RewritePatternSet patterns(&getContext());
    populateXeVMToLLVMConversionPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::populateXeVMToLLVMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LoadStorePrefetchToOCLPattern<BlockLoad2dOp>,
               LoadStorePrefetchToOCLPattern<BlockStore2dOp>,
               LoadStorePrefetchToOCLPattern<BlockPrefetch2dOp>>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert XeVM to LLVM.
struct XeVMToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateXeVMToLLVMConversionPatterns(patterns);
  }
};
} // namespace

void mlir::registerConvertXeVMToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, XeVMDialect *dialect) {
    dialect->addInterfaces<XeVMToLLVMDialectInterface>();
  });
}
