//===-- XeVMToLLVM.cpp - XeVM to LLVM dialect conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Conversion/XeGPUToXeVM/XeGPUToXeVM.h"
#include "gc/Dialect/LLVMIR/XeVMDialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "xegpu-to-xevm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEGPUTOXEVMPASS
#include "gc/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace xegpu;

namespace {

enum class NdDescI32Layout : uint32_t {
  BasePtr = 0,
  BaseShapeW = 2,
  BaseShapeH = 3,
  TensorOffsetW = 4,
  TensorOffsetH = 5
};

template <typename T>
std::tuple<bool, int32_t, int32_t> checkAllLinear(ArrayRef<T> denseAttr) {
  assert(!denseAttr.empty());
  const int32_t intercept{static_cast<int32_t>(denseAttr[0])};
  if (denseAttr.size() < 2)
    return {true, 0, intercept};
  const T slope{denseAttr[1] - denseAttr[0]};
  for (size_t i = 1; i < denseAttr.size(); ++i)
    if (denseAttr[i] - denseAttr[i - 1] != slope)
      return {false, 0, 0};
  return {true, static_cast<int32_t>(slope), intercept};
}

mlir::VectorType encodeVectorTypeTo(mlir::VectorType currentVecType,
                                    mlir::Type toElemType) {
  auto elemType = currentVecType.getElementType();
  auto currentBitWidth = elemType.getIntOrFloatBitWidth();
  auto newBitWidth = toElemType.getIntOrFloatBitWidth();
  const int size =
      currentVecType.getNumElements() * currentBitWidth / newBitWidth;
  return mlir::VectorType::get(size, toElemType);
}

class CreateNdDescToXeVMPattern : public OpConversionPattern<CreateNdDescOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNdDescOp op, CreateNdDescOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto resultDesc = cast<TensorDescType>(op.getResult().getType());
    auto sgMap = resultDesc.getSGMapAttr();
    if (!sgMap) {
      op.emitError() << "XeVM expects SGMap attribute to be present for tensor "
                        "descriptors";
      return mlir::failure();
    }
    auto source = op.getSource();
    Type payloadElemTy = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    VectorType payloadTy = VectorType::get(8, payloadElemTy);
    VectorType payloadI64Ty = VectorType::get(4, i64Ty);
    Value payload = rewriter.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(payloadTy, IntegerAttr::get(payloadElemTy, 0)));

    Value baseAddr;
    Value baseShapeW;
    Value baseShapeH;
    Value offsetW;
    Value offsetH;

    if (auto sourceTy = source.getType(); isa<MemRefType>(sourceTy)) {
      baseAddr =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, source);
      auto sourceMemrefTy = cast<MemRefType>(sourceTy);
      if (!sourceMemrefTy.hasStaticShape()) {
        op.emitError() << "Expected static memref shape.";
        return mlir::failure();
      }
      auto rank = sourceMemrefTy.getRank();
      if (rank != 2) {
        op.emitError() << "Expected a 2D memref.";
        return mlir::failure();
      }
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        OpFoldResult ofr = op.getMixedOffsets()[idx];
        if (auto v = llvm::dyn_cast_if_present<Value>(ofr)) {
          val =
              rewriter.create<arith::IndexCastOp>(loc, i64Ty, ofr.get<Value>());
          val = rewriter.create<arith::TruncIOp>(loc, payloadElemTy, val);
        } else {
          int32_t off = llvm::cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
          val = rewriter.create<arith::ConstantIntOp>(loc, off, payloadElemTy);
        }
        return val;
      };
      offsetW = createOffset(rank - 1);
      offsetH = createOffset(rank - 2);
      baseShapeW = rewriter.create<arith::ConstantIntOp>(
          loc, sourceMemrefTy.getDimSize(rank - 1), payloadElemTy);
      baseShapeH = rewriter.create<arith::ConstantIntOp>(
          loc, sourceMemrefTy.getDimSize(rank - 2), payloadElemTy);
    } else if (isa<IntegerType>(sourceTy)) {
      op.emitError()
          << "Integer as source are currently not supported by the pass.";
      return mlir::failure();
    } else {
      op.emitError() << "Unknown source type.";
      return mlir::failure();
    }

    baseAddr = rewriter.create<arith::IndexCastUIOp>(loc, i64Ty, baseAddr);
    Value payLoadAsI64 =
        rewriter.create<vector::BitCastOp>(loc, payloadI64Ty, payload);
    payLoadAsI64 = rewriter.create<vector::InsertOp>(
        loc, baseAddr, payLoadAsI64,
        static_cast<int>(NdDescI32Layout::BasePtr));
    payload = rewriter.create<vector::BitCastOp>(loc, payloadTy, payLoadAsI64);
    payload = rewriter.create<vector::InsertOp>(
        loc, baseShapeW, payload,
        static_cast<int>(NdDescI32Layout::BaseShapeW));
    payload = rewriter.create<vector::InsertOp>(
        loc, baseShapeH, payload,
        static_cast<int>(NdDescI32Layout::BaseShapeH));
    payload = rewriter.create<vector::InsertOp>(
        loc, offsetW, payload,
        static_cast<int>(NdDescI32Layout::TensorOffsetW));
    payload = rewriter.create<vector::InsertOp>(
        loc, offsetH, payload,
        static_cast<int>(NdDescI32Layout::TensorOffsetH));
    rewriter.replaceOp(op, payload);
    return success();
  }
};

class LoadNdToXeVMPattern : public OpConversionPattern<LoadNdOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadNdOp op, LoadNdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto tdesc = adaptor.getTensorDesc();
    auto tdescTy = op.getTensorDescType();

    VectorType payloadI64Ty = VectorType::get(4, rewriter.getI64Type());
    VectorType payloadI32Ty = VectorType::get(8, rewriter.getI32Type());

    Value payLoadAsI64 =
        rewriter.create<vector::BitCastOp>(loc, payloadI64Ty, tdesc);
    Value basePtr = rewriter.create<vector::ExtractOp>(
        loc, payLoadAsI64, static_cast<int>(NdDescI32Layout::BasePtr));

    Value baseShapeW = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeW));
    Value baseShapeH = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeH));
    Value offsetW = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::TensorOffsetW));
    Value offsetH = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::TensorOffsetH));

    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, 1); // TODO : proper address space selection
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtr);

    auto elemType = tdescTy.getElementType();
    const uint32_t elemBitSize = elemType.getIntOrFloatBitWidth();
    Value elemByteSize = rewriter.create<arith::ConstantIntOp>(
        loc, elemBitSize / 8, rewriter.getI32Type());
    Value surfaceW =
        rewriter.create<arith::MulIOp>(loc, baseShapeW, elemByteSize);

    auto tileW = tdescTy.getDimSize(1);
    auto tileH = tdescTy.getDimSize(0);

    const bool vnni = op.getPacked().value_or(false);
    auto transposeValue = op.getTranspose();
    bool transpose =
        transposeValue.has_value() && transposeValue.value()[0] == 1;
    int32_t vblocks = 1;

    VectorType resultTy = cast<VectorType>(op.getValue().getType());
    VectorType loadedTy = encodeVectorTypeTo(
        resultTy,
        vnni ? rewriter.getI32Type() : rewriter.getIntegerType(elemBitSize));
    Value resultFlatVec = rewriter.create<xevm::BlockLoad2dOp>(
        loc, loadedTy, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW,
        offsetH, elemBitSize, tileW, tileH, vblocks, transpose, vnni);
    resultFlatVec = rewriter.create<vector::BitCastOp>(
        loc, encodeVectorTypeTo(loadedTy, resultTy.getElementType()),
        resultFlatVec);
    Value resultVec =
        rewriter.create<vector::ShapeCastOp>(loc, resultTy, resultFlatVec);
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

class StoreNdToXeVMPattern : public OpConversionPattern<StoreNdOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StoreNdOp op, StoreNdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto tdesc = adaptor.getTensorDesc();
    auto tdescTy = op.getTensorDescType();

    VectorType payloadI64Ty = VectorType::get(4, rewriter.getI64Type());
    Value payLoadAsI64 =
        rewriter.create<vector::BitCastOp>(loc, payloadI64Ty, tdesc);
    Value basePtr = rewriter.create<vector::ExtractOp>(
        loc, payLoadAsI64, static_cast<int>(NdDescI32Layout::BasePtr));
    Value baseShapeX = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeW));
    Value baseShapeY = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeH));
    Value offsetX = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::TensorOffsetW));
    Value offsetY = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::TensorOffsetH));

    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, 1); // TODO : proper address space selection
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtr);

    auto elemType = tdescTy.getElementType();
    const uint32_t elemBitSize = elemType.getIntOrFloatBitWidth();
    Value elemByteSize = rewriter.create<arith::ConstantIntOp>(
        loc, elemBitSize / 8, rewriter.getI32Type());
    Value surfaceW =
        rewriter.create<arith::MulIOp>(loc, baseShapeX, elemByteSize);

    auto tileW = tdescTy.getDimSize(1);
    auto tileH = tdescTy.getDimSize(0);

    int32_t vblocks = 1;
    VectorType sourceVecTy = cast<VectorType>(op.getValue().getType());
    VectorType sourceFlatVecTy = VectorType::get(sourceVecTy.getNumElements(),
                                                 sourceVecTy.getElementType());
    Value sourceFlatVec = rewriter.create<vector::ShapeCastOp>(
        loc, sourceFlatVecTy, op.getValue());
    VectorType sourceFlatAsVecI32Ty =
        encodeVectorTypeTo(sourceFlatVecTy, rewriter.getI32Type());
    Value sourceFlatAsVecI32 = rewriter.create<vector::BitCastOp>(
        loc, sourceFlatAsVecI32Ty, sourceFlatVec);
    rewriter.replaceOpWithNewOp<xevm::BlockStore2dOp>(
        op, basePtrLLVM, surfaceW, baseShapeY, surfaceW, offsetX, offsetY,
        elemBitSize, tileW, tileH, vblocks, sourceFlatAsVecI32);
    return success();
  }
};

class CreateDescToXeVMPattern : public OpConversionPattern<CreateDescOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateDescOp op, CreateDescOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto offsets = op.getOffsets();
    bool allLinear{false};
    int32_t slope{0};
    int32_t intercept{0};
    if (auto cstOp = dyn_cast<arith::ConstantOp>(offsets.getDefiningOp())) {
      if (auto denseAttr = cstOp->getAttrOfType<DenseI32ArrayAttr>(
              cstOp.getValueAttrName())) {
        std::tie(allLinear, slope, intercept) =
            checkAllLinear(denseAttr.asArrayRef());
      } else if (auto denseAttr = cstOp->getAttrOfType<DenseI64ArrayAttr>(
                     cstOp.getValueAttrName())) {
        std::tie(allLinear, slope, intercept) =
            checkAllLinear(denseAttr.asArrayRef());
      } else {
        op.emitError()
            << "Unknown offsets source, must be a compile-time constant array.";
      }
    }
    if (!allLinear)
      op.emitError() << "Expected linear offsets pattern.";

    auto memrefTy = cast<MemRefType>(op.getSource().getType());
    Value subGroupAddr =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                op.getSource());
    Value elemByteWidth = rewriter.create<arith::ConstantIndexOp>(
        loc, memrefTy.getElementTypeBitWidth() / 8);
    Value offsetIntercept =
        rewriter.create<arith::ConstantIndexOp>(loc, intercept);
    offsetIntercept =
        rewriter.create<arith::MulIOp>(loc, elemByteWidth, offsetIntercept);
    Value offsetSlope = rewriter.create<arith::ConstantIndexOp>(loc, slope);
    offsetSlope =
        rewriter.create<arith::MulIOp>(loc, elemByteWidth, offsetSlope);
    Value laneId = rewriter.create<gpu::LaneIdOp>(loc, /*upperBound=*/nullptr);
    Value laneOffset = rewriter.create<arith::MulIOp>(loc, laneId, offsetSlope);
    laneOffset =
        rewriter.create<arith::AddIOp>(loc, laneOffset, offsetIntercept);
    auto laneAddr =
        rewriter.create<arith::AddIOp>(loc, subGroupAddr, laneOffset);
    rewriter.replaceOp(op, laneAddr);
    return success();
  }
};

class LoadGatherToXeVMPattern : public OpConversionPattern<LoadGatherOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadGatherOp op, LoadGatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(ctxt, 1);
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);

    VectorType resType = cast<VectorType>(op.getValue().getType());
    VectorType resFlatType =
        VectorType::get(resType.getNumElements(), resType.getElementType());
    VectorType resFlatI32Ty =
        encodeVectorTypeTo(resFlatType, rewriter.getI32Type());
    Value loaded =
        rewriter.create<LLVM::LoadOp>(loc, resFlatI32Ty, basePtrLLVM);
    Value sourceFlatVec =
        rewriter.create<vector::BitCastOp>(loc, resFlatType, loaded);
    sourceFlatVec =
        rewriter.create<vector::ShapeCastOp>(loc, resType, sourceFlatVec);

    rewriter.eraseOp(op);
    return success();
  }
};

class StoreScatterToXeVMPattern : public OpConversionPattern<StoreScatterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StoreScatterOp op, StoreScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(ctxt, 1);
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);

    VectorType srcType = cast<VectorType>(op.getValue().getType());
    VectorType srcFlatType =
        VectorType::get(srcType.getNumElements(), srcType.getElementType());
    Value srcFlatVec =
        rewriter.create<vector::ShapeCastOp>(loc, srcFlatType, op.getValue());
    VectorType srcFlatI32Ty =
        encodeVectorTypeTo(srcFlatType, rewriter.getI32Type());
    Value sourceFlatVec =
        rewriter.create<vector::BitCastOp>(loc, srcFlatI32Ty, srcFlatVec);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, sourceFlatVec, basePtrLLVM);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertXeGPUToXeVMPass
    : public impl::ConvertXeGPUToXeVMPassBase<ConvertXeGPUToXeVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, XeGPUDialect, xevm::XeVMDialect,
                    vector::VectorDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion([&](IndexType type) { return type; });
    typeConverter.addConversion([&](xegpu::TensorDescType type) -> Type {
      if (type.isScattered()) {
        return IndexType::get(&getContext());
      }
      auto i32Type = IntegerType::get(&getContext(), 32);
      return VectorType::get(8, i32Type);
    });

    ConversionTarget target(getContext());
    target.addLegalDialect<xevm::XeVMDialect, LLVM::LLVMDialect,
                           vector::VectorDialect, arith::ArithDialect,
                           memref::MemRefDialect>();
    target.addIllegalDialect<XeGPUDialect>();

    RewritePatternSet patterns(&getContext());
    populateXeGPUToXeVMConversionPatterns(patterns, typeConverter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
namespace mlir {
void populateXeGPUToXeVMConversionPatterns(RewritePatternSet &patterns,
                                           LLVMTypeConverter &typeConverter) {
  patterns.add<CreateNdDescToXeVMPattern, LoadNdToXeVMPattern,
               StoreNdToXeVMPattern>(typeConverter, patterns.getContext());
  patterns.add<CreateDescToXeVMPattern, LoadGatherToXeVMPattern,
               StoreScatterToXeVMPattern>(typeConverter, patterns.getContext());
}
} // namespace mlir
