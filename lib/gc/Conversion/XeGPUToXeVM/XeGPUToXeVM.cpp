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

static int32_t getNumericXeVMAddrSpace(xegpu::MemorySpace xeGpuMemspace) {
  switch (xeGpuMemspace) {
  case xegpu::MemorySpace::Global:
    return static_cast<int>(mlir::xevm::XeVMMemorySpace::kCrossWorkgroup);
  case xegpu::MemorySpace::SLM:
    return static_cast<int>(mlir::xevm::XeVMMemorySpace::kWorkgroup);
  }
  llvm_unreachable("Unknown XeGPU memory space.");
}

template <typename T>
std::tuple<bool, int32_t, int32_t> checkAllLinear(SmallVector<T> denseAttr) {
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

class UpdateNdOffsetToXeVMPattern
    : public OpConversionPattern<UpdateNdOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateNdOffsetOp op, UpdateNdOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto offsets = op.getOffsets();
    auto tdesc = adaptor.getTensorDesc();
    for (size_t offsetDim = 0; offsetDim < offsets.size(); offsetDim++) {
      auto offset = offsets[offsetDim];
      if (auto cst =
              dyn_cast_if_present<arith::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast_if_present<mlir::IntegerAttr>(cst.getValue());
            attr && !attr.getInt())
          continue;
      const int offsetPos =
          static_cast<int>(offsetDim ? NdDescI32Layout::TensorOffsetW
                                     : NdDescI32Layout::TensorOffsetH);
      auto oldOffset =
          rewriter.create<vector::ExtractOp>(loc, tdesc, offsetPos);
      offset = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                     offset);
      auto newOffset = rewriter.create<arith::AddIOp>(loc, oldOffset, offset);
      tdesc =
          rewriter.create<vector::InsertOp>(loc, newOffset, tdesc, offsetPos);
    }
    rewriter.replaceOp(op, tdesc);
    return success();
  }
};

template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                               OpType, LoadNdOp, StoreNdOp>::value>>
class LoadStoreNdToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
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
        ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
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
    int32_t vblocks = 1;
    if (elemBitSize == 16) {
      vblocks = (tileW + 16 - 1) / 16;
      tileW = 16;
    }
    VectorType srcOrDstVecTy = cast<VectorType>(op.getValue().getType());
    if constexpr (std::is_same_v<OpType, LoadNdOp>) {
      const bool vnni = op.getPacked().value_or(false);
      auto transposeValue = op.getTranspose();
      bool transpose =
          transposeValue.has_value() && transposeValue.value()[0] == 1;
      VectorType loadedTy = encodeVectorTypeTo(
          srcOrDstVecTy,
          vnni ? rewriter.getI32Type() : rewriter.getIntegerType(elemBitSize));
      Value resultFlatVec = rewriter.create<xevm::BlockLoad2dOp>(
          loc, loadedTy, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW,
          offsetH, elemBitSize, tileW, tileH, vblocks, transpose, vnni);
      resultFlatVec = rewriter.create<vector::BitCastOp>(
          loc, encodeVectorTypeTo(loadedTy, srcOrDstVecTy.getElementType()),
          resultFlatVec);
      auto newOp = rewriter.create<vector::ShapeCastOp>(loc, srcOrDstVecTy,
                                                        resultFlatVec);
      rewriter.replaceOp(op, newOp);
    } else {
      VectorType srcFlatVecTy = VectorType::get(srcOrDstVecTy.getNumElements(),
                                                srcOrDstVecTy.getElementType());
      Value srcFlatVec = rewriter.create<vector::ShapeCastOp>(loc, srcFlatVecTy,
                                                              op.getValue());
      srcFlatVecTy = encodeVectorTypeTo(srcFlatVecTy,
                                        rewriter.getIntegerType(elemBitSize));
      srcFlatVec =
          rewriter.create<vector::BitCastOp>(loc, srcFlatVecTy, srcFlatVec);
      rewriter.create<xevm::BlockStore2dOp>(
          loc, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW, offsetH,
          elemBitSize, tileW, tileH, vblocks, srcFlatVec);
      rewriter.eraseOp(op);
    }
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
      if (auto denseAttr = cstOp->getAttrOfType<DenseElementsAttr>(
              cstOp.getValueAttrName())) {
        SmallVector<int32_t> intValues;
        for (APInt val : denseAttr.getValues<APInt>())
          intValues.push_back(static_cast<int32_t>(val.getSExtValue()));
        std::tie(allLinear, slope, intercept) = checkAllLinear(intValues);
      } else {
        op.emitError() << "Unknown offsets source, expected a dense array.";
        return failure();
      }
    } else {
      op.emitError()
          << "Unknown offsets source, must be a compile-time constant array.";
      return failure();
    }
    if (!allLinear) {
      op.emitError() << "Expected linear offsets pattern.";
      return failure();
    }

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
    Value sgSize = rewriter.create<arith::ConstantIndexOp>(
        loc, 16); // LaneIdOp doesn't work in llvm-spv
    Value threadId = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value laneId = rewriter.create<arith::RemUIOp>(loc, threadId, sgSize);
    Value laneOffset = rewriter.create<arith::MulIOp>(loc, laneId, offsetSlope);
    laneOffset =
        rewriter.create<arith::AddIOp>(loc, laneOffset, offsetIntercept);
    auto laneAddr =
        rewriter.create<arith::AddIOp>(loc, subGroupAddr, laneOffset);
    rewriter.replaceOp(op, laneAddr);
    return success();
  }
};

class UpdateOffsetToXeVMPattern : public OpConversionPattern<UpdateOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateOffsetOp op, UpdateOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto elemByteSize =
        op.getTensorDesc().getType().getElementType().getIntOrFloatBitWidth() /
        8;
    // Value laneId = rewriter.create<gpu::LaneIdOp>(loc,
    // /*upperBound=*/nullptr);
    Value sgSize = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    Value threadId = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value laneId = rewriter.create<arith::RemUIOp>(loc, threadId, sgSize);
    Value offsetForLane =
        rewriter.create<vector::ExtractOp>(loc, adaptor.getOffsets(), laneId);
    Value factor = rewriter.create<arith::ConstantIndexOp>(loc, elemByteSize);
    offsetForLane = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offsetForLane);
    offsetForLane = rewriter.create<arith::MulIOp>(loc, factor, offsetForLane);
    Value newOffsetForLane = rewriter.create<arith::AddIOp>(
        loc, adaptor.getTensorDesc(), offsetForLane);
    rewriter.replaceOp(op, newOffsetForLane);
    return success();
  }
};

template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                               OpType, LoadGatherOp, StoreScatterOp>::value>>
class LoadStoreToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdesc = op.getTensorDescType();
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdesc.getMemorySpace()));
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);
    VectorType srcOrDstVecTy = cast<VectorType>(op.getValue().getType());
    VectorType srcOrDstFlatVecTy = VectorType::get(
        srcOrDstVecTy.getNumElements(), srcOrDstVecTy.getElementType());
    if constexpr (std::is_same_v<OpType, LoadGatherOp>) {
      Value loaded =
          rewriter.create<LLVM::LoadOp>(loc, srcOrDstFlatVecTy, basePtrLLVM);
      auto newOp =
          rewriter.create<vector::ShapeCastOp>(loc, srcOrDstVecTy, loaded);
      rewriter.replaceOp(op, newOp);
    } else {
      Value srcFlatVec = rewriter.create<vector::ShapeCastOp>(
          loc, srcOrDstFlatVecTy, op.getValue());
      rewriter.create<LLVM::StoreOp>(loc, srcFlatVec, basePtrLLVM);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class PrefetchToXeVMPattern : public OpConversionPattern<PrefetchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrefetchOp op, PrefetchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdescTy = op.getTensorDescType();
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value ptrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);
    auto translateCacheHint = [&](std::optional<xegpu::CachePolicy> hint) {
      auto hintVal =
          hint.has_value() ? hint.value() : xegpu::CachePolicy::UNCACHED;
      switch (hintVal) {
      case xegpu::CachePolicy::CACHED:
        return xevm::LoadCacheControl::C;
      case xegpu::CachePolicy::UNCACHED:
        return xevm::LoadCacheControl::UC;
      case xegpu::CachePolicy::STREAMING:
        return xevm::LoadCacheControl::S;
      case xegpu::CachePolicy::READ_INVALIDATE:
        return xevm::LoadCacheControl::IAR;
        llvm_unreachable("Unsupported cache control.");
      }
    };
    xevm::LoadCacheControl l1 = translateCacheHint(op.getL1Hint());
    xevm::LoadCacheControl l3 = translateCacheHint(op.getL3Hint());
    rewriter.create<xevm::PrefetchOp>(loc, ptrLLVM, xevm::AddrSpace::GLOBAL, l1,
                                      l3);
  }
};
class FenceToXeVMPattern : public OpConversionPattern<FenceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FenceOp op, FenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    xevm::MemoryScope memScope;
    switch (op.getFenceScope()) {
    case xegpu::FenceScope::Workgroup:
      memScope = xevm::MemoryScope::WORKGROUP;
      break;
    case xegpu::FenceScope::GPU:
      memScope = xevm::MemoryScope::GPU;
      break;
      llvm_unreachable("Unknown XeGPU fence scope.");
    }
    xevm::AddrSpace addrSpace;
    switch (op.getMemoryKind()) {
    case xegpu::MemorySpace::Global:
      addrSpace = xevm::AddrSpace::GLOBAL;
      break;
    case xegpu::MemorySpace::SLM:
      addrSpace = xevm::AddrSpace::SHARED;
      break;
      llvm_unreachable("Unknown XeGPU fence scope.");
    }
    rewriter.create<xevm::MemfenceOp>(loc, memScope, addrSpace);
    rewriter.eraseOp(op);
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
                    memref::MemRefDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion([&](IndexType type) -> Type { return type; });
    typeConverter.addConversion([&](VectorType type) -> Type {
      unsigned rank = type.getRank();
      auto elemType = type.getElementType();
      if (llvm::isa<mlir::IndexType>(elemType))
        elemType = mlir::IntegerType::get(&getContext(), 64);
      if (rank < 1 || type.getNumElements() == 1)
        return elemType;
      unsigned sum = 1;
      for (unsigned i = 0; i < rank; i++) {
        sum *= type.getShape()[i];
      }
      return VectorType::get(sum, elemType);
    });
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
                           memref::MemRefDialect, gpu::GPUDialect>();
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
  patterns.add<CreateNdDescToXeVMPattern, UpdateNdOffsetToXeVMPattern,
               LoadStoreNdToXeVMPattern<xegpu::LoadNdOp>,
               LoadStoreNdToXeVMPattern<xegpu::StoreNdOp>>(
      typeConverter, patterns.getContext());
  patterns.add<CreateDescToXeVMPattern, UpdateOffsetToXeVMPattern,
               LoadStoreToXeVMPattern<xegpu::LoadGatherOp>,
               LoadStoreToXeVMPattern<xegpu::StoreScatterOp>>(
      typeConverter, patterns.getContext());
  patterns.add<FenceToXeVMPattern>(typeConverter, patterns.getContext());
}
} // namespace mlir
