//===-- ValueUtils.cpp - Zero-checking utilities ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace utils {

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val) {
  return matchPattern(val, m_AnyZeroFloat()) || matchPattern(val, m_Zero());
}

// Returns true if the attribute represent "all zeros"
static bool isZeroAttr(Attribute attribute) {
  return TypeSwitch<Attribute, bool>(attribute)
      .Case<FloatAttr>([](auto attr) { return attr.getValueAsDouble() == 0.0; })
      .Case<IntegerAttr>([](auto attr) { return attr.getInt() == 0; })
      .Case<DenseElementsAttr>([](auto attr) {
        if (!attr.getElementType().isIntOrFloat())
          return false;
        if (!attr.isSplat())
          return false;
        auto splat = attr.template getSplatValue<Attribute>();
        return isZeroAttr(splat);
      })
      .Default([](auto attr) { return false; });
}

// Prototypes
static bool isZeroOp(Operation *);

// Returns true if the value represents a zero filled tensor.
// Recurse into isZeroOp for defining ops if not immediately obvious
// Looks past linalg generic's argument (which don't have defining ops)
bool isZeroTensor(Value val) {
  if (!val)
    return false;
  if (isValConstZero(val))
    return true;

  Operation *defOp = nullptr;

  // Block arguments don't have a defining op, but they do have an op arg
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    // We need to find the argument to the linalg on the same order as this one
    auto *linalgOp = arg.getParentRegion()->getParentOp();
    if (!isa<linalg::GenericOp>(linalgOp))
      return false;
    auto index = arg.getArgNumber();
    auto linalgArg = linalgOp->getOperand(index);
    defOp = linalgArg.getDefiningOp();
  } else {
    defOp = val.getDefiningOp();
  }
  return isZeroOp(defOp);
}

// Returns true if the operation represents a zero filled tensor
// Recurses into isZeroTensor for operands and isZeroAttr for attributes
static bool isZeroOp(Operation *defOp) {
  if (!defOp)
    return false;

  return TypeSwitch<Operation *, bool>(defOp)
      .Case<arith::ConstantOp>([&](auto op) {
        // Dense attributes don't match APFloat.isZero()
        auto attr = op.getValue();
        return isZeroAttr(attr);
      })
      .Case<linalg::FillOp, linalg::CopyOp>([&](auto op) {
        if (op.getInputs().size() != 1)
          return false;
        return isZeroTensor(op.getInputs()[0]);
      })
      .Case<memref::CopyOp, memref::SubViewOp, tensor::CastOp,
            tensor::ExtractSliceOp>(
          [&](auto op) { return isZeroTensor(op.getSource()); })
      .Case<memref::GetGlobalOp>([&](auto op) {
        auto name = op.getName();
        auto module = defOp->getParentOfType<ModuleOp>();
        auto global = module.lookupSymbol<memref::GlobalOp>(name);
        auto attr = global.getInitialValueAttr();
        return isZeroAttr(attr);
      })
      .Default([&](Operation *op) { return false; });
}

FailureOr<SmallVector<int64_t>> getStrides(Value value) {
  auto valueType = value.getType();
  if (!isa<MemRefType>(valueType))
    return failure();
  auto memrefType = cast<MemRefType>(valueType);
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset)))
    return failure();
  return strides;
}

FailureOr<SmallVector<int64_t>> getStaticStrides(Value value) {
  auto strides = getStrides(value);
  if (failed(strides))
    return failure();
  if (llvm::any_of(*strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides;
}

std::pair<Value, Value> getPtrAndOffset(OpBuilder &builder, Value operand) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  assert(memrefType && "Expect a memref value");

  Location loc = operand.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  // Insert right after operand producer for better opt chances.
  builder.setInsertionPointAfterValue(operand);

  MemRefType baseMemrefType = MemRefType::get({}, memrefType.getElementType());
  Type basePtrType = builder.getIndexType();
  Type offsetType = builder.getIndexType();
  SmallVector<Type> sizesTypes(memrefType.getRank(), offsetType);
  SmallVector<Type> stridesTypes(memrefType.getRank(), offsetType);
  auto meta = builder.create<memref::ExtractStridedMetadataOp>(
      loc, baseMemrefType, offsetType, sizesTypes, stridesTypes, operand);
  Value alignedPointerAsIndex =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, basePtrType,
                                                             operand);
  Value alignedPointerAsI64 = builder.create<arith::IndexCastOp>(
      loc, builder.getIntegerType(64), alignedPointerAsIndex);
  // TODO: non-POD will require an LLVMTypeConverter.
  Value alignedPointer = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()),
      alignedPointerAsI64);
  Value offset = meta.getOffset();
  return std::make_pair(alignedPointer, offset);
}

Value flattenMemref(PatternRewriter &rewriter, Location loc, Value srcMemref) {
  auto srcType = cast<MemRefType>(srcMemref.getType());

  assert(srcType && "Expected a memref type");

  auto shapeNd = srcType.getShape();
  int64_t flatSize =
      std::accumulate(shapeNd.begin(), shapeNd.end(), 1, std::multiplies<>());

  Value offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value size = rewriter.create<arith::ConstantIndexOp>(loc, flatSize);
  Value stride = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Use memref.reinterpret_cast to flatten the memref
  auto flatMemRefType = MemRefType::get({flatSize}, srcType.getElementType(),
                                        nullptr, srcType.getMemorySpace());
  auto flatMemref =
      rewriter
          .create<memref::ReinterpretCastOp>(loc, flatMemRefType, srcMemref,
                                             offset, size, stride)
          .getResult();
  return flatMemref;
}

bool hasSharedMemSpace(mlir::Value memref) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!type)
    return false;

  auto memSpace = type.getMemorySpace();
  if (!memSpace)
    return false;

  if (auto gpuAttr = mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(memSpace))
    return gpuAttr.getValue() == mlir::gpu::AddressSpace::Private;

  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(memSpace))
    return intAttr.getValue() ==
           static_cast<int64_t>(mlir::gpu::AddressSpace::Private);

  return false;
}

std::tuple<SmallVector<Value>, Value>
computeSubviewOffsets(PatternRewriter &rewriter, Location loc, Value memref) {
  auto fillVal = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto origShape = dyn_cast<MemRefType>(memref.getType()).getShape();

  SmallVector<Value> resolvedOffsets(origShape.size(), fillVal);

  while (auto subViewOp = memref.getDefiningOp<memref::SubViewOp>()) {
    auto currentOffsets = getAsOpFoldResult(resolvedOffsets);
    resolvedOffsets.clear();

    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, memref.getLoc(), subViewOp.getMixedOffsets(),
        subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), currentOffsets,
        resolvedOffsets);
    memref = subViewOp.getOperand(0);
  }

  return std::make_tuple(std::move(resolvedOffsets), memref);
}

SmallVector<OpFoldResult> getMemrefStrides(PatternRewriter &rewriter,
                                           Location loc, Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());

  auto stridedLayout = dyn_cast<StridedLayoutAttr>(type.getLayout());
  if (stridedLayout) {
    auto strides = stridedLayout.getStrides();
    return getMixedValues(strides, {}, rewriter);
  }

  auto sizes = getMixedValues(type.getShape(), {}, rewriter);
  auto strides = memref::computeStridesIRBlock(loc, rewriter, sizes);
  return strides;
}

FailureOr<Value> squeezeMemref(PatternRewriter &rewriter, Location loc,
                               Value memref, size_t maxDims = 2) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  auto shape = type.getShape();

  if (shape.size() <= maxDims)
    return memref;

  for (size_t i = 0; i < shape.size() - maxDims; i++)
    if (shape[i] != 1)
      return failure();

  auto offsets =
      getMixedValues(SmallVector<int64_t>(shape.size(), 0), {}, rewriter);
  auto sizes = getMixedValues(shape, {}, rewriter);
  auto staticStrides = utils::getStaticStrides(memref).value();
  auto strides =
      getMixedValues(SmallVector<int64_t>(shape.size(), 1), {}, rewriter);

  SmallVector<int64_t> newShape(shape.begin() + shape.size() - maxDims,
                                shape.end());
  SmallVector<int64_t> newStrides(
      staticStrides.begin() + shape.size() - maxDims, staticStrides.end());

  int64_t newOffset = 0;
  if (auto memrefLayout = dyn_cast<StridedLayoutAttr>(type.getLayout()))
    newOffset = memrefLayout.getOffset();

  auto newLayout = StridedLayoutAttr::get(
      rewriter.getContext(), /*offset=*/newOffset, /*strides=*/newStrides);
  MemRefType newMemRefType = MemRefType::get(newShape, type.getElementType(),
                                             newLayout, type.getMemorySpace());

  auto squeezedSubview =
      rewriter
          .create<memref::SubViewOp>(loc, newMemRefType, memref, offsets, sizes,
                                     strides)
          .getResult();
  return squeezedSubview;
}

LogicalResult maybeSqueezeDims(PatternRewriter &rewriter,
                               linalg::LinalgOp linalgOp, size_t maxDims) {
  SmallVector<std::pair<size_t, Value>> newOperands;
  auto operands = linalgOp->getOperands();
  auto loc = linalgOp.getLoc();

  for (size_t i = 0; i < operands.size(); i++) {
    auto operand = operands[i];
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) {
      // Skip non-memref operands
      continue;
    }

    if (type.getShape().size() <= maxDims)
      continue;

    auto res = squeezeMemref(rewriter, loc, operand, maxDims);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Can't squeeze memref to the desired number of dimensions");
    }

    auto flatSubview = res.value();
    newOperands.emplace_back(i, flatSubview);
  }

  for (auto [i, operand] : newOperands)
    linalgOp->setOperand(i, operand);

  return success();
}

bool canSqueezeDims(llvm::ArrayRef<int64_t> shape, size_t maxDims) {
  if (shape.size() <= maxDims)
    return true;

  for (size_t i = 0; i < shape.size() - maxDims; i++)
    if (shape[i] != 1)
      return false;

  return true;
}

} // namespace utils
} // namespace mlir
