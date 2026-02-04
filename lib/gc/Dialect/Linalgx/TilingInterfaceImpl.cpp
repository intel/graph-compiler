
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "IndexingUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// Attention Helpers
//===----------------------------------------------------------------------===//

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return arith::ConstantIndexOp::create(builder, loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.createOrFold<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.createOrFold<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getIndexAttr(t.getDimSize(dim));
}

/// Permutes the offset and size arrays by the result indexes of the provided
/// affine map.
static SmallVector<Range> getPermutedRange(AffineMap permutation,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes) {
  auto one = IntegerAttr::get(IndexType::get(permutation.getContext()), 1);
  assert(permutation.isProjectedPermutation() &&
         "Affine map should be a projected permutation");
  SmallVector<Range> output;
  for (AffineExpr dimExpr : permutation.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    output.push_back(Range{offsets[dim], sizes[dim], one});
  }
  return output;
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
                    ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(src.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return tensor::ExtractSliceOp::create(b, loc, src, offsets, sizes,
                                              strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return memref::SubViewOp::create(b, loc, src, offsets, sizes, strides);
      })
      .Default([&](Type t) -> Operation * {
        assert(false && "invalid type");
        return nullptr;
      });
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<Range> slice) {
  SmallVector<OpFoldResult> offsets =
      llvm::map_to_vector(slice, [](Range x) { return x.offset; });
  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(slice, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> strides =
      llvm::map_to_vector(slice, [](Range x) { return x.stride; });
  return getSlice(b, loc, src, offsets, sizes, strides);
}

static SmallVector<Range>
getAttentionIterationDomain(Location loc, OpBuilder &b, int64_t domainRank,
                            ArrayRef<Value> values,
                            ArrayRef<AffineMap> indexingMaps) {
  SmallVector<Range> loopBounds(domainRank);
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  for (Range &bound : loopBounds) {
    bound.offset = zero;
    bound.stride = one;
  }

  SmallVector<bool> dimsFound(domainRank, false);
  auto fillSizes = [&](Value val, AffineMap indexingMap) {
    for (auto [idx, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      auto dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (dimsFound[pos]) {
        continue;
      }
      dimsFound[pos] = true;
      loopBounds[pos].size = getDim(b, loc, val, idx);
    }
  };

  for (auto [val, indexingMap] : llvm::zip_equal(values, indexingMaps)) {
    fillSizes(val, indexingMap);
  }

  return loopBounds;
}

static SmallVector<utils::IteratorType>
getAttentionIteratorTypes(int64_t domainRank, AffineMap qMap, AffineMap kMap,
                          AffineMap vMap, AffineMap oMap) {
  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(qMap, kMap, vMap, oMap);
  assert(succeeded(maybeOpInfo) && "Failed to infer attention op details");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  // All dimensions other than k1 and k2 are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(domainRank,
                                                 utils::IteratorType::parallel);

  for (auto dim :
       llvm::concat<const int64_t>(opInfo.getK1Dims(), opInfo.getK2Dims())) {
    iteratorTypes[dim] = utils::IteratorType::reduction;
  }

  return iteratorTypes;
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

SmallVector<Range> AttentionOp::getIterationDomain(OpBuilder &b) {
  // Attention shape can be determined from Q, K, V alone.
  SmallVector<Value> shapedValues = {getQuery(), getKey(), getValue()};
  SmallVector<AffineMap> indexingMaps = {getQueryMap(), getKeyMap(),
                                         getValueMap()};
  return getAttentionIterationDomain(getLoc(), b, getIterationDomainRank(),
                                     shapedValues, indexingMaps);
}

SmallVector<utils::IteratorType> AttentionOp::getLoopIteratorTypes() {
  return getAttentionIteratorTypes(getIterationDomainRank(), getQueryMap(),
                                   getKeyMap(), getValueMap(), getOutputMap());
}

FailureOr<TilingResult>
AttentionOp::getTiledImplementation(OpBuilder &builder,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) {
  assert(offsets.size() == static_cast<size_t>(getIterationDomainRank()));
  assert(sizes.size() == static_cast<size_t>(getIterationDomainRank()));

  Location loc = getLoc();

  SmallVector<Range> querySlice =
      getPermutedRange(getQueryMap(), offsets, sizes);
  SmallVector<Range> keySlice = getPermutedRange(getKeyMap(), offsets, sizes);
  SmallVector<Range> valueSlice =
      getPermutedRange(getValueMap(), offsets, sizes);
  SmallVector<Range> outputSlice =
      getPermutedRange(getOutputMap(), offsets, sizes);

  Value scale = getScale();

  SmallVector<Value> tiledOperands;
  SmallVector<Operation *> slices;

  // Query
  {
    Operation *querySliceOp = getSlice(builder, loc, getQuery(), querySlice);
    tiledOperands.emplace_back(querySliceOp->getResult(0));
    slices.push_back(querySliceOp);
  }

  // Key
  {
    Operation *keySliceOp = getSlice(builder, loc, getKey(), keySlice);
    tiledOperands.emplace_back(keySliceOp->getResult(0));
    slices.push_back(keySliceOp);
  }

  // Value
  {
    Operation *valueSliceOp = getSlice(builder, loc, getValue(), valueSlice);
    tiledOperands.emplace_back(valueSliceOp->getResult(0));
    slices.push_back(valueSliceOp);
  }

  // Scale
  tiledOperands.emplace_back(scale);

  // Mask
  Value attnMask = getMask();
  if (attnMask) {
    SmallVector<Range> maskSlice =
        getPermutedRange(*getMaskMap(), offsets, sizes);
    Operation *maskSliceOp = getSlice(builder, loc, attnMask, maskSlice);
    tiledOperands.emplace_back(maskSliceOp->getResult(0));
    slices.push_back(maskSliceOp);
  }

  // Output
  {
    Operation *outputSliceOp = getSlice(builder, loc, getOutput(), outputSlice);
    tiledOperands.emplace_back(outputSliceOp->getResult(0));
    slices.push_back(outputSliceOp);
  }

  SmallVector<Type> resultTypes;
  if (hasPureTensorSemantics()) {
    int64_t baseIdx = attnMask ? 5 : 4;
    resultTypes.push_back(tiledOperands[baseIdx].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp}, SmallVector<Value>(tiledOp->getResults()), slices};
}

LogicalResult AttentionOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.clear();
  resultSizes.clear();

  AffineMap resultIndexingMap;
  switch (resultNumber) {
  case 0:
    resultIndexingMap = getOutputMap();
    break;
  default:
    return failure();
  }

  for (AffineExpr dimExpr : resultIndexingMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    resultOffsets.push_back(offsets[dim]);
    resultSizes.push_back(sizes[dim]);
  }
  return success();
}

FailureOr<TilingResult>
AttentionOp::generateResultTileValue(OpBuilder &builder, unsigned resultNumber,
                                     ArrayRef<OpFoldResult> offsets,
                                     ArrayRef<OpFoldResult> sizes) {
  // Input offsets and sizes here are from the POV of the outputMap. We need to
  // normalize these offsets and size for it to be useful.

  // Initialize normalized offsets with 0s and normalized sizes with original
  // size.
  SmallVector<Range> iterationDomain(getIterationDomain(builder));
  SmallVector<OpFoldResult> normalizedSizes =
      llvm::map_to_vector(iterationDomain, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> normalizedOffsets(getIterationDomainRank(),
                                              builder.getIndexAttr(0));
  ArrayRef<AffineExpr> outputDims = getOutputMap().getResults();
  for (size_t i = 0; i < outputDims.size(); i++) {
    int dim = cast<AffineDimExpr>(outputDims[i]).getPosition();
    normalizedOffsets[dim] = offsets[i];
    normalizedSizes[dim] = sizes[i];
  }
  return getTiledImplementation(builder, normalizedOffsets, normalizedSizes);
}