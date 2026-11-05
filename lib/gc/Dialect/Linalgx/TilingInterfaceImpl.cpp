
#include "IndexingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalgx;

namespace {

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

} // namespace

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
  return getTiledImplementation(builder, offsets, sizes, {});
}

FailureOr<TilingResult> AttentionOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  (void)innerTileAlignments;
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
  return generateResultTileValue(builder, resultNumber, offsets, sizes, {});
}

FailureOr<TilingResult> AttentionOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
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
  return getTiledImplementation(builder, normalizedOffsets, normalizedSizes,
                                innerTileAlignments);
}

//===----------------------------------------------------------------------===//
// GroupedMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<Range> GroupedMatmulOp::getIterationDomain(OpBuilder &b) {
  Location loc = getLoc();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<Range> bounds(4);
  for (Range &r : bounds) {
    r.offset = zero;
    r.stride = one;
  }
  // The row dim is group-relative, so its bound is the largest group. Without
  // the hint, the total row count is the only safe bound. Everything else comes
  // from the operand shapes.
  bounds[kExpertDim].size = getDim(b, loc, getWeights(), 0);
  if (Value maxGroupSize = getMaxGroupSize()) {
    bounds[kMDim].size = maxGroupSize;
  } else {
    bounds[kMDim].size = getDim(b, loc, getLhs(), 0);
  }
  bounds[kNDim].size = getDim(b, loc, getWeights(), 1);
  bounds[kKDim].size = getDim(b, loc, getLhs(), 1);
  return bounds;
}

/// Emits the [start, end) row range of expert `expertIdx` from `offsets`,
/// which holds the end row of each group:
///   start = expertIdx == 0 ? 0 : offsets[expertIdx - 1]
///   end   = offsets[expertIdx]
/// E.g. for offsets = [2, 2, 5]:
///   expertIdx 0 -> [0, 2)
///   expertIdx 1 -> [2, 2)  (empty group)
///   expertIdx 2 -> [2, 5)
static std::pair<Value, Value> emitGroupRange(OpBuilder &b, Location loc,
                                              Value offsets, Value expertIdx) {
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value isFirst =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, expertIdx, zero);
  // Clamped so that the load stays in bounds for expert 0; its value is
  // discarded by the select below.
  Value prevIdx = arith::SelectOp::create(
      b, loc, isFirst, zero, arith::SubIOp::create(b, loc, expertIdx, one));
  auto toIndex = [&](Value v) {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v).getResult();
  };
  Value prev =
      toIndex(tensor::ExtractOp::create(b, loc, offsets, ValueRange{prevIdx}));
  Value start = arith::SelectOp::create(b, loc, isFirst, zero, prev);
  Value end = toIndex(
      tensor::ExtractOp::create(b, loc, offsets, ValueRange{expertIdx}));
  return {start, end};
}

/// Row offset and size of the tile at group-relative offset `mOffset`, clamped
/// to the real length of expert `expertIdx`'s group:
///   row0 = min(start + mOffset, end)
///   size = clamp(mSize, 0, end - row0)
/// A tile past the end of a short group gets size 0 and does nothing, which is
/// what keeps the iteration domain rectangular without padding the rows. `row0`
/// is clamped too, so that even an empty tile addresses a valid row.
static std::pair<OpFoldResult, OpFoldResult>
emitClampedRowTile(OpBuilder &b, Location loc, Value offsets, Value expertIdx,
                   OpFoldResult mOffset, OpFoldResult mSize) {
  auto [start, end] = emitGroupRange(b, loc, offsets, expertIdx);
  Value mOff = getValueOrCreateConstantIndexOp(b, loc, mOffset);
  Value tile = getValueOrCreateConstantIndexOp(b, loc, mSize);

  Value row0 = arith::MinSIOp::create(
      b, loc, arith::AddIOp::create(b, loc, start, mOff), end);
  Value avail = arith::SubIOp::create(b, loc, end, row0);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value size = arith::MaxSIOp::create(
      b, loc, zero, arith::MinSIOp::create(b, loc, tile, avail));
  return {OpFoldResult(row0), OpFoldResult(size)};
}

SmallVector<utils::IteratorType> GroupedMatmulOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction};
}

FailureOr<TilingResult>
GroupedMatmulOp::getTiledImplementation(OpBuilder &builder,
                                        ArrayRef<OpFoldResult> offsets,
                                        ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(builder, offsets, sizes, {});
}

/// Replaces a tile of the grouped matmul with a plain linalg.matmul. The tile
/// covers a single expert, so the expert's weight matrix is just a slice of
/// `weights`, and the rows of the tile are those of the expert's group.
///
/// E.g. for offsets = [2, 2, 5], the tile of expert 2 with group-relative rows
/// [0, 32), N cols [n0, n0 + 64) and the full K = 256 is clamped to the 3 rows
/// of the group, i.e. rows [2, 5):
///
///   %lhs = tensor.extract_slice %lhs[2, 0] [3, 256]
///   %wei = tensor.extract_slice %weights[2, n0, 0] [1, 64, 256]
///            : tensor<ExNx256xf16> to tensor<64x256xf16>
///   %out = tensor.extract_slice %output[2, n0] [3, 64]
///   %res = linalg.matmul indexing_maps = [(m, n, k) -> (m, k),
///                                         (m, n, k) -> (n, k),
///                                         (m, n, k) -> (m, n)]
///            ins(%lhs, %wei) outs(%out)
///
/// The row offset and size are runtime values computed from `offsets`, see
/// emitClampedRowTile().
FailureOr<TilingResult> GroupedMatmulOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  (void)innerTileAlignments;
  assert(offsets.size() == 4 && sizes.size() == 4);

  if (!hasPureTensorSemantics())
    return emitOpError("tiling is only implemented for tensor semantics");

  // One expert per tile: the weight matrix is picked per tile, so the expert
  // dim must be tiled to 1 (it is a leading dim, so the heuristics do that
  // anyway).
  auto expertTile = getConstantIntValue(sizes[kExpertDim]);
  if (!expertTile || *expertTile != 1)
    return emitOpError("expects the expert dimension to be tiled to 1");

  Location loc = getLoc();
  OpFoldResult one = builder.getIndexAttr(1);
  Value expertIdx =
      getValueOrCreateConstantIndexOp(builder, loc, offsets[kExpertDim]);

  // Absolute row range of this tile inside the concatenated rows, clamped to
  // the group's real length.
  auto [rowOffset, rowSize] = emitClampedRowTile(
      builder, loc, getOffsets(), expertIdx, offsets[kMDim], sizes[kMDim]);

  // Drops the leading dim of a [1, x, y] slice type, giving [x, y].
  auto dropLeadingDim = [](RankedTensorType t) {
    return RankedTensorType::get(t.getShape().drop_front(), t.getElementType());
  };

  // lhs [total, K] -> [rows, tile_k] and output [total, N] -> [rows, tile_n]:
  // both natively 2-D, nothing to rank-reduce.
  SmallVector<OpFoldResult> strides2(2, one);
  auto lhsSlice = tensor::ExtractSliceOp::create(
      builder, loc, getLhs(),
      SmallVector<OpFoldResult>{rowOffset, offsets[kKDim]},
      SmallVector<OpFoldResult>{rowSize, sizes[kKDim]}, strides2);
  auto outSlice = tensor::ExtractSliceOp::create(
      builder, loc, getOutput(),
      SmallVector<OpFoldResult>{rowOffset, offsets[kNDim]},
      SmallVector<OpFoldResult>{rowSize, sizes[kNDim]}, strides2);

  // weights [E, N, K] -> [tile_n, tile_k]: the expert dim is sliced to 1 and
  // then dropped. Rank-reducing extract_slice is handled by the tile-and-fuse
  // driver - tensor::replaceExtractSliceWithTiledProducer restores the reduced
  // type after fusing a producer.
  SmallVector<OpFoldResult> strides3(3, one);
  SmallVector<OpFoldResult> weiOffsets{expertIdx, offsets[kNDim],
                                       offsets[kKDim]};
  SmallVector<OpFoldResult> weiSizes{one, sizes[kNDim], sizes[kKDim]};
  auto weiTileType = dropLeadingDim(tensor::ExtractSliceOp::inferResultType(
      cast<RankedTensorType>(getWeights().getType()), weiSizes));
  auto weiSlice = tensor::ExtractSliceOp::create(
      builder, loc, weiTileType, getWeights(), weiOffsets, weiSizes, strides3);

  auto outTileType = cast<RankedTensorType>(outSlice.getResult().getType());

  // With the expert resolved, the tile is a plain matmul with a transposed B:
  //   out[m, n] += lhs[m, k] * wei[n, k]
  MLIRContext *ctx = builder.getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<Attribute> maps = {
      AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, ctx)),
      AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, ctx)),
      AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, ctx))};
  NamedAttribute indexingMaps(builder.getStringAttr("indexing_maps"),
                              builder.getArrayAttr(maps));

  auto matmul = linalg::MatmulOp::create(
      builder, loc, TypeRange{outTileType},
      ValueRange{lhsSlice.getResult(), weiSlice.getResult()},
      ValueRange{outSlice.getResult()}, ArrayRef<NamedAttribute>{indexingMaps});

  return TilingResult{{matmul.getOperation()},
                      SmallVector<Value>(matmul->getResults()),
                      {lhsSlice.getOperation(), weiSlice.getOperation(),
                       outSlice.getOperation()}};
}

LogicalResult GroupedMatmulOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0) return failure();
  Location loc = getLoc();
  Value expertIdx =
      getValueOrCreateConstantIndexOp(builder, loc, offsets[kExpertDim]);
  // The output map is (e, m, n, k) -> (m, n), but `m` is group-relative in the
  // domain and absolute in the tensor, so the row position comes from the same
  // clamp getTiledImplementation() uses rather than from a lookup through the
  // map. CSE merges the two offsets loads.
  auto [rowOffset, rowSize] = emitClampedRowTile(
      builder, loc, getOffsets(), expertIdx, offsets[kMDim], sizes[kMDim]);
  resultOffsets = {rowOffset, offsets[kNDim]};
  resultSizes = {rowSize, sizes[kNDim]};
  return success();
}

FailureOr<TilingResult> GroupedMatmulOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return generateResultTileValue(builder, resultNumber, offsets, sizes, {});
}

FailureOr<TilingResult> GroupedMatmulOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  (void)builder;
  (void)resultNumber;
  (void)offsets;
  (void)sizes;
  (void)innerTileAlignments;
  // Not implementable: a consumer hands us a slice of the *rows* of the result,
  // and recovering the (expert, group-relative row) pair from an absolute row
  // needs a search through `offsets`. Returning failure just means this op is
  // never fused as a producer; it is still tiled as a consumer.
  return failure();
}