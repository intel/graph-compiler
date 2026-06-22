#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Utils/Transform.h"

#include "IndexingUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalgx;

// Default reduction-dim (K2) tile, i.e. the step of the flash-attention k-loop.
static constexpr int64_t kReductionStep = 64;

// Get a dimension of a shaped value as an OpFoldResult (Attribute if static,
// Value via tensor.dim if dynamic).
static OpFoldResult getDimOFR(OpBuilder &b, Location loc, Value v,
                              int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (!t.isDynamicDim(dim)) return b.getIndexAttr(t.getDimSize(dim));
  return tensor::DimOp::create(b, loc, v, dim).getResult();
}

static Value getDimValue(OpBuilder &b, Location loc, Value v, int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (!t.isDynamicDim(dim))
    return arith::ConstantIndexOp::create(b, loc, t.getDimSize(dim));
  return tensor::DimOp::create(b, loc, v, dim);
}

static Value createEmpty(OpBuilder &b, Location loc,
                         ArrayRef<OpFoldResult> sizes, Type elemType) {
  return tensor::EmptyOp::create(b, loc, sizes, elemType);
}

static Value createFilled(OpBuilder &b, Location loc,
                          ArrayRef<OpFoldResult> sizes, Type elemType,
                          Value fillValue) {
  return linalg::FillOp::create(
             b, loc, ValueRange{fillValue},
             ValueRange{createEmpty(b, loc, sizes, elemType)})
      .getResult(0);
}

namespace {
// The roles a decomposed op's iteration dim can play in the attention domain.
// M  = query sequence, N = value head, K1 = query/key head (the QK reduction),
// K2 = key sequence (the k-loop / online-softmax reduction).
enum AttnDim { M, N, K1, K2 };

// The gc.tiling.* attributes of the original attention op plus its tile sizes
// decomposed per attention-domain role (indexed by AttnDim). `enabled` is false
// when the op carries no wg_tile_sizes (nothing to propagate).
struct TileInfo {
  DictionaryAttr srcAttrs;
  std::array<int64_t, 4> tile{0, 0, 0, 0};
  bool enabled = false;
};

// Decompose the attention op's wg_tile_sizes into per-role tiles (M/N/K1/K2).
static TileInfo getTileInfo(AttentionOp attn) {
  TileInfo ti;
  ti.srcAttrs = attn->getDiscardableAttrDictionary();
  auto wgAttr =
      ti.srcAttrs.getAs<DenseI64ArrayAttr>(mlir::gc::GC_ATTR_WG_TILE_SIZES);
  FailureOr<AttentionOpDetail> detail =
      AttentionOpDetail::get(attn.getQueryMap(), attn.getKeyMap(),
                             attn.getValueMap(), attn.getOutputMap());
  if (!wgAttr || failed(detail)) return ti;

  ArrayRef<int64_t> tiles = wgAttr.asArrayRef();
  auto pick = [&](ArrayRef<int64_t> dims) -> int64_t {
    for (int64_t d : dims)
      if (d >= 0 && d < (int64_t)tiles.size() && tiles[d] != 0) return tiles[d];
    return 0;
  };
  ti.tile[M] = pick(detail->getMDims());
  ti.tile[N] = pick(detail->getNDims());
  ti.tile[K1] = pick(detail->getK1Dims());
  // K2 is tiled by the k-loop, so its tile is the loop step.
  ti.tile[K2] = kReductionStep;
  ti.enabled = true;
  return ti;
}

// Tag a generated op: copy the original discardable attrs and set wg_tile_sizes
// from the tiles of `dims` (in op iteration order). An empty `dims` copies the
// original attrs verbatim (e.g. for the k-loop, which keeps the attention op's
// untouched full-domain wg_tile_sizes).
static void tagOp(const TileInfo &ti, Operation *op, ArrayRef<AttnDim> dims) {
  if (!ti.enabled || !op) return;
  if (dims.empty()) {
    for (NamedAttribute na : ti.srcAttrs)
      op->setAttr(na.getName(), na.getValue());
    return;
  }
  for (NamedAttribute na : ti.srcAttrs)
    if (na.getName() != mlir::gc::GC_ATTR_WG_TILE_SIZES)
      op->setAttr(na.getName(), na.getValue());
  SmallVector<int64_t> tiles =
      llvm::map_to_vector(dims, [&](AttnDim d) { return ti.tile[d]; });
  op->setAttr(mlir::gc::GC_ATTR_WG_TILE_SIZES,
              DenseI64ArrayAttr::get(op->getContext(), tiles));
}
} // namespace

FailureOr<SmallVector<Value>> AttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();

  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  Value scale = getScale();
  Value mask = getMask();
  Value output = getOutput();

  auto queryType = cast<ShapedType>(query.getType());
  auto keyType = cast<ShapedType>(key.getType());
  auto outputType = cast<ShapedType>(output.getType());

  ArrayRef<int64_t> queryShape = queryType.getShape();
  ArrayRef<int64_t> keyShape = keyType.getShape();

  Type queryElemType = queryType.getElementType();
  Type outputElemType = outputType.getElementType();
  Type accElemType = b.getF32Type();

  assert((queryShape.size() == 3 || queryShape.size() == 4) &&
         "Query must be 3D or 4D");
  assert(queryShape.size() == keyShape.size() && "Q and K must have same rank");
  assert((outputElemType.isF32() || outputElemType.isF16()) &&
         "Output must be f32 or f16");

  int64_t keySeqDimStatic = keyShape[keyShape.size() - 2];

  // Collapse the leading (unit) batch dims so the body works on 2D tensors.
  SmallVector<ReassociationIndices> reassociation;
  if (queryShape.size() == 3) {
    assert(queryShape[0] == 1 && "Leading dimension must be 1");
    reassociation = {{0, 1}, {2}};
  } else {
    assert(queryShape[0] == 1 && queryShape[1] == 1 &&
           "Leading dimensions must be 1");
    reassociation = {{0, 1, 2}, {3}};
  }

  Value Q_block = tensor::CollapseShapeOp::create(b, loc, query, reassociation);

  // Dynamic-aware sizes of the collapsed problem.
  OpFoldResult seqDim = getDimOFR(b, loc, Q_block, 0);  // M
  OpFoldResult headDim = getDimOFR(b, loc, Q_block, 1); // K1 == N

  TileInfo tileInfo = getTileInfo(*this);
  auto tag = [&](Operation *op, ArrayRef<AttnDim> dims = {}) {
    tagOp(tileInfo, op, dims);
  };

  // Constants (accumulation always in f32 for precision).
  Value minusInf = arith::ConstantOp::create(
      b, loc, accElemType,
      b.getFloatAttr(
          accElemType,
          APFloat::getInf(cast<FloatType>(accElemType).getFloatSemantics(),
                          /*Negative=*/true)));
  Value zero = arith::ConstantOp::create(b, loc, accElemType,
                                         b.getFloatAttr(accElemType, 0.0));

  Value scaleF32 = scale;
  if (scale.getType() != accElemType)
    scaleF32 = arith::ExtFOp::create(b, loc, accElemType, scale);

  // Accumulator inits. These fills precede the k-loop and deliberately get no
  // tiling attributes.
  Value m_init = createFilled(b, loc, {seqDim}, accElemType, minusInf);
  Value l_init = createFilled(b, loc, {seqDim}, accElemType, zero);
  Value acc_init = createFilled(b, loc, {seqDim, headDim}, accElemType, zero);

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value step = arith::ConstantIndexOp::create(b, loc, kReductionStep);
  Value K_collapsed =
      tensor::CollapseShapeOp::create(b, loc, key, reassociation);
  Value seqLen = getDimValue(b, loc, K_collapsed, 0); // K2

  auto forOp = scf::ForOp::create(
      b, loc, c0, seqLen, step, ValueRange{acc_init, m_init, l_init},
      [&](OpBuilder &nb, Location loc, Value k, ValueRange iterArgs) {
        Value acc = iterArgs[0], mPrev = iterArgs[1], lPrev = iterArgs[2];

        // Per-tile K2 block size. Static when the key-seq dim divides the step;
        // otherwise an affine.min so the tail tile / dynamic case is handled
        // and loop-peeling can later fold it to a constant.
        OpFoldResult k2Size;
        if (keySeqDimStatic != ShapedType::kDynamic &&
            keySeqDimStatic % kReductionStep == 0) {
          k2Size = nb.getIndexAttr(kReductionStep);
        } else {
          AffineExpr d0, s0;
          bindDims(nb.getContext(), d0);
          bindSymbols(nb.getContext(), s0);
          AffineMap minMap = AffineMap::get(
              1, 1, {nb.getAffineConstantExpr(kReductionStep), s0 - d0},
              nb.getContext());
          k2Size = affine::AffineMinOp::create(nb, loc, minMap,
                                               ValueRange{k, seqLen})
                       .getResult();
        }

        // Extract and collapse the K/V tiles for this k-step.
        SmallVector<OpFoldResult> offsets, sizes, strides;
        for (size_t i = 0; i < keyShape.size(); i++) {
          bool isSeq = i == keyShape.size() - 2;
          offsets.push_back(isSeq ? OpFoldResult(k) : nb.getIndexAttr(0));
          sizes.push_back(isSeq ? k2Size : getDimOFR(nb, loc, key, i));
          strides.push_back(nb.getIndexAttr(1));
        }
        Value K_block = tensor::CollapseShapeOp::create(
            nb, loc,
            tensor::ExtractSliceOp::create(nb, loc, key, offsets, sizes,
                                           strides),
            reassociation);
        Value V_block = tensor::CollapseShapeOp::create(
            nb, loc,
            tensor::ExtractSliceOp::create(nb, loc, value, offsets, sizes,
                                           strides),
            reassociation);
        OpFoldResult k2 = getDimOFR(nb, loc, K_block, 0);

        // K^T: [k2, K1] -> [K1, k2].
        auto transposeOp = linalg::TransposeOp::create(
            nb, loc, K_block,
            createEmpty(nb, loc, {headDim, k2}, queryElemType),
            ArrayRef<int64_t>{1, 0});
        Value K_T = transposeOp.getResult()[0];
        tag(transposeOp, {K1, K2});

        // QK = Q @ K^T : [M, K1] x [K1, k2] -> [M, k2].
        Value qkInit = createFilled(nb, loc, {seqDim, k2}, accElemType, zero);
        tag(qkInit.getDefiningOp(), {M, K2});
        auto qkMatmul = linalg::MatmulOp::create(
            nb, loc, ValueRange{Q_block, K_T}, ValueRange{qkInit});
        Value qk = qkMatmul.getResult(0);
        tag(qkMatmul, {M, K2, K1}); // matmul iteration order (m, n, k)

        // Scale: qk *= scale.
        Value scaleTensor =
            createFilled(nb, loc, {seqDim, k2}, accElemType, scaleF32);
        tag(scaleTensor.getDefiningOp(), {M, K2});
        auto scaleMul = linalg::MulOp::create(
            nb, loc, ValueRange{qk, scaleTensor},
            createEmpty(nb, loc, {seqDim, k2}, accElemType));
        Value scaled = scaleMul.getResult(0);
        tag(scaleMul, {M, K2});

        // Optional mask add.
        if (mask) {
          Value maskBlock = tensor::ExtractSliceOp::create(
              nb, loc, mask, ArrayRef<OpFoldResult>{nb.getIndexAttr(0), k},
              ArrayRef<OpFoldResult>{getDimOFR(nb, loc, mask, 0), k2Size},
              ArrayRef<OpFoldResult>{nb.getIndexAttr(1), nb.getIndexAttr(1)});
          if (cast<ShapedType>(mask.getType()).getElementType() !=
              accElemType) {
            auto castOp = linalg::GenericOp::create(
                nb, loc,
                TypeRange{
                    createEmpty(nb, loc, {seqDim, k2}, accElemType).getType()},
                ValueRange{maskBlock},
                ValueRange{createEmpty(nb, loc, {seqDim, k2}, accElemType)},
                ArrayRef<AffineMap>{nb.getMultiDimIdentityMap(2),
                                    nb.getMultiDimIdentityMap(2)},
                ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                              utils::IteratorType::parallel},
                [&](OpBuilder &eb, Location el, ValueRange args) {
                  linalg::YieldOp::create(
                      eb, el,
                      arith::ExtFOp::create(eb, el, accElemType, args[0])
                          .getResult());
                });
            maskBlock = castOp.getResult(0);
            tag(castOp, {M, K2});
          }
          auto addOp = linalg::AddOp::create(
              nb, loc, ValueRange{scaled, maskBlock},
              createEmpty(nb, loc, {seqDim, k2}, accElemType));
          scaled = addOp.getResult(0);
          tag(addOp, {M, K2});
        }

        // rowMax = max(scaled, dim=1); mCur = max(mPrev, rowMax).
        Value maxInit = createFilled(nb, loc, {seqDim}, accElemType, minusInf);
        tag(maxInit.getDefiningOp(), {M});
        auto rowMaxOp = linalg::ReduceOp::create(
            nb, loc, scaled, maxInit, ArrayRef<int64_t>{1},
            [&](OpBuilder &rb, Location rl, ValueRange args) {
              linalg::YieldOp::create(
                  rb, rl,
                  arith::MaximumFOp::create(rb, rl, args[0], args[1])
                      .getResult());
            });
        tag(rowMaxOp, {M, K2});
        auto mCurOp = linalg::MaxOp::create(
            nb, loc, ValueRange{mPrev, rowMaxOp.getResult(0)},
            createEmpty(nb, loc, {seqDim}, accElemType));
        Value mCur = mCurOp.getResult(0);
        tag(mCurOp, {M});

        // P = exp(scaled - mCur).
        auto mBcastOp = linalg::BroadcastOp::create(
            nb, loc, mCur, createEmpty(nb, loc, {seqDim, k2}, accElemType),
            ArrayRef<int64_t>{1});
        tag(mBcastOp, {M, K2});
        auto centerOp = linalg::SubOp::create(
            nb, loc, ValueRange{scaled, mBcastOp.getResult()[0]},
            createEmpty(nb, loc, {seqDim, k2}, accElemType));
        tag(centerOp, {M, K2});
        auto expOp = linalg::ExpOp::create(
            nb, loc, centerOp.getResult(0),
            createEmpty(nb, loc, {seqDim, k2}, accElemType));
        Value P = expOp.getResult(0);
        tag(expOp, {M, K2});

        // rowSum = sum(P, dim=1).
        Value sumInit = createFilled(nb, loc, {seqDim}, accElemType, zero);
        tag(sumInit.getDefiningOp(), {M});
        auto rowSumOp = linalg::ReduceOp::create(
            nb, loc, P, sumInit, ArrayRef<int64_t>{1},
            [&](OpBuilder &rb, Location rl, ValueRange args) {
              linalg::YieldOp::create(
                  rb, rl,
                  arith::AddFOp::create(rb, rl, args[0], args[1]).getResult());
            });
        tag(rowSumOp, {M, K2});

        // alpha = exp(mPrev - mCur); lCur = alpha * lPrev + rowSum.
        auto alphaSubOp =
            linalg::SubOp::create(nb, loc, ValueRange{mPrev, mCur},
                                  createEmpty(nb, loc, {seqDim}, accElemType));
        tag(alphaSubOp, {M});
        auto alphaOp =
            linalg::ExpOp::create(nb, loc, alphaSubOp.getResult(0),
                                  createEmpty(nb, loc, {seqDim}, accElemType));
        Value alpha = alphaOp.getResult(0);
        tag(alphaOp, {M});
        auto lScaleOp =
            linalg::MulOp::create(nb, loc, ValueRange{lPrev, alpha},
                                  createEmpty(nb, loc, {seqDim}, accElemType));
        tag(lScaleOp, {M});
        auto lCurOp = linalg::AddOp::create(
            nb, loc, ValueRange{lScaleOp.getResult(0), rowSumOp.getResult(0)},
            createEmpty(nb, loc, {seqDim}, accElemType));
        Value lCur = lCurOp.getResult(0);
        tag(lCurOp, {M});

        // accScaled = acc * alpha (broadcast over N).
        auto alphaBcastOp = linalg::BroadcastOp::create(
            nb, loc, alpha,
            createEmpty(nb, loc, {seqDim, headDim}, accElemType),
            ArrayRef<int64_t>{1});
        tag(alphaBcastOp, {M, N});
        auto accScaleOp = linalg::MulOp::create(
            nb, loc, ValueRange{acc, alphaBcastOp.getResult()[0]},
            createEmpty(nb, loc, {seqDim, headDim}, accElemType));
        tag(accScaleOp, {M, N});

        // P (truncated to query type) @ V, accumulated into accScaled.
        auto pTruncOp = linalg::GenericOp::create(
            nb, loc,
            TypeRange{
                createEmpty(nb, loc, {seqDim, k2}, queryElemType).getType()},
            ValueRange{P},
            ValueRange{createEmpty(nb, loc, {seqDim, k2}, queryElemType)},
            ArrayRef<AffineMap>{nb.getMultiDimIdentityMap(2),
                                nb.getMultiDimIdentityMap(2)},
            ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                          utils::IteratorType::parallel},
            [&](OpBuilder &tb, Location tl, ValueRange args) {
              linalg::YieldOp::create(
                  tb, tl,
                  arith::TruncFOp::create(tb, tl, queryElemType, args[0])
                      .getResult());
            });
        tag(pTruncOp, {M, K2});
        auto pvMatmul = linalg::MatmulOp::create(
            nb, loc, ValueRange{pTruncOp.getResult(0), V_block},
            ValueRange{accScaleOp.getResult(0)});
        tag(pvMatmul, {M, N, K2}); // matmul iteration order (m, n, k)

        scf::YieldOp::create(nb, loc,
                             ValueRange{pvMatmul.getResult(0), mCur, lCur});
      });

  // The loop keeps the attention op's full-domain attrs verbatim.
  tag(forOp);

  // Normalize: out = acc / lFinal (broadcast over N).
  Value acc = forOp.getResult(0);
  Value lFinal = forOp.getResult(2);
  auto lBcastOp = linalg::BroadcastOp::create(
      b, loc, lFinal, createEmpty(b, loc, {seqDim, headDim}, accElemType),
      ArrayRef<int64_t>{1});
  tag(lBcastOp, {M, N});
  auto divOp = linalg::DivOp::create(
      b, loc, ValueRange{acc, lBcastOp.getResult()[0]},
      createEmpty(b, loc, {seqDim, headDim}, accElemType));
  Value result = divOp.getResult(0);
  tag(divOp, {M, N});

  // Truncate to the output element type if needed.
  if (outputElemType != accElemType) {
    auto truncOp = linalg::GenericOp::create(
        b, loc,
        TypeRange{
            createEmpty(b, loc, {seqDim, headDim}, outputElemType).getType()},
        ValueRange{result},
        ValueRange{createEmpty(b, loc, {seqDim, headDim}, outputElemType)},
        ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2),
                            b.getMultiDimIdentityMap(2)},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                      utils::IteratorType::parallel},
        [&](OpBuilder &tb, Location tl, ValueRange args) {
          linalg::YieldOp::create(
              tb, tl,
              arith::TruncFOp::create(tb, tl, outputElemType, args[0])
                  .getResult());
        });
    result = truncOp.getResult(0);
    tag(truncOp, {M, N});
  }

  // Expand back to the original (batched) output shape.
  SmallVector<OpFoldResult> outShape;
  for (int64_t i = 0, r = outputType.getRank(); i < r; ++i) {
    if (i == r - 2) outShape.push_back(seqDim);
    else if (i == r - 1) outShape.push_back(headDim);
    else outShape.push_back(b.getIndexAttr(outputType.getDimSize(i)));
  }
  result = tensor::ExpandShapeOp::create(b, loc, outputType, result,
                                         reassociation, outShape);
  return SmallVector<Value>{result};
}
