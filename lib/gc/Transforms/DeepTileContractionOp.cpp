//===-- DeepTileContractionOp.cpp - tile named op deeply --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./TilingUtil.hpp"
#include "gc/Analysis/MatmulConfigAnalysis.h"
#include "gc/Dialect/Linalgx/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Transforms/Passes.h"

#include <llvm/Support/Debug.h>

#include <memory>

#define DEBUG_TYPE "gc-deep-tile-contraction-op"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_DEEPTILECONTRACTIONOP
#include "gc/Transforms/Passes.h.inc"

namespace {

// Util function to tensor view a ranked tensor to another ranked tensor without
// change the data layout
static Value
tensorViewRankedTensor(RewriterBase &rewriter, RankedTensorType outTensorType,
                       Value value,
                       ArrayRef<int64_t> permutation = SmallVector<int64_t>{}) {
  Value result, currentValue = value;
  Location loc = currentValue.getLoc();
  RankedTensorType inTensorType =
      cast<RankedTensorType>(currentValue.getType());
  ArrayRef<int64_t> inShape = inTensorType.getShape();
  ArrayRef<int64_t> outShape = outTensorType.getShape();
  mlir::Type tensorElementType = inTensorType.getElementType();

  // Check if the input and output tensor have the same shape
  if (inShape == outShape)
    return currentValue;

  if (outShape.size() < inShape.size()) {
    SmallVector<ReassociationIndices> reassocIndices;
    uint64_t outIdx = 0UL, inIdx = 0UL;
    while (inIdx < inShape.size() && outIdx < outShape.size()) {
      ReassociationIndices firstEntry;
      int64_t remaining = outShape[outIdx++];
      if (remaining == 1) {
        firstEntry.push_back(inIdx++);
        reassocIndices.push_back(firstEntry);
        continue;
      }
      while (remaining > 1) {
        remaining /= inShape[inIdx];
        firstEntry.push_back(inIdx++);
      }
      reassocIndices.push_back(firstEntry);
    }
    result = rewriter.create<tensor::CollapseShapeOp>(
        loc, outTensorType, currentValue, reassocIndices);
  } else if (outShape.size() > inShape.size()) {
    SmallVector<ReassociationIndices> reassocIndices;
    uint64_t outIdx = 0UL, inIdx = 0UL;
    while (outIdx < outShape.size() && inIdx < inShape.size()) {
      ReassociationIndices firstEntry;
      int64_t remaining = inShape[inIdx++];
      if (remaining == 1) {
        firstEntry.push_back(outIdx++);
        reassocIndices.push_back(firstEntry);
        continue;
      }
      while (remaining > 1) {
        remaining /= outShape[outIdx];
        firstEntry.push_back(outIdx++);
      }
      reassocIndices.push_back(firstEntry);
    }
    result = rewriter.create<tensor::ExpandShapeOp>(
        loc, outTensorType, currentValue, reassocIndices);
  } else {
    result = rewriter.create<tensor::CastOp>(loc, outTensorType, currentValue);
  }

  // Transpose the tensor if permutation is not empty
  if (!permutation.empty()) {
    SmallVector<int64_t> transposeShape;
    for (int64_t idx : permutation)
      transposeShape.push_back(outShape[idx]);
    Operation *initOp = rewriter.create<tensor::EmptyOp>(loc, transposeShape,
                                                         tensorElementType);
    Operation *transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, result, initOp->getResult(0), permutation);
    result = transposeOp->getResult(0);
  }
  return result;
}

// Build the linalg region for a linalg op
static void buildLinalgRegion(Operation *op, bool createTemporaryOp = false) {
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (const Value &opOperand : op->getOperands()) {
    argTypes.push_back(getElementTypeOrSelf(opOperand.getType()));
    argLocs.push_back(opOperand.getLoc());
  }
  size_t initSize = op->getResults().size();
  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  b.setInsertionPointToStart(body);
  if (createTemporaryOp) {
    unsigned argNum = body->getNumArguments();
    SmallVector<Value> vals;
    for (size_t i = initSize; i > 0; i--)
      vals.push_back(body->getArgument(argNum - i));
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(body);
    Location loc = b.getUnknownLoc();
    b.create<linalg::YieldOp>(loc, ValueRange(vals));
  } else {
    linalg::LinalgDialect *dialect =
        static_cast<linalg::LinalgDialect *>(op->getDialect());
    linalg::LinalgDialect::RegionBuilderFunType fun =
        dialect->getRegionBuilder("linalg.matmul");
    fun(b, *body, op->getAttrs());
  }
}

// Check if the linalgOp need to be legalized to f32 accumulation type
static bool needToLegalizeDtype(linalg::LinalgOp linalgOp) {
  mlir::Type dataType =
      dyn_cast<mlir::ShapedType>(linalgOp.getDpsInputs()[0].getType())
          .getElementType();
  mlir::Type resultType =
      dyn_cast<mlir::ShapedType>(linalgOp.getDpsInits()[0].getType())
          .getElementType();
  return (dataType.isBF16() || dataType.isF16()) && dataType == resultType;
}

struct DtypeLegalizeResult {
  Operation *linalgOp = nullptr;
  Operation *castOp = nullptr;
};

// Split a low precision matmul(bf16xbf16->bf16) to a combination
// matmul(bf16xbf16->f32) + cast(f32->bf16)
// if needFurtherFuse=true, a middle temporary linalgOp(bf16xbf16->(f32,bf16))
// will be created
static FailureOr<DtypeLegalizeResult>
matmulDtypeLegalize(RewriterBase &rewriter, Operation *op,
                    bool needCopyInit = true, bool needFurtherFuse = false) {
  linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return failure();

  Location loc = linalgOp->getLoc();
  DtypeLegalizeResult result;

  if (needToLegalizeDtype(linalgOp)) {
    rewriter.setInsertionPoint(linalgOp);
    IRMapping mapping;
    Operation *initOp = linalgOp.getDpsInits()[0].getDefiningOp();
    Value initValue = initOp->getResult(0);
    ShapedType initType = cast<ShapedType>(initValue.getType());
    ArrayRef<int64_t> tensorShape = initType.getShape();
    SmallVector<OpFoldResult> mixedShape;
    for (auto &&[i, t] : llvm::enumerate(tensorShape)) {
      if (initType.isDynamicDim(i)) {
        Value val = rewriter.create<tensor::DimOp>(loc, initValue, i);
        mixedShape.push_back(val);
      } else {
        mixedShape.push_back(getAsIndexOpFoldResult(rewriter.getContext(), t));
      }
    }
    Operation *currentOp;

    currentOp = rewriter.create<tensor::EmptyOp>(
        loc, mixedShape, Float32Type::get(op->getContext()));
    if (needCopyInit)
      currentOp = rewriter.create<linalg::CopyOp>(loc, initOp->getResult(0),
                                                  currentOp->getResult(0));
    SmallVector<Value> newOperands = linalgOp->getOperands();
    Value oldInit = newOperands.back();
    newOperands.back() = currentOp->getResult(0);

    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    indexingMaps.push_back(indexingMaps.back());
    SmallVector<NamedAttribute> attrs(linalgOp->getAttrs());
    SmallVector<Type> types = {currentOp->getResult(0).getType()};
    if (needFurtherFuse) {
      NamedAttribute segmentSize = rewriter.getNamedAttr(
          "operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 2}));
      for (auto &attr : attrs) {
        if (attr.getName() == "indexing_maps")
          attr.setValue(rewriter.getAffineMapArrayAttr(indexingMaps));
        if (attr.getName() == "operandSegmentSizes")
          attr.setValue(segmentSize.getValue());
      }
      types.push_back(oldInit.getType());
      newOperands.push_back(oldInit);
    }
    OperationState state(loc, linalgOp->getName(), newOperands, types, attrs);
    state.addRegion();
    currentOp = rewriter.create(state);
    buildLinalgRegion(currentOp, needFurtherFuse);
    linalg::CopyOp castOp = rewriter.create<linalg::CopyOp>(
        loc, currentOp->getResult(0), initOp->getResult(0));
    result.linalgOp = currentOp;
    result.castOp = castOp;
  }

  return result;
}

// Find the parent fill op of a value and will penetrate pack/pad ops
static Operation *findParentFillOp(Value val) {
  SmallVector<StringRef> skipOpList = {"tensor.pack", "tensor.pad"};
  Operation *currentOp = val.getDefiningOp();
  while (currentOp &&
         llvm::find(skipOpList, currentOp->getName().getStringRef()) !=
             skipOpList.end() &&
         !isa<linalg::FillOp>(currentOp)) {
    currentOp = currentOp->getOperand(0).getDefiningOp();
  }
  if (currentOp && isa<linalg::FillOp>(currentOp))
    return currentOp;
  return nullptr;
}

// Get the parallel dims of a matmul op
static void getMatmulParallelDims(linalg::LinalgOp linalgOp,
                                  unsigned operandIdx,
                                  SmallVectorImpl<unsigned> &dims) {
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(operandIdx));
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  ArrayRef<AffineExpr> results = map.getResults();
  for (const AffineExpr &dim : results) {
    AffineDimExpr dimExpr = dyn_cast<AffineDimExpr>(dim);
    if (dimExpr && iteratorTypes[dimExpr.getPosition()] ==
                       mlir::utils::IteratorType::parallel)
      dims.push_back(dimExpr.getPosition());
  }
}

// set the dynamic size to static size for ExtractSliceOp according to the tile
// config
static void setStaticSizeForExtractSliceOp(RewriterBase &rewriter,
                                           Operation *op, bool isExtract,
                                           SmallVector<int64_t> size,
                                           int shrinDimNum = 0) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {
    SmallVector<OpFoldResult> mixedOffsets = extractSlice.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = extractSlice.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = extractSlice.getMixedStrides();
    auto targetTensor = mlir::RankedTensorType::get(
        SmallVector<int64_t>(size.begin() + shrinDimNum, size.end()),
        extractSlice.getResult().getType().getElementType());
    for (auto &&[i, s] : llvm::enumerate(size))
      mixedSizes[i] = getAsIndexOpFoldResult(rewriter.getContext(), s);
    Operation *newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        extractSlice->getLoc(), extractSlice.getSource(), mixedOffsets,
        mixedSizes, mixedStrides);
    if (shrinDimNum > 0) {
      rewriter.setInsertionPointAfter(newExtractSliceOp);
      Value viewResult = tensorViewRankedTensor(
          rewriter, targetTensor, newExtractSliceOp->getResult(0));
      rewriter.replaceOp(extractSlice, viewResult);
    } else {
      rewriter.replaceOp(extractSlice, newExtractSliceOp);
    }
  }
}

// set the dynamic size to static size for InsertSliceOp according to the tile
// config
static void setStaticSizeForInsertSliceOp(RewriterBase &rewriter, Operation *op,
                                          Value source,
                                          SmallVector<int64_t> size) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(op)) {
    SmallVector<OpFoldResult> mixedOffsets = insertSlice.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSlice.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSlice.getMixedStrides();
    for (auto &&[i, s] : llvm::enumerate(size))
      mixedSizes[i] = getAsIndexOpFoldResult(rewriter.getContext(), s);
    auto targetTensor = mlir::RankedTensorType::get(
        size, insertSlice.getDest().getType().getElementType());
    Value viewResult = tensorViewRankedTensor(rewriter, targetTensor, source);
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        insertSlice, viewResult, insertSlice.getDest(), mixedOffsets,
        mixedSizes, mixedStrides);
  }
}

using InnermostFullResultCallBackFn = std::function<FailureOr<linalg::LinalgOp>(
    RewriterBase &rewriter, Location loc, linalg::LinalgOp linalgop)>;

using FinalReduceCallBackFn = std::function<FailureOr<linalg::LinalgOp>(
    RewriterBase &rewriter, Location loc,
    linalg::ForallReductionTilingResult result)>;

struct OuterLoopGenerationOption {
  enum LoopType { ForOp, ForallOp };
  SmallVector<SmallVector<size_t>> nestedTileSizes;
  SmallVector<LoopType> loopType;
  SmallVector<SmallVector<size_t>> loopDim;
  SmallVector<InnermostFullResultCallBackFn> innermostFullResultCallBacks;
  SmallVector<FinalReduceCallBackFn> finalReduceCallBacks;
  bool isPartialResult = false;
};

struct OuterLoopGenerationResult {
  /// Tiled operations that are generated during tiling. The order does not
  /// matter except the last op. The replacements are expected to be the results
  /// of the last op.
  SmallVector<Operation *> tiledOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  SmallVector<LoopLikeOpInterface> reductionLoops;
};

// Generate outer loop for a linalg op
static FailureOr<OuterLoopGenerationResult>
generateOuterLoop(RewriterBase &b, linalg::LinalgOp linalgOp,
                  const OuterLoopGenerationOption &option) {
  OuterLoopGenerationResult result;
  SmallVector<SmallVector<size_t>> nestedTileSizes = option.nestedTileSizes;
  SmallVector<OuterLoopGenerationOption::LoopType> loopType = option.loopType;
  SmallVector<SmallVector<size_t>> loopDim = option.loopDim;
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  if (loopType.size() != loopDim.size() ||
      loopDim.size() != nestedTileSizes.size())
    return b.notifyMatchFailure(
        linalgOp,
        "loopType, loopDim and nestedTileSizes should have the same size");

  if (linalgOp.hasPureBufferSemantics())
    return b.notifyMatchFailure(
        linalgOp, "currentOp should not has pure buffer semantics");
  linalg::LinalgOp currentOp = linalgOp;

  bool hasFullResult = !option.isPartialResult;
  for (auto &&[i, loopType] : llvm::enumerate(loopType)) {
    ArrayRef<size_t> currentDim = loopDim[i];
    ArrayRef<size_t> currentTileSize = nestedTileSizes[i];
    if (loopType == OuterLoopGenerationOption::LoopType::ForOp) {
      for (auto &&[d, tile] : llvm::zip(currentDim, currentTileSize)) {
        scf::SCFTilingOptions tileOption;
        SmallVector<OpFoldResult> TileSizes(
            currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
        TileSizes[d] = getAsIndexOpFoldResult(b.getContext(), tile);
        tileOption.setTileSizes(TileSizes);
        tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(currentOp);
        if (iteratorTypes[d] == mlir::utils::IteratorType::reduction &&
            tile != 0 && hasFullResult) {
          for (const auto &fn : option.innermostFullResultCallBacks) {
            FailureOr<linalg::LinalgOp> result =
                fn(b, currentOp->getLoc(), currentOp);
            if (succeeded(result))
              currentOp = *result;
          }
          hasFullResult = false;
        }
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
            b, cast<TilingInterface>(currentOp.getOperation()), tileOption);
        if (failed(tilingResult))
          return failure();

        b.replaceOp(currentOp, tilingResult->replacements);
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
        if (iteratorTypes[d] == mlir::utils::IteratorType::reduction)
          result.reductionLoops.push_back(tilingResult->loops.back());
        result.loops.push_back(tilingResult->loops.back());
      }
    } else if (loopType == OuterLoopGenerationOption::LoopType::ForallOp) {
      SmallVector<OpFoldResult> tileSizes(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
      SmallVector<OpFoldResult> threads(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
      SmallVector<unsigned> reductionDims;
      SmallVector<Range> loopRanges =
          cast<TilingInterface>(currentOp.getOperation()).getIterationDomain(b);
      currentOp.getReductionDims(reductionDims);
      bool tileOnReduction = false;
      for (auto &&[d, tile] : llvm::zip(currentDim, currentTileSize)) {
        if (llvm::find(reductionDims, d) != reductionDims.end() && tile != 0 &&
            (!getConstantIntValue(loopRanges[d].size) ||
             tile !=
                 static_cast<size_t>(*getConstantIntValue(loopRanges[d].size))))
          tileOnReduction = true;
        if (llvm::find(reductionDims, d) != reductionDims.end() &&
            !dyn_cast<PartialReductionOpInterface>(currentOp.getOperation())) {
          tileSizes[d] = getAsIndexOpFoldResult(b.getContext(), 0);
          tileOnReduction = false;
        } else {
          tileSizes[d] = getAsIndexOpFoldResult(b.getContext(), tile);
        }
      }

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(currentOp);
      if (tileOnReduction) {
        for (auto &&[idx, tile] : llvm::enumerate(tileSizes))
          if (isConstantIntValue(tile, 0) &&
              llvm::find(reductionDims, idx) != reductionDims.end())
            tileSizes[idx] = loopRanges[idx].size;
        SmallVector<OpFoldResult> newParallelDims;
        for (auto iter : llvm::enumerate(reductionDims))
          newParallelDims.push_back(
              getAsIndexOpFoldResult(b.getContext(), iter.index()));
        FailureOr<linalg::ForallReductionTilingResult> tilingResult =
            linalgX::tileReductionUsingForall(
                b, cast<PartialReductionOpInterface>(currentOp.getOperation()),
                {}, tileSizes, newParallelDims, std::nullopt);
        if (failed(tilingResult) &&
            llvm::hasSingleElement(tilingResult->parallelTiledOps))
          return failure();
        currentOp =
            dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOps.back());
        if (!tilingResult->mergeOps.empty()) {
          for (const auto &fn : option.finalReduceCallBacks) {
            FailureOr<linalg::LinalgOp> result =
                fn(b, currentOp->getLoc(), *tilingResult);
            if (succeeded(result))
              currentOp = *result;
          }
        }
      } else {
        scf::SCFTilingOptions tileOption;
        tileOption.setTileSizes(tileSizes);
        tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
            b, cast<TilingInterface>(currentOp.getOperation()), tileOption);
        if (failed(tilingResult))
          return failure();
        b.replaceOp(currentOp, tilingResult->replacements);
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
      }
    }
  }
  result.tiledOps.emplace_back(currentOp);
  return result;
}

// Turn a OpFoldResult into a Value
static Value turnOpFoldResultIntoValue(RewriterBase &rewriter, Location loc,
                                       OpFoldResult result) {
  if (auto value = dyn_cast<Value>(result))
    return value;
  if (auto attr = dyn_cast<Attribute>(result)) {
    if (auto val = dyn_cast<IntegerAttr>(attr)) {
      if (val.getType().isIndex())
        return rewriter.create<arith::ConstantIndexOp>(loc, val.getInt());
      else
        return rewriter.create<arith::ConstantIntOp>(loc, val.getInt(),
                                                     val.getType());
    }
  }
  return Value();
}

/*
matmul(A, B) -> C
---------------->
forall([PM, PN, PK]: [MThreads, NThreads, KThreads]) {
    CSlice = [KThreads, PM * MOuterBlock: (PM + 1) * MOuterBlock,
     PN * NOuterBlock: (PN + 1) * NOuterBlock]
    ASlice = A[PM * MOuterBlock: (PM + 1) * MOuterBlock, PK * KOuterBlock * (PK
+ 1) * KOuterBlock]
    BSlice = B[PK * KOuterBlock * (PK + 1) * KOuterBlock, PN *
NOuterBlock: (PN + 1) * NOuterBlock] CSlice2 = CSlice[PK, PM * MOuterBlock: (PM
+ 1) * MOuterBlock, PN * NOuterBlock: (PN + 1) * NOuterBlock]
    MNumBlock = MOuterBlock / MBlock
    NNumBlock = NOuterBlock / NBlock
    KNumBlock = KOuterBlock / KBlovk
    for([om, on, ok]: [MNumBlock, NNumBlock, KNumBlock]) {
      ASlice2 = ASlice[om * MBlock: (om + 1) * MBlock, ok * KBlock: (ok + 1) *
KBlock]
      BSlice2 = BSlice[0, ok * KBlock: (ok + 1) * KBlock, on * NBlock: (on +
1) * NBlock]
      CSlice3 = CSlice2[0, om * MBlock: (om + 1) * MBlock, on * NBlock:
(on + 1) * NBlock] (init with 0 when ok == 0)
      MNumInnerBlock = MBlock / iim_block_
      ...
      for([im, in]: [MNumInnerBlock, NNumInnerBlock]) {
        ASlice3 = ASlice2[im * iim_block_: (im + 1) * iim_block_, :]
        BSlice3 = BSlice2[0, im * iim_block_: (im + 1) * iim_block_, :]
        CSlice4 = CSlice3[0, im * iim_block_: (im + 1) * iim_block_, in *
iin_block_: (in + 1) * iin_block_] (init with 0 when ok == 0)
        brgemm(bs=KNumInnerBlock, M=iim_block_, N=iin_block_, K=iik_block,
A=ASlice3, B=BSlice3, C=CSlice4, onlyUpdate=(ok!=0));
      }
    }
}
C = final_reduce(CSlice)
*/
struct DeepTileMatmul : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  static FailureOr<OuterLoopGenerationResult>
  outerLoopGeneration(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      gc::MatmulConfig cfg, bool hasFillOp) {
    SmallVector<unsigned> KDimPos, MDimPos, NDimPos;
    linalgOp.getReductionDims(KDimPos);
    getMatmulParallelDims(linalgOp, 0, MDimPos);
    getMatmulParallelDims(linalgOp, 1, NDimPos);
    OuterLoopGenerationOption option;

    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();
    SmallVector<Range> loopRange =
        cast<TilingInterface>(linalgOp.getOperation())
            .getIterationDomain(rewriter);
    size_t KFirstDim = *getConstantIntValue(loopRange[KDimPos[0]].size);
    size_t MFirstDim = *getConstantIntValue(loopRange[MDimPos[0]].size);
    size_t NFirstDim = *getConstantIntValue(loopRange[NDimPos[0]].size);

    size_t KParallelBlockSize =
        cfg.KThreads == 1
            ? 0
            : (KDimPos.size() > 1
                   ? llvm::divideCeil(KFirstDim, cfg.KThreads)
                   : llvm::divideCeil(llvm::divideCeil(KFirstDim, cfg.KBlock),
                                      cfg.KThreads) *
                         cfg.KBlock);
    size_t MParallelBlockSize =
        MDimPos.size() > 1
            ? llvm::divideCeil(MFirstDim, cfg.MThreads)
            : llvm::divideCeil(llvm::divideCeil(MFirstDim, cfg.MBlock),
                               cfg.MThreads) *
                  cfg.MBlock;
    size_t NParallelBlockSize =
        NDimPos.size() > 1
            ? llvm::divideCeil(NFirstDim, cfg.NThreads)
            : llvm::divideCeil(llvm::divideCeil(NFirstDim, cfg.NBlock),
                               cfg.NThreads) *
                  cfg.NBlock;
    size_t KOuterBlockSize = KDimPos.size() > 1
                                 ? (cfg.KBlock - 1) / cfg.innerMostKBlock + 1
                                 : cfg.KBlock;
    size_t MOuterBlockSize = MDimPos.size() > 1
                                 ? (cfg.MBlock - 1) / cfg.innerMostMBlock + 1
                                 : cfg.MBlock;
    size_t NOuterBlockSize = NDimPos.size() > 1
                                 ? (cfg.NBlock - 1) / cfg.innerMostNBlock + 1
                                 : cfg.NBlock;

    // Outer loop tile size
    for (auto &&[tile, dim] :
         llvm::zip(SmallVector<size_t>{KParallelBlockSize, MParallelBlockSize,
                                       NParallelBlockSize},
                   SmallVector<size_t>{KDimPos[0], MDimPos[0], NDimPos[0]})) {
      option.nestedTileSizes.emplace_back(SmallVector<size_t>{tile});
      option.loopType.emplace_back(
          OuterLoopGenerationOption::LoopType::ForallOp);
      option.loopDim.emplace_back(SmallVector<size_t>{dim});
    }

    // Middle loop tile size
    for (auto &&[tile, dim] :
         llvm::zip(SmallVector<size_t>{MOuterBlockSize, NOuterBlockSize,
                                       KOuterBlockSize},
                   SmallVector<size_t>{MDimPos[0], NDimPos[0], KDimPos[0]})) {
      option.nestedTileSizes.emplace_back(SmallVector<size_t>{tile});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{dim});
    }
    if (llvm::hasSingleElement(KDimPos)) {
      option.nestedTileSizes.emplace_back(SmallVector<size_t>{cfg.KBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{KDimPos.back()});
    }
    // Inner loop tile size
    if (llvm::hasSingleElement(MDimPos)) {
      option.nestedTileSizes.emplace_back(
          SmallVector<size_t>{cfg.innerMostMBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{MDimPos.back()});
    }
    if (llvm::hasSingleElement(NDimPos)) {
      option.nestedTileSizes.emplace_back(
          SmallVector<size_t>{cfg.innerMostNBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{NDimPos.back()});
    }
    for (size_t dim = 0UL; dim < linalgOp.getNumLoops(); ++dim) {
      if (dim != MDimPos.back() && dim != NDimPos.back() &&
          iteratorTypes[dim] != mlir::utils::IteratorType::reduction) {
        option.nestedTileSizes.emplace_back(SmallVector<size_t>{1});
        option.loopType.emplace_back(
            OuterLoopGenerationOption::LoopType::ForOp);
        option.loopDim.emplace_back(SmallVector<size_t>{dim});
      }
    }

    // cast the low precision matmul to f32 when partial accumulation(result not
    // full) is needed
    auto lowPrecisionCast =
        [&](RewriterBase &rewriter, Location loc,
            linalg::LinalgOp linalgop) -> FailureOr<linalg::LinalgOp> {
      FailureOr<DtypeLegalizeResult> legalizedResult = matmulDtypeLegalize(
          rewriter, linalgop.getOperation(), !hasFillOp, true);
      if (succeeded(legalizedResult) && legalizedResult->castOp &&
          legalizedResult->linalgOp) {
        Operation *linalgOp = legalizedResult->linalgOp;
        rewriter.replaceOp(linalgop,
                           linalgOp->getResult(linalgOp->getNumResults() - 1));
        return dyn_cast<linalg::LinalgOp>(linalgOp);
      }
      return failure();
    };
    option.innermostFullResultCallBacks.push_back(lowPrecisionCast);

    if (hasFillOp) {
      auto removeReduncantFill =
          [&](RewriterBase &rewriter, Location loc,
              const linalg::ForallReductionTilingResult &result)
          -> FailureOr<linalg::LinalgOp> {
        ArrayRef<Value> initValue = result.initialValues;
        if (llvm::hasSingleElement(initValue) &&
            isa<linalg::FillOp>(initValue[0].getDefiningOp()))
          rewriter.replaceOp(initValue[0].getDefiningOp(),
                             dyn_cast<DestinationStyleOpInterface>(
                                 initValue[0].getDefiningOp())
                                 .getDpsInits()[0]);
        return dyn_cast<linalg::LinalgOp>(result.parallelTiledOps.back());
      };
      option.finalReduceCallBacks.push_back(removeReduncantFill);
    }

    return generateOuterLoop(rewriter, linalgOp, option);
  }

  struct innerBodyGenerationOption {
    Operation *fillOp;
    bool needLowPrecisionCast;
    SmallVector<LoopLikeOpInterface> KLoopHandles;
    MatmulConfig cfg;
  };

  LogicalResult innerBodyGeneration(RewriterBase &rewriter,
                                    linalg::LinalgOp originOp,
                                    linalg::LinalgOp currentOp,
                                    innerBodyGenerationOption &option) const {
    Location loc = currentOp->getLoc();
    FailureOr<SmallVector<SmallVector<DimType>>> operandDimTypes =
        getOprandDimType(originOp);
    MatmulConfig &cfg = option.cfg;
    ArrayRef<int64_t> AShape =
        originOp.getShape(originOp.getDpsInputOperand(0));
    ArrayRef<int64_t> BShape =
        originOp.getShape(originOp.getDpsInputOperand(1));
    ArrayRef<int64_t> CShape = originOp.getShape(originOp.getDpsInitOperand(0));

    if (failed(operandDimTypes))
      return failure();

    size_t MDimNum = std::count_if((*operandDimTypes)[0].begin(),
                                   (*operandDimTypes)[0].end(),
                                   [](DimType d) { return d == DimType::M; });
    size_t NDimNum = std::count_if((*operandDimTypes)[1].begin(),
                                   (*operandDimTypes)[1].end(),
                                   [](DimType d) { return d == DimType::N; });
    // TODO: support plain in/block out format
    // Calculate the innermost block size according to the config
    SmallVector<int64_t> AInnermostDims, BInnermostDims, CInnermostDims;
    bool firstM = true, firstK = true, firstN = true;
    if (MDimNum > 1) {
      for (auto &&[idx, iter] : llvm::enumerate((*operandDimTypes)[0])) {
        if (iter == DimType::M && firstM) {
          AInnermostDims.push_back(1);
          firstM = false;
        } else if (iter == DimType::Batch) {
          AInnermostDims.push_back(1);
        } else if (iter == DimType::K && firstK) {
          AInnermostDims.push_back(cfg.KBlock / cfg.innerMostKBlock);
          firstK = false;
        } else {
          AInnermostDims.push_back(AShape[idx]);
        }
      }
      firstM = true;
      firstN = true;
      for (auto &&[idx, iter] : llvm::enumerate((*operandDimTypes)[2])) {
        if (iter == DimType::M && firstM) {
          CInnermostDims.push_back(1);
          firstM = false;
        } else if (iter == DimType::Batch) {
          CInnermostDims.push_back(1);
        } else if (iter == DimType::N && firstN) {
          CInnermostDims.push_back(1);
          firstN = false;
        } else {
          CInnermostDims.push_back(CShape[idx]);
        }
      }
    } else {
      AInnermostDims = SmallVector<int64_t>{cfg.innerMostMBlock,
                                            cfg.KBlock / cfg.innerMostKBlock *
                                                cfg.innerMostKBlock};
      CInnermostDims =
          SmallVector<int64_t>{cfg.innerMostMBlock, cfg.innerMostNBlock};
    }

    if (NDimNum > 1) {
      firstN = true;
      firstK = true;
      for (auto &&[idx, iter] : llvm::enumerate((*operandDimTypes)[1])) {
        if (iter == DimType::N && firstN) {
          BInnermostDims.push_back(1);
          firstN = false;
        } else if (iter == DimType::Batch) {
          BInnermostDims.push_back(1);
        } else if (iter == DimType::K && firstK) {
          BInnermostDims.push_back(cfg.KBlock / cfg.innerMostKBlock);
          firstK = false;
        } else {
          BInnermostDims.push_back(BShape[idx]);
        }
      }
    } else {
      BInnermostDims = SmallVector<int64_t>{cfg.KBlock / cfg.innerMostKBlock *
                                                cfg.innerMostKBlock,
                                            cfg.innerMostNBlock};
    }
    // Get the data/wei/dst data type
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(currentOp);
    mlir::Type dataType =
        dyn_cast<mlir::ShapedType>(currentOp.getDpsInputs()[0].getType())
            .getElementType();
    mlir::Type weightType =
        dyn_cast<mlir::ShapedType>(currentOp.getDpsInputs()[1].getType())
            .getElementType();
    mlir::Type resultType =
        dyn_cast<mlir::ShapedType>(currentOp.getDpsInits()[0].getType())
            .getElementType();

    // update the extractSlice to static size, replace it with
    // useBlockedLayout when
    setStaticSizeForExtractSliceOp(rewriter,
                                   currentOp.getDpsInputs()[1].getDefiningOp(),
                                   true, BInnermostDims, NDimNum > 1);
    setStaticSizeForExtractSliceOp(rewriter,
                                   currentOp.getDpsInputs()[0].getDefiningOp(),
                                   true, AInnermostDims, MDimNum > 1);
    for (const Value &init : currentOp.getDpsInits()) {
      setStaticSizeForExtractSliceOp(rewriter, init.getDefiningOp(), true,
                                     CInnermostDims, MDimNum > 1 ? 2 : 0);
    }

    // View the tensor to brgemm required format
    Value dataOprand = tensorViewRankedTensor(
        rewriter,
        mlir::RankedTensorType::get(
            MDimNum > 1 ? SmallVector<int64_t>(AInnermostDims.begin() + 1,
                                               AInnermostDims.end())
                        : SmallVector<int64_t>{cfg.innerMostMBlock,
                                               cfg.KBlock / cfg.innerMostKBlock,
                                               cfg.innerMostKBlock},
            dataType),
        currentOp.getDpsInputs()[0],
        MDimNum == 1 ? SmallVector<int64_t>{1, 0, 2} : SmallVector<int64_t>{});
    Value weightOprand = tensorViewRankedTensor(
        rewriter,
        mlir::RankedTensorType::get(
            NDimNum > 1 ? SmallVector<int64_t>(BInnermostDims.begin() + 1,
                                               BInnermostDims.end())
                        : SmallVector<int64_t>{cfg.KBlock / cfg.innerMostKBlock,
                                               cfg.innerMostKBlock,
                                               cfg.innerMostNBlock},
            weightType),
        currentOp.getDpsInputs()[1]);
    Value resultOprand = tensorViewRankedTensor(
        rewriter,
        mlir::RankedTensorType::get(
            SmallVector<int64_t>(CInnermostDims.begin() + (MDimNum > 1 ? 2 : 0),
                                 CInnermostDims.end()),
            resultType),
        currentOp.getDpsInits()[0]);

    // Create the brgemm op and replace the origin linalg op
    linalg::LinalgOp matmul;
    if (dyn_cast<mlir::ShapedType>(weightOprand.getType()).getShape().size() ==
        3) {
      matmul = rewriter.create<linalg::BatchReduceMatmulOp>(
          loc, resultOprand.getType(), ValueRange{dataOprand, weightOprand},
          resultOprand);
    } else {
      auto inputRange = ValueRange{dataOprand, weightOprand};
      auto resRange = ValueRange{resultOprand};
      auto res = linalgx::makeGenericPackedMatmulOp(
          rewriter, loc, linalgx::PackingType::VNNI_BRMM3D, inputRange,
          resRange);
      if (succeeded(res))
        matmul = *res;
      else
        return failure();
    }

    Value result = matmul.getOperation()->getResult(0);

    // Insert the result back to the original tensor
    for (Operation *user : currentOp->getResult(0).getUsers())
      setStaticSizeForInsertSliceOp(rewriter, user, result, CInnermostDims);

    if (option.needLowPrecisionCast) {
      // fuse the low precision cast to the innermost body
      rewriter.setInsertionPointAfter(currentOp);
      Value cond;
      for (LoopLikeOpInterface &loop : option.KLoopHandles) {
        Value induceVar = turnOpFoldResultIntoValue(
            rewriter, loc, *loop.getSingleInductionVar());
        Value upBound = turnOpFoldResultIntoValue(rewriter, loc,
                                                  *loop.getSingleUpperBound());
        Value step =
            turnOpFoldResultIntoValue(rewriter, loc, *loop.getSingleStep());
        Value currentCond =
            rewriter.create<arith::AddIOp>(loc, induceVar, step);
        currentCond = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, currentCond, upBound);
        cond = cond ? rewriter.create<arith::AndIOp>(loc, cond, currentCond)
                    : currentCond;
      }
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(
          loc, TypeRange{currentOp.getDpsInits().back().getType()},
          cond ? cond : rewriter.create<arith::ConstantIntOp>(loc, true, 1),
          true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Region &region = ifOp.getThenRegion();
        rewriter.setInsertionPointToStart(&region.back());
        linalg::CopyOp castOp = rewriter.create<linalg::CopyOp>(
            loc, matmul->getResult(0), currentOp.getDpsInits().back());
        rewriter.create<scf::YieldOp>(loc, castOp->getResult(0));
      }
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Region &region = ifOp.getElseRegion();
        rewriter.setInsertionPointToStart(&region.back());
        rewriter.create<scf::YieldOp>(loc, currentOp.getDpsInits().back());
      }
      // set static size for the insertSliceOp of copyOp
      for (Operation *user : currentOp->getResult(1).getUsers())
        setStaticSizeForInsertSliceOp(rewriter, user, ifOp->getResult(0),
                                      CInnermostDims);
      rewriter.replaceOp(currentOp, {matmul->getResult(0), ifOp->getResult(0)});
    } else {
      rewriter.replaceOp(currentOp, matmul->getResult(0));
    }
    currentOp = matmul;

    // Fuse the fill op to the innermost body
    if (auto fillOp = llvm::dyn_cast_or_null<linalg::FillOp>(option.fillOp)) {
      Value fillValue = fillOp.getDpsInputs()[0];
      if (cfg.KThreads <= 1)
        // if use k slicing, the fill op is still need to be kept for the reduce
        // init
        rewriter.replaceUsesWithIf(fillOp.getResult(0), fillOp.getDpsInits()[0],
                                   [&](OpOperand &operand) {
                                     return isa<LoopLikeOpInterface>(
                                         operand.getOwner());
                                   });

      rewriter.setInsertionPointAfter(currentOp);
      Value cond;
      arith::ConstantIndexOp zeroConst =
          rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (LoopLikeOpInterface &loop : option.KLoopHandles) {
        Value induceVar = loop.getLoopRegions().front()->front().getArgument(0);
        Value currentCond = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, induceVar, zeroConst);
        cond = cond ? rewriter.create<arith::AndIOp>(loc, cond, currentCond)
                    : currentCond;
      }
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(
          loc, TypeRange{currentOp.getDpsInits()[0].getType()},
          cond ? cond : rewriter.create<arith::ConstantIntOp>(loc, true, 1),
          true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Region &region = ifOp.getThenRegion();
        rewriter.setInsertionPointToStart(&region.back());
        linalg::FillOp fillOp = rewriter.create<linalg::FillOp>(
            loc, fillValue, currentOp.getDpsInits()[0]);
        IRMapping mapping;
        mapping.map(currentOp.getDpsInits()[0], fillOp.getResult(0));
        Operation *res = rewriter.clone(*(currentOp.getOperation()), mapping);
        rewriter.create<scf::YieldOp>(loc, res->getResult(0));
      }
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Region &region = ifOp.getElseRegion();
        rewriter.setInsertionPointToStart(&region.back());
        Operation *res = rewriter.clone(*(currentOp.getOperation()));
        rewriter.create<scf::YieldOp>(loc, res->getResult(0));
      }
      rewriter.replaceOp(currentOp, ifOp);
    }
    return success();
  }

  bool checkLinalgMatmulType(linalg::LinalgOp linalgOp) const {
    return llvm::isa<linalg::MatmulOp>(linalgOp) ||
           linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                            linalgx::PackingType::VNNI_MM2D) ||
           linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                            linalgx::PackingType::VNNI_MM4D) ||
           linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                            linalgx::PackingType::MM4D);
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!checkLinalgMatmulType(linalgOp))
      return failure();
    if (linalgOp.hasPureBufferSemantics())
      return failure();

    if (linalgOp.getOperation()->getParentOfType<scf::ForallOp>() ||
        !linalgOp || linalgOp.getNumDpsInputs() != 2)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp);
    linalg::LinalgOp originOp =
        dyn_cast<linalg::LinalgOp>(*rewriter.clone(*(linalgOp.getOperation())));
    Operation *fillOp = findParentFillOp(linalgOp.getDpsInits()[0]);

    // Step 1. Split matmul(bf16xbf16->bf16) to matmul(bf16xbf16->f32) +
    // cast(f32->bf16) if K slicing is needed
    MatmulConfigAnalysis cfgAnalysis =
        MatmulConfigAnalysis(originOp.getOperation());
    cfgAnalysis.setAllowUndivisibleInnerBlock(false);
    MatmulConfig cfg = cfgAnalysis.getConfig();
    if (!llvm::isa<linalg::GenericOp>(linalgOp))
      linalgOp = *linalg::generalizeNamedOp(rewriter, linalgOp);
    bool needLowPrecisionCast = needToLegalizeDtype(linalgOp);
    if (cfg.KThreads > 1) {
      FailureOr<DtypeLegalizeResult> result =
          matmulDtypeLegalize(rewriter, linalgOp.getOperation());
      if (succeeded(result) && result->castOp && result->linalgOp) {
        rewriter.replaceOp(linalgOp, result->castOp);
        linalgOp = dyn_cast<linalg::LinalgOp>(result->linalgOp);
      }
      needLowPrecisionCast = false;
    }

    // Step 2. Outer loop generation
    FailureOr<OuterLoopGenerationResult> outerLoopResult = outerLoopGeneration(
        rewriter, linalgOp, cfg, fillOp && isa<linalg::FillOp>(fillOp));
    if (failed(outerLoopResult))
      return failure();
    linalgOp = dyn_cast<linalg::LinalgOp>(outerLoopResult->tiledOps.back());

    // Step 3 generate inner loop body, convert the linalg.generic to brgemm
    innerBodyGenerationOption option = innerBodyGenerationOption{
        fillOp, needLowPrecisionCast, outerLoopResult->reductionLoops, cfg};

    if (failed(innerBodyGeneration(rewriter, originOp, linalgOp, option)))
      return failure();
    rewriter.eraseOp(originOp);
    return success();
  }
};

struct DeepTileContractionOp
    : public impl::DeepTileContractionOpBase<DeepTileContractionOp> {
public:
  void runOnOperation() final {
    MLIRContext &ctx = getContext();
    IRRewriter rewriter(&ctx);
    RewritePatternSet patterns(&ctx);

    patterns.add<DeepTileMatmul>(patterns.getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

    for (Dialect *dialect : ctx.getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx.getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace gc
} // namespace mlir