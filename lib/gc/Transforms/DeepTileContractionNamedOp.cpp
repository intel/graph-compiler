//===-- DeepTileContractionNamedOp.cpp - DESC -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./Tiling.hpp"
#include "gc/Analysis/MatmulConfigAnalysis.h"
#include "gc/Dialect/Arith/Utils/EasyBuild.h"
#include "gc/Dialect/Linalgx/IR/LinalgxOps.h"
#include "gc/IR/EasyBuild.h"
#include "gc/IR/EasyBuildSCF.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Transforms/Passes.h"

#include <llvm/Support/Debug.h>

#include <memory>

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_DEEPTILECONTRACTIONNAMEDOP
#include "gc/Transforms/Passes.h.inc"

namespace {

static Value
tensorViewRankedTensor(RewriterBase &rewriter, RankedTensorType outTensorType,
                       Value value,
                       ArrayRef<int64_t> permutation = SmallVector<int64_t>{}) {
  // TODO: add support for plain layout transpose
  Value result, currentValue = value;
  auto loc = currentValue.getLoc();
  auto inTensorType = cast<RankedTensorType>(currentValue.getType());
  auto inShape = inTensorType.getShape();
  auto outShape = outTensorType.getShape();
  auto tensorElementType = inTensorType.getElementType();

  if (inShape == outShape) {
    return currentValue;
  }

  if (outTensorType.getNumDynamicDims() != inTensorType.getNumDynamicDims()) {
    SmallVector<int64_t> alignOutShape(outShape.begin(), outShape.end());
    if (outShape.size() < inShape.size()) {
      SmallVector<int64_t> oneVector(inShape.size() - outShape.size(), 1);
      alignOutShape.insert(alignOutShape.begin(), oneVector.begin(),
                           oneVector.end());
    } else {
      alignOutShape.erase(alignOutShape.begin(),
                          alignOutShape.begin() +
                              (outShape.size() - inShape.size()));
    }
    auto type = RankedTensorType::get(alignOutShape, tensorElementType);
    currentValue = rewriter.create<tensor::CastOp>(loc, type, currentValue);
    if (type == outTensorType) {
      return currentValue;
    }
  }

  if (outShape.size() < inShape.size()) {
    SmallVector<ReassociationIndices> reassocIndices;
    uint64_t outIdx = 0UL, inIdx = 0UL;
    while (inIdx < inShape.size() && outIdx < outShape.size()) {
      ReassociationIndices firstEntry;
      auto remaining = outShape[outIdx++];
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
      auto remaining = inShape[inIdx++];
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

  if (!permutation.empty()) {
    SmallVector<int64_t> transposeShape;
    for (auto idx : permutation) {
      transposeShape.push_back(outShape[idx]);
    }
    auto initOp = rewriter.create<tensor::EmptyOp>(loc, transposeShape,
                                                   tensorElementType);
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, result, initOp->getResult(0), permutation);
    result = transposeOp->getResult(0);
  }
  return result;
}

bool isDummyLoop(LoopLikeOpInterface loop) {
  std::optional<int64_t> tripCount = mlir::constantTripCount(
      *loop.getSingleLowerBound(), *loop.getSingleUpperBound(),
      *loop.getSingleStep());
  if (tripCount) {
    return *tripCount == 1;
  }
  return false;
}

static void buildLinalgRegion(Operation *op, bool createTemporaryOp = false) {
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (const Value &opOperand : op->getOperands()) {
    argTypes.push_back(getElementTypeOrSelf(opOperand.getType()));
    argLocs.push_back(opOperand.getLoc());
  }
  auto initSize = op->getResults().size();
  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  b.setInsertionPointToStart(body);
  if (createTemporaryOp) {
    auto argNum = body->getNumArguments();
    SmallVector<Value> vals;
    for (auto i = initSize; i > 0; i--) {
      vals.push_back(body->getArgument(argNum - i));
    }
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(body);
    Location loc = b.getUnknownLoc();
    b.create<linalg::YieldOp>(loc, ValueRange(vals));
  } else {
    auto *dialect = static_cast<linalg::LinalgDialect *>(op->getDialect());
    linalg::LinalgDialect::RegionBuilderFunType fun =
        dialect->getRegionBuilder("linalg.matmul");
    fun(b, *body, op->getAttrs());
  }
}

struct DtypeLegalizeResult {
  Operation *linalgOp = nullptr;
  Operation *castOp = nullptr;
};

bool needToLegalizeDtype(linalg::LinalgOp linalgOp) {
  auto dataType =
      dyn_cast<mlir::RankedTensorType>(linalgOp.getDpsInputs()[0].getType())
          .getElementType();
  auto resultType =
      dyn_cast<mlir::RankedTensorType>(linalgOp.getDpsInits()[0].getType())
          .getElementType();
  return (dataType.isBF16() || dataType.isF16()) && dataType == resultType;
}

// Split a low precision matmul(bf16xbf16->bf16) to a combination
// matmul(bf16xbf16->f32) + cast(f32->bf16)
// if needFurtherFuse=true, a middle temporary linalgOp(bf16xbf16->(f32,bf16))
// will be created
static FailureOr<DtypeLegalizeResult>
matmulDtypeLegalize(RewriterBase &rewriter, Operation *op,
                    bool needCopyInit = true, bool needFurtherFuse = false) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  DtypeLegalizeResult result;
  if (!linalgOp)
    return failure();

  if (needToLegalizeDtype(linalgOp)) {
    rewriter.setInsertionPoint(linalgOp);
    IRMapping mapping;
    auto initOp = linalgOp.getDpsInits()[0].getDefiningOp();
    auto initValue = initOp->getResult(0);
    auto initType = cast<ShapedType>(initValue.getType());
    auto tensorShape = initType.getShape();
    SmallVector<OpFoldResult> mixedShape;
    for (auto i = 0UL; i < tensorShape.size(); i++) {
      if (initType.isDynamicDim(i)) {
        Value val =
            rewriter.create<tensor::DimOp>(linalgOp.getLoc(), initValue, i);
        mixedShape.push_back(val);
      } else {
        mixedShape.push_back(
            getAsIndexOpFoldResult(rewriter.getContext(), tensorShape[i]));
      }
    }
    Operation *currentOp;

    currentOp = rewriter.create<tensor::EmptyOp>(
        linalgOp.getLoc(), mixedShape, Float32Type::get(op->getContext()));
    if (needCopyInit) {
      currentOp = rewriter.create<linalg::CopyOp>(
          linalgOp.getLoc(), initOp->getResult(0), currentOp->getResult(0));
    }
    SmallVector<Value> newOperands = linalgOp->getOperands();
    auto oldInit = newOperands.back();
    newOperands.back() = currentOp->getResult(0);

    auto indexingMaps = linalgOp.getIndexingMapsArray();
    indexingMaps.push_back(indexingMaps.back());
    SmallVector<NamedAttribute> attrs(linalgOp->getAttrs());
    SmallVector<Type> types = {currentOp->getResult(0).getType()};
    if (needFurtherFuse) {
      auto segmentSize = rewriter.getNamedAttr(
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
    OperationState state(linalgOp->getLoc(), linalgOp->getName(), newOperands,
                         types, attrs);
    state.addRegion();
    currentOp = rewriter.create(state);
    buildLinalgRegion(currentOp, needFurtherFuse);
    auto castOp = rewriter.create<linalg::CopyOp>(
        linalgOp.getLoc(), currentOp->getResult(0), initOp->getResult(0));
    result.linalgOp = currentOp;
    result.castOp = castOp;
  }

  return result;
}

static Operation *findParentFillOp(Value val) {
  SmallVector<StringRef> skipOpList = {"tensor.pack", "tensor.pad"};
  auto currentOp = val.getDefiningOp();
  while (currentOp &&
         llvm::find(skipOpList, currentOp->getName().getStringRef()) !=
             skipOpList.end() &&
         !isa<linalg::FillOp>(currentOp)) {
    currentOp = currentOp->getResult(0).getDefiningOp();
  }
  if (isa<linalg::FillOp>(currentOp)) {
    return currentOp;
  }

  return nullptr;
}

[[maybe_unused]] static LogicalResult
indexRolling(RewriterBase &b, Block *insertBlock, Location loc, Value v,
             Value rollingIdx, Value maximumRange, Value step) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(insertBlock);
  mlir::easybuild::EasyBuilder eb{b, loc};
  auto vWraped = eb.wrap<mlir::easybuild::EBUnsigned>(v);
  auto rollingIdxWraped = eb.wrap<mlir::easybuild::EBUnsigned>(rollingIdx);
  auto stepWraped = eb.wrap<mlir::easybuild::EBUnsigned>(step);
  auto maximumRangeWraped = eb.wrap<mlir::easybuild::EBUnsigned>(step);
  auto newV = (vWraped + rollingIdxWraped) * stepWraped %
              (maximumRangeWraped / stepWraped * stepWraped);
  v.replaceAllUsesWith(newV);
  return failure();
}

static void getMatmulParallelDims(linalg::LinalgOp linalgOp,
                                  unsigned operandIdx,
                                  SmallVectorImpl<unsigned> &dims) {
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(operandIdx));
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  ArrayRef<AffineExpr> results = map.getResults();
  for (auto dim : results) {
    auto dimExpr = dyn_cast<AffineDimExpr>(dim);
    if (dimExpr && iteratorTypes[dimExpr.getPosition()] ==
                       mlir::utils::IteratorType::parallel) {
      dims.push_back(dimExpr.getPosition());
    }
  }
}

static unsigned getOprandDim(linalg::LinalgOp &linalgOp, unsigned iteratorPos,
                             unsigned operandIdx) {
  Value Operand;
  unsigned dimPos;
  [[maybe_unused]] auto result =
      linalgOp.mapIterationSpaceDimToOperandDim(iteratorPos, Operand, dimPos);
  return linalgOp.getShape(linalgOp.getDpsInputOperand(operandIdx))[dimPos];
}

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
    for (auto i = 0UL; i < mixedSizes.size(); i++) {
      mixedSizes[i] = getAsIndexOpFoldResult(rewriter.getContext(), size[i]);
    }
    if (shrinDimNum > 0) {
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          extractSlice,
          mlir::RankedTensorType::get(
              SmallVector<int64_t>(size.begin() + shrinDimNum, size.end()),
              extractSlice.getResult().getType().getElementType()),
          extractSlice.getSource(), mixedOffsets, mixedSizes, mixedStrides);
    } else {
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          extractSlice, extractSlice.getSource(), mixedOffsets, mixedSizes,
          mixedStrides);
    }
  }
}

static void setStaticSizeForInsertSliceOp(RewriterBase &rewriter, Operation *op,
                                          Value source,
                                          SmallVector<int64_t> size) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(op)) {
    SmallVector<OpFoldResult> mixedOffsets = insertSlice.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSlice.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSlice.getMixedStrides();
    for (auto i = 0UL; i < mixedSizes.size(); i++) {
      mixedSizes[i] = getAsIndexOpFoldResult(rewriter.getContext(), size[i]);
    }
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        insertSlice, source, insertSlice.getDest(), mixedOffsets, mixedSizes,
        mixedStrides);
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

static FailureOr<OuterLoopGenerationResult>
generateOuterLoop(RewriterBase &b, linalg::LinalgOp linalgOp,
                  const OuterLoopGenerationOption &option) {
  // TODO: handle the return value
  OuterLoopGenerationResult result;
  auto nestedTileSizes = option.nestedTileSizes;
  auto loopType = option.loopType;
  auto loopDim = option.loopDim;
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  if (loopType.size() != loopDim.size() ||
      loopDim.size() != nestedTileSizes.size()) {
    return b.notifyMatchFailure(
        linalgOp,
        "loopType, loopDim and nestedTileSizes should have the same size");
  }

  if (linalgOp.hasPureBufferSemantics())
    return b.notifyMatchFailure(
        linalgOp, "currentOp should not has pure buffer semantics");
  linalg::LinalgOp currentOp = linalgOp;

  bool hasFullResult = !option.isPartialResult;
  for (auto loopTypeIter : llvm::enumerate(loopType)) {
    auto [i, loopType] = loopTypeIter;
    auto currentDim = loopDim[i];
    auto currentTileSize = nestedTileSizes[i];
    if (loopType == OuterLoopGenerationOption::LoopType::ForOp) {
      for (auto [d, tile] : llvm::zip(currentDim, currentTileSize)) {
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
            auto result = fn(b, currentOp.getLoc(), currentOp);
            if (succeeded(result)) {
              currentOp = *result;
            }
          }
          hasFullResult = false;
        }
        auto tilingResult = scf::tileUsingSCF(
            b, cast<TilingInterface>(currentOp.getOperation()), tileOption);
        if (failed(tilingResult))
          return failure();

        if (!isDummyLoop(tilingResult->loops.back())) {
          b.replaceOp(currentOp, tilingResult->replacements);
          currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
          if (iteratorTypes[d] == mlir::utils::IteratorType::reduction) {
            result.reductionLoops.push_back(tilingResult->loops.back());
          }
          result.loops.push_back(tilingResult->loops.back());
        }
      }
    } else if (loopType == OuterLoopGenerationOption::LoopType::ForallOp) {
      SmallVector<OpFoldResult> tileSizes(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
      SmallVector<OpFoldResult> threads(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
      SmallVector<unsigned> reductionDims;
      currentOp.getReductionDims(reductionDims);
      for (auto [d, tile] : llvm::zip(currentDim, currentTileSize)) {
        if (llvm::find(reductionDims, d) != reductionDims.end() &&
            !dyn_cast<PartialReductionOpInterface>(currentOp.getOperation()))
          tileSizes[d] = getAsIndexOpFoldResult(b.getContext(), 0);
        else
          tileSizes[d] = getAsIndexOpFoldResult(b.getContext(), tile);
      }
      SmallVector<Range> loopRanges =
          cast<TilingInterface>(currentOp.getOperation()).getIterationDomain(b);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(currentOp);
      if (auto partialInterface =
              dyn_cast<PartialReductionOpInterface>(currentOp.getOperation())) {
        for (auto [idx, tile] : llvm::enumerate(tileSizes)) {
          if (isConstantIntValue(tile, 0)) {
            tileSizes[idx] = loopRanges[idx].size;
          }
        }

        SmallVector<OpFoldResult> newParallelDims;
        for (auto i = 0UL; i < reductionDims.size(); i++) {
          newParallelDims.push_back(getAsIndexOpFoldResult(b.getContext(), i));
        }
        auto tilingResult = linalgX::tileAllUsingForall(
            b, cast<PartialReductionOpInterface>(currentOp.getOperation()), {},
            tileSizes, newParallelDims, std::nullopt);
        if (failed(tilingResult) &&
            tilingResult->parallelTiledOps.size() == 1UL)
          return failure();
        currentOp =
            dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOps.back());
        if (!tilingResult->mergeOps.empty()) {
          for (const auto &fn : option.finalReduceCallBacks) {
            auto result = fn(b, currentOp.getLoc(), *tilingResult);
            if (succeeded(result)) {
              currentOp = *result;
            }
          }
        }
      } else if (auto tilingInterface =
                     cast<TilingInterface>(currentOp.getOperation())) {
        auto tilingResult = linalg::tileToForallOpUsingTileSizes(
            b, tilingInterface, tileSizes, std::nullopt);
        if (failed(tilingResult))
          return failure();
        b.replaceOp(currentOp, tilingResult->tileOp);
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOp);
      }
    }
  }
  result.tiledOps.emplace_back(currentOp);
  return result;
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
      BSlice2 = BSlice[0, om * MBlock: (om + 1) * MBlock, ok * KBlock: (ok +
1) * KBlock]
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
struct deepTileMatmul : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  FailureOr<OuterLoopGenerationResult>
  outerLoopGeneration(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      gc::MatmulConfig cfg, bool hasFillOp) const {
    SmallVector<unsigned> KDimPos, MDimPos, NDimPos;
    linalgOp.getReductionDims(KDimPos);
    getMatmulParallelDims(linalgOp, 0, MDimPos);
    getMatmulParallelDims(linalgOp, 1, NDimPos);
    OuterLoopGenerationOption option;
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    auto KFirstDim = getOprandDim(linalgOp, KDimPos[0], 1);
    auto MFirstDim = getOprandDim(linalgOp, MDimPos[0], 0);
    auto NFirstDim = getOprandDim(linalgOp, NDimPos[0], 1);
    auto KParallelBlockSize =
        KDimPos.size() > 1
            ? llvm::divideCeil(KFirstDim, cfg.KThreads)
            : llvm::divideCeil(llvm::divideCeil(KFirstDim, cfg.KBlock),
                               cfg.KThreads) *
                  cfg.KBlock;
    auto MParallelBlockSize =
        MDimPos.size() > 1
            ? llvm::divideCeil(MFirstDim, cfg.MThreads)
            : llvm::divideCeil(llvm::divideCeil(MFirstDim, cfg.MBlock),
                               cfg.MThreads) *
                  cfg.MBlock;
    auto NParallelBlockSize =
        NDimPos.size() > 1
            ? llvm::divideCeil(NFirstDim, cfg.NThreads)
            : llvm::divideCeil(llvm::divideCeil(NFirstDim, cfg.NBlock),
                               cfg.NThreads) *
                  cfg.NBlock;
    auto KOuterBlockSize = KDimPos.size() > 1
                               ? (cfg.KBlock - 1) / cfg.innerMostKBlock + 1
                               : cfg.KBlock;
    auto MOuterBlockSize = MDimPos.size() > 1
                               ? (cfg.MBlock - 1) / cfg.innerMostMBlock + 1
                               : cfg.MBlock;
    auto NOuterBlockSize = NDimPos.size() > 1
                               ? (cfg.NBlock - 1) / cfg.innerMostNBlock + 1
                               : cfg.NBlock;
    // Outermost Numa loop
    option.nestedTileSizes.emplace_back(
        SmallVector<size_t>{uint32_t(MFirstDim / 2)});
    option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForallOp);
    option.loopDim.emplace_back(SmallVector<size_t>{MDimPos[0]});
    // Outer
    option.nestedTileSizes.emplace_back(SmallVector<size_t>{
        MParallelBlockSize, NParallelBlockSize, KParallelBlockSize});
    option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForallOp);
    option.loopDim.emplace_back(
        SmallVector<size_t>{MDimPos[0], NDimPos[0], KDimPos[0]});
    // Middle
    for (auto [tile, dim] :
         llvm::zip(SmallVector<size_t>{MOuterBlockSize, NOuterBlockSize,
                                       KOuterBlockSize},
                   SmallVector<size_t>{MDimPos[0], NDimPos[0], KDimPos[0]})) {
      option.nestedTileSizes.emplace_back(SmallVector<size_t>{tile});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{dim});
    }
    // Inner
    if (KDimPos.size() == 1) {
      option.nestedTileSizes.emplace_back(SmallVector<size_t>{cfg.KBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{KDimPos.back()});
    }
    if (MDimPos.size() == 1) {
      option.nestedTileSizes.emplace_back(
          SmallVector<size_t>{cfg.innerMostMBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{MDimPos.back()});
    }
    if (NDimPos.size() == 1) {
      option.nestedTileSizes.emplace_back(
          SmallVector<size_t>{cfg.innerMostNBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<size_t>{NDimPos.back()});
    }
    for (auto dim = 0UL; dim < linalgOp.getNumLoops(); dim++) {
      if (dim != MDimPos.back() && dim != NDimPos.back() &&
          iteratorTypes[dim] != mlir::utils::IteratorType::reduction) {
        option.nestedTileSizes.emplace_back(SmallVector<size_t>{1});
        option.loopType.emplace_back(
            OuterLoopGenerationOption::LoopType::ForOp);
        option.loopDim.emplace_back(SmallVector<size_t>{dim});
      }
    }

    auto lowPrecisionCast =
        [&](RewriterBase &rewriter, Location loc,
            linalg::LinalgOp linalgop) -> FailureOr<linalg::LinalgOp> {
      auto legalizedResult = matmulDtypeLegalize(
          rewriter, linalgop.getOperation(), !hasFillOp, true);
      if (legalizedResult->castOp && legalizedResult->linalgOp) {
        auto linalgOp = legalizedResult->linalgOp;
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
        auto initValue = result.initialValues;
        if (initValue.size() == 1 &&
            isa<linalg::FillOp>(initValue[0].getDefiningOp())) {
          rewriter.replaceOp(initValue[0].getDefiningOp(),
                             dyn_cast<DestinationStyleOpInterface>(
                                 initValue[0].getDefiningOp())
                                 .getDpsInits()[0]);
        }
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
  };

  LogicalResult innerBodyGeneration(RewriterBase &rewriter,
                                    linalg::LinalgOp originOp,
                                    linalg::LinalgOp currentOp,
                                    innerBodyGenerationOption &option) const {
    mlir::easybuild::EasyBuilder eb{rewriter, originOp.getLoc()};
    auto operandDimTypes = getOprandDimType(originOp);
    auto cfg = MatmulConfigAnalysis(originOp.getOperation()).getConfig();
    auto AShape = originOp.getShape(originOp.getDpsInputOperand(0));
    auto BShape = originOp.getShape(originOp.getDpsInputOperand(1));
    auto CShape = originOp.getShape(originOp.getDpsInitOperand(0));

    auto MDimNum = std::count_if((*operandDimTypes)[0].begin(),
                                 (*operandDimTypes)[0].end(),
                                 [](DimType d) { return d == DimType::M; });
    auto NDimNum = std::count_if((*operandDimTypes)[1].begin(),
                                 (*operandDimTypes)[1].end(),
                                 [](DimType d) { return d == DimType::N; });
    // TODO: support plain in/block out format
    SmallVector<int64_t> AInnermostDims, BInnermostDims, CInnermostDims;
    bool firstM = true, firstK = true, firstN = true;
    if (MDimNum > 1) {
      for (auto [idx, iter] : llvm::enumerate((*operandDimTypes)[0])) {
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
      for (auto [idx, iter] : llvm::enumerate((*operandDimTypes)[2])) {
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
      for (auto [idx, iter] : llvm::enumerate((*operandDimTypes)[1])) {
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

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(currentOp);
    auto dataType =
        dyn_cast<mlir::RankedTensorType>(currentOp.getDpsInputs()[0].getType())
            .getElementType();
    auto weightType =
        dyn_cast<mlir::RankedTensorType>(currentOp.getDpsInputs()[1].getType())
            .getElementType();
    auto resultType =
        dyn_cast<mlir::RankedTensorType>(currentOp.getDpsInits()[0].getType())
            .getElementType();

    // update the extractSlice to static size, replace it with
    // useBlockedLayout when
    setStaticSizeForExtractSliceOp(rewriter,
                                   currentOp.getDpsInputs()[1].getDefiningOp(),
                                   true, BInnermostDims, NDimNum > 1);
    setStaticSizeForExtractSliceOp(rewriter,
                                   currentOp.getDpsInputs()[0].getDefiningOp(),
                                   true, AInnermostDims, MDimNum > 1);
    for (auto init : currentOp.getDpsInits()) {
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
    if (dyn_cast<mlir::RankedTensorType>(weightOprand.getType())
            .getShape()
            .size() == 3) {
      matmul = rewriter.create<linalg::BatchReduceMatmulOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
    } else {
      matmul = rewriter.create<linalgx::BatchReduceMatmulVnniOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
    }
    Value result = matmul.getOperation()->getResult(0);

    // Insert the result back to the original tensor
    for (Operation *user : currentOp->getResult(0).getUsers()) {
      setStaticSizeForInsertSliceOp(rewriter, user, result, CInnermostDims);
    }

    if (option.needLowPrecisionCast) {
      rewriter.setInsertionPointAfter(currentOp);
      auto cond = eb(true);
      for (auto loop : option.KLoopHandles) {
        auto induceVar =
            eb.wrap<mlir::easybuild::EBUnsigned>(*loop.getSingleInductionVar());
        auto upBound =
            eb.wrap<mlir::easybuild::EBUnsigned>(*loop.getSingleUpperBound());
        auto step = eb.wrap<mlir::easybuild::EBUnsigned>(*loop.getSingleStep());
        auto currentCond = (induceVar + step) >= upBound;
        cond = cond & currentCond;
      }
      EB_scf_if(cond, {currentOp.getDpsInits().back().getType()}) {
        auto castOp = rewriter.create<linalg::CopyOp>(
            matmul.getLoc(), matmul->getResult(0),
            currentOp.getDpsInits().back());
        eb.yield(castOp->getResult(0));
      }
      EB_else { eb.yield(currentOp.getDpsInits().back()); }
      auto ifOp = eb.getLastOperaion();
      // set static size for the insertSliceOp of copyOp
      for (Operation *user : currentOp->getResult(1).getUsers()) {
        setStaticSizeForInsertSliceOp(rewriter, user, ifOp->getResult(0),
                                      CInnermostDims);
      }
      rewriter.replaceOp(currentOp, {matmul->getResult(0), ifOp->getResult(0)});
    } else {
      rewriter.replaceOp(currentOp, matmul->getResult(0));
    }
    currentOp = matmul;
    // Fuse the fill op to the innermost body
    if (auto fillOp = llvm::dyn_cast_or_null<linalg::FillOp>(option.fillOp)) {
      auto fillValue = fillOp.getDpsInputs()[0];
      if (cfg.KThreads <= 1) {
        // if use k slicing, the fill op is still need to be kept for the reduce
        // init
        rewriter.replaceUsesWithIf(fillOp.getResult(0), fillOp.getDpsInits()[0],
                                   [&](OpOperand &operand) {
                                     return isa<LoopLikeOpInterface>(
                                         operand.getOwner());
                                   });
      }

      rewriter.setInsertionPointAfter(currentOp);
      auto cond = eb(true);
      for (auto loop : option.KLoopHandles) {
        auto induceVar = eb.wrap<mlir::easybuild::EBUnsigned>(
            loop.getLoopRegions().front()->front().getArgument(0));
        auto currentCond = induceVar == eb.toIndex(0);
        cond = cond & currentCond;
      }
      EB_scf_if(cond, {currentOp.getDpsInits()[0].getType()}) {
        auto fillOp = rewriter.create<linalg::FillOp>(
            currentOp->getLoc(), fillValue, currentOp.getDpsInits()[0]);
        IRMapping mapping;
        mapping.map(currentOp.getDpsInits()[0], fillOp.getResult(0));
        auto res = rewriter.clone(*(currentOp.getOperation()), mapping);
        eb.yield(res->getResult(0));
      }
      EB_else {
        auto res = rewriter.clone(*(currentOp.getOperation()));
        eb.yield(res->getResult(0));
      }
      auto ifOp = eb.getLastOperaion();
      rewriter.replaceOp(currentOp, ifOp);
    }
    return success();
  }

  bool checkLinalgMatmulType(linalg::LinalgOp linalgOp) const {
    return llvm::isa<linalg::MatmulOp>(linalgOp) ||
           llvm::isa<linalgx::Mm2DVnniOp>(linalgOp) ||
           llvm::isa<linalgx::Mm4DVnniOp>(linalgOp) ||
           llvm::isa<linalgx::MultiBatchMatmulOp>(linalgOp) ||
           llvm::isa<linalg::BatchMatmulOp>(linalgOp);
    // || llvm::isa<linalg::GenericOp>(linalgOp);
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
    auto cfg = MatmulConfigAnalysis(originOp.getOperation()).getConfig();
    if (!isa<linalg::GenericOp>(linalgOp))
      linalgOp = *linalg::generalizeNamedOp(rewriter, linalgOp);
    bool needLowPrecisionCast = needToLegalizeDtype(linalgOp);
    if (cfg.KThreads > 1) {
      auto result = matmulDtypeLegalize(rewriter, linalgOp.getOperation());
      if (result->castOp && result->linalgOp) {
        rewriter.replaceOp(linalgOp, result->castOp);
        linalgOp = dyn_cast<linalg::LinalgOp>(result->linalgOp);
      }
      needLowPrecisionCast = false;
    }

    // Step 2. Outer loop generation
    auto outerLoopResult = outerLoopGeneration(
        rewriter, linalgOp, cfg, fillOp && isa<linalg::FillOp>(fillOp));
    if (failed(outerLoopResult)) {
      return failure();
    }
    linalgOp = dyn_cast<linalg::LinalgOp>(outerLoopResult->tiledOps.back());

    // Step 3 generate inner loop body, convert the linalg.generic to brgemm
    auto option = innerBodyGenerationOption{fillOp, needLowPrecisionCast,
                                            outerLoopResult->reductionLoops};

    if (failed(innerBodyGeneration(rewriter, originOp, linalgOp, option))) {
      return failure();
    }
    rewriter.eraseOp(originOp);
    return success();
  }
};

struct tileReduce : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.hasPureBufferSemantics())
      return failure();
    if (reduceOp.getOperation()->getParentOfType<LoopLikeOpInterface>())
      return failure();

    auto iteratorTypes = reduceOp.getIteratorTypesArray();

    SmallVector<Range> loopRanges =
        cast<TilingInterface>(reduceOp.getOperation())
            .getIterationDomain(rewriter);

    linalg::LinalgOp currentOp = reduceOp;
    auto cnt = 0;
    auto loopType = scf::SCFTilingOptions::LoopType::ForallOp;
    for (auto [iter, range] : llvm::zip(iteratorTypes, loopRanges)) {
      if (iter == mlir::utils::IteratorType::parallel) {
        scf::SCFTilingOptions tileOption;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(currentOp);
        SmallVector<OpFoldResult> TileSizes(
            currentOp.getNumLoops(),
            getAsIndexOpFoldResult(rewriter.getContext(), 0));
        auto rangeSize = getConstantIntValue(range.size);
        TileSizes[cnt] = getAsIndexOpFoldResult(
            rewriter.getContext(),
            cnt == loopRanges.size() - 1 && rangeSize && rangeSize >= 32 &&
                    loopType != scf::SCFTilingOptions::LoopType::ForallOp
                ? 32
                : 1);
        tileOption.setTileSizes(TileSizes);
        tileOption.setLoopType(loopType);
        auto tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(currentOp.getOperation()),
            tileOption);
        if (!isDummyLoop(tilingResult->loops.back())) {
          rewriter.replaceOp(currentOp, tilingResult->replacements);
          currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
          if (loopType == scf::SCFTilingOptions::LoopType::ForallOp)
            loopType = scf::SCFTilingOptions::LoopType::ForOp;
        }
      }
      cnt++;
    }

    cnt = 0;
    for (auto [iter, range] : llvm::zip(iteratorTypes, loopRanges)) {
      if (iter == mlir::utils::IteratorType::reduction) {
        scf::SCFTilingOptions tileOption;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(currentOp);
        SmallVector<OpFoldResult> TileSizes(
            currentOp.getNumLoops(),
            getAsIndexOpFoldResult(rewriter.getContext(), 0));
        TileSizes[cnt] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
        tileOption.setTileSizes(TileSizes);
        tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
        auto tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(currentOp.getOperation()),
            tileOption);
        if (!isDummyLoop(tilingResult->loops.back())) {
          rewriter.replaceOp(currentOp, tilingResult->replacements);
          currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
          ;
        }
      }
      cnt++;
    }
    return success();
  }
};

struct DeepTileContractionNamedOp
    : public impl::DeepTileContractionNamedOpBase<DeepTileContractionNamedOp> {
public:
  void runOnOperation() final {
    auto &ctx = getContext();
    IRRewriter rewriter(&ctx);
    RewritePatternSet patterns(&ctx);

    patterns.add<deepTileMatmul>(patterns.getContext());
    patterns.add<tileReduce>(patterns.getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    linalg::ControlDropUnitDims options;
    options.rankReductionStrategy =
        linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

    for (auto *dialect : ctx.getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx.getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir