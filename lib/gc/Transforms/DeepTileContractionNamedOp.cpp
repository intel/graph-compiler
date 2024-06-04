//===-- DeepTileContractionNamedOp.cpp - DESC -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./Tiling.hpp"
#include "gc/Dialect/Arith/Utils/EasyBuild.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>

#include "gc/Transforms/Passes.h"

#include <llvm/Support/Debug.h>

#include <memory>

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_DEEPTILECONTRACTIONNAMEDOP
#include "gc/Transforms/Passes.h.inc"

namespace {

struct SystemDesc {
  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads();
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel);
};

struct MatmulConfig {
  int MBlock, NBlock, KBlock;
  int MThreads, NThreads, KThreads;
  int innerMostMBlock, innerMostNBlock, innerMostKBlock;
};

template <typename T> inline T divAndCeil(T a, T b) { return (a - 1) / b + 1; }

enum DimType { Batch, M, N, K };

static FailureOr<SmallVector<SmallVector<DimType>>>
getOprandDimType(linalg::LinalgOp &linalgOp) {
  if (isa<linalg::MatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N}};
  } else if (llvm::isa<linalgx::Mm2DVnniOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (llvm::isa<linalgx::Mm4DVnniOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::BatchMatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::Batch, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::N}};
  }
  return failure();
}

[[maybe_unused]] static SmallVector<unsigned>
extractDimTypeIdx(ArrayRef<DimType> tyList, DimType ty) {
  SmallVector<unsigned> idxList;
  for (auto [idx, type] : llvm::enumerate(tyList)) {
    if (type == ty) {
      idxList.push_back(idx);
    }
  }
  return idxList;
}

MatmulConfig getDefaultMatmulConfig(linalg::LinalgOp &linalgOp) {
  // TODO: build a more complex heuristic to determine the best tiling
  auto M = linalgOp.getShape(linalgOp.getDpsInputOperand(0))[0];
  auto N = linalgOp.getShape(linalgOp.getDpsInputOperand(1))[1];
  auto K = linalgOp.getShape(linalgOp.getDpsInputOperand(1))[0];
  MatmulConfig cfg;

  // innermost Block
  auto defaultBlock = 32;
  cfg.innerMostMBlock = M % defaultBlock == 0 ? defaultBlock : M;
  cfg.innerMostNBlock = N % defaultBlock == 0 ? defaultBlock : N;
  cfg.innerMostKBlock = K % defaultBlock == 0 ? defaultBlock : K;

  // Number of block
  auto MNumBlock = M / cfg.innerMostMBlock;
  auto NNumBlock = N / cfg.innerMostNBlock;
  auto KNumBlock = K / cfg.innerMostKBlock;

  // Threads
  cfg.MThreads = 32;
  cfg.NThreads = 1;
  cfg.KThreads = 1;

  // Block
  cfg.MBlock = divAndCeil((int)MNumBlock, cfg.MThreads) * cfg.innerMostMBlock;
  cfg.NBlock = divAndCeil((int)NNumBlock, cfg.NThreads) * cfg.innerMostNBlock;
  cfg.KBlock = divAndCeil((int)KNumBlock, cfg.KThreads) * cfg.innerMostKBlock;
  cfg.innerMostMBlock = 32;
  cfg.innerMostNBlock = 32;
  cfg.innerMostKBlock = 32;
  cfg.MBlock = 64;
  cfg.NBlock = 64;
  cfg.KBlock = 64;
  cfg.MThreads = 2;
  cfg.NThreads = 2;
  cfg.KThreads = 2;
  return cfg;
}

static Value tensorViewRankedTensor(RewriterBase &rewriter,
                                    RankedTensorType outTensorType,
                                    Value value) {
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
    ReassociationIndices firstEntry;
    for (auto i = 0UL; i < inShape.size() - outShape.size() + 1; i++) {
      firstEntry.push_back(i);
    }
    reassocIndices.push_back(firstEntry);
    for (auto i = inShape.size() - outShape.size() + 1UL; i < inShape.size();
         i++) {
      reassocIndices.push_back({(int)i});
    }
    result = rewriter.create<tensor::CollapseShapeOp>(
        loc, outTensorType, currentValue, reassocIndices);
  } else if (outShape.size() > inShape.size()) {
    SmallVector<ReassociationIndices> reassocIndices;
    ReassociationIndices firstEntry;
    for (auto i = 0UL; i < outShape.size() - inShape.size() + 1; i++) {
      firstEntry.push_back((int)i);
    }
    reassocIndices.push_back(firstEntry);
    for (auto i = outShape.size() - inShape.size() + 1UL; i < outShape.size();
         i++) {
      reassocIndices.push_back({(int)i});
    }
    result = rewriter.create<tensor::ExpandShapeOp>(
        loc, outTensorType, currentValue, reassocIndices);
  } else {
    result = rewriter.create<tensor::CastOp>(loc, outTensorType, currentValue);
  }
  return result;
}

struct OuterLoopGenerationOption {
  enum LoopType { ForOp, ForallOp };
  SmallVector<SmallVector<int>> nestedTileSizes;
  SmallVector<LoopType> loopType;
  SmallVector<SmallVector<int>> loopDim;
  bool hasFillOp;
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

static void buildLinalgRegion(Operation *op) {
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (const Value &opOperand : op->getOperands()) {
    argTypes.push_back(getElementTypeOrSelf(opOperand.getType()));
    argLocs.push_back(opOperand.getLoc());
  }
  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  b.setInsertionPointToStart(body);
  auto *dialect = static_cast<linalg::LinalgDialect *>(op->getDialect());
  linalg::LinalgDialect::RegionBuilderFunType fun =
      dialect->getRegionBuilder("linalg.matmul");
  fun(b, *body, op->getAttrs());
}

struct DtypeLegalizeResult {
  Operation *linalgOp = nullptr;
  Operation *castOp = nullptr;
};

// Split a low precision matmul(bf16xbf16->bf16) to a combination
// matmul(bf16xbf16->f32) + cast(f32->bf16)
static FailureOr<DtypeLegalizeResult>
matmulDtypeLegalize(RewriterBase &rewriter, Operation *op,
                    bool needCopyInit = true) {

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  DtypeLegalizeResult result;
  if (!linalgOp)
    return failure();

  auto dataType =
      dyn_cast<mlir::RankedTensorType>(linalgOp.getDpsInputs()[0].getType())
          .getElementType();
  auto resultType =
      dyn_cast<mlir::RankedTensorType>(linalgOp.getDpsInits()[0].getType())
          .getElementType();

  if ((dataType.isBF16() || dataType.isF16()) && dataType == resultType) {
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
    newOperands.back() = currentOp->getResult(0);
    OperationState state(linalgOp->getLoc(), linalgOp->getName(), newOperands,
                         currentOp->getResult(0).getType(),
                         linalgOp->getAttrs());
    state.addRegion();
    currentOp = rewriter.create(state);
    buildLinalgRegion(currentOp);

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
        // TODO: refactor here to use a callback function
        if (iteratorTypes[d] == mlir::utils::IteratorType::reduction &&
            tile != 0) {
          auto result = matmulDtypeLegalize(b, currentOp.getOperation(),
                                            !option.hasFillOp);
          if (result->castOp && result->linalgOp) {
            b.replaceOp(currentOp, result->castOp);
            currentOp = dyn_cast<linalg::LinalgOp>(result->linalgOp);
          }
        }
        auto tilingResult = scf::tileUsingSCF(
            b, cast<TilingInterface>(currentOp.getOperation()), tileOption);
        if (failed(tilingResult))
          return failure();
        b.replaceOp(currentOp, tilingResult->replacements);
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
        if (iteratorTypes[d] == mlir::utils::IteratorType::reduction) {
          result.reductionLoops.push_back(tilingResult->loops.back());
        }
        result.loops.push_back(tilingResult->loops.back());
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
        if (failed(tilingResult))
          return failure();
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOp);
        if (option.hasFillOp && tilingResult->mergeOp) {
          auto fillOp = findParentFillOp(tilingResult->loops.getDpsInits()[0]);
          if (fillOp) {
            b.replaceOp(fillOp, dyn_cast<DestinationStyleOpInterface>(*fillOp)
                                    .getDpsInits()[0]);
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

static LogicalResult setStaticSizeForExtractSliceOp(RewriterBase &rewriter,
                                                    Operation *op,
                                                    bool isExtract,
                                                    SmallVector<int64_t> size,
                                                    int shrinDimNum = 0) {
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
  } else {
    return failure();
  }
  return mlir::success();
}

static LogicalResult setStaticSizeForInsertSliceOp(RewriterBase &rewriter,
                                                   Operation *op, Value source,
                                                   SmallVector<int64_t> size) {
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
  } else {
    return failure();
  }
  return success();
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
                      MatmulConfig cfg, bool hasFillOp) const {
    SmallVector<unsigned> KDimPos, MDimPos, NDimPos;
    linalgOp.getReductionDims(KDimPos);
    getMatmulParallelDims(linalgOp, 0, MDimPos);
    getMatmulParallelDims(linalgOp, 1, NDimPos);
    bool useBlockedLayout = KDimPos.size() > 1;

    OuterLoopGenerationOption option;
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    auto KFirstDim = (int)getOprandDim(linalgOp, KDimPos[0], 1);
    auto MFirstDim = (int)getOprandDim(linalgOp, MDimPos[0], 0);
    auto NFirstDim = (int)getOprandDim(linalgOp, NDimPos[0], 1);
    auto KParallelBlockSize =
        useBlockedLayout
            ? divAndCeil(KFirstDim, cfg.KThreads)
            : divAndCeil(divAndCeil(KFirstDim, cfg.KBlock), cfg.KThreads) *
                  cfg.KBlock;
    auto MParallelBlockSize =
        useBlockedLayout
            ? divAndCeil(MFirstDim, cfg.MThreads)
            : divAndCeil(divAndCeil(MFirstDim, cfg.MBlock), cfg.MThreads) *
                  cfg.MBlock;
    auto NParallelBlockSize =
        useBlockedLayout
            ? divAndCeil(NFirstDim, cfg.NThreads)
            : divAndCeil(divAndCeil(NFirstDim, cfg.NBlock), cfg.NThreads) *
                  cfg.NBlock;
    auto KOuterBlockSize = useBlockedLayout
                               ? (cfg.KBlock - 1) / cfg.innerMostKBlock + 1
                               : cfg.KBlock;
    auto MOuterBlockSize = useBlockedLayout
                               ? (cfg.MBlock - 1) / cfg.innerMostMBlock + 1
                               : cfg.MBlock;
    auto NOuterBlockSize = useBlockedLayout
                               ? (cfg.NBlock - 1) / cfg.innerMostNBlock + 1
                               : cfg.NBlock;
    // Outer
    if (cfg.KThreads > 1) {
      auto result =
          matmulDtypeLegalize(rewriter, linalgOp.getOperation(), !hasFillOp);
      if (result->castOp && result->linalgOp) {
        rewriter.replaceOp(linalgOp, result->castOp);
        linalgOp = dyn_cast<linalg::LinalgOp>(result->linalgOp);
      }
    }
    option.nestedTileSizes.emplace_back(SmallVector<int>{
        MParallelBlockSize, NParallelBlockSize, KParallelBlockSize});
    option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForallOp);
    option.loopDim.emplace_back(
        SmallVector<int>{(int)MDimPos[0], (int)NDimPos[0], (int)KDimPos[0]});
    // Middle
    for (auto [tile, dim] :
         llvm::zip(SmallVector<int>{MOuterBlockSize, NOuterBlockSize,
                                    KOuterBlockSize},
                   SmallVector<int>{(int)MDimPos[0], (int)NDimPos[0],
                                    (int)KDimPos[0]})) {
      option.nestedTileSizes.emplace_back(SmallVector<int>{tile});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<int>{dim});
    }
    // Inner
    if (!useBlockedLayout) {
      option.nestedTileSizes.emplace_back(SmallVector<int>{cfg.KBlock});
      option.loopType.emplace_back(OuterLoopGenerationOption::LoopType::ForOp);
      option.loopDim.emplace_back(SmallVector<int>{(int)KDimPos.back()});
    }
    for (auto dim = 0UL; dim < linalgOp.getNumLoops(); dim++) {
      if (dim != MDimPos.back() && dim != NDimPos.back() &&
          iteratorTypes[dim] != mlir::utils::IteratorType::reduction) {
        option.nestedTileSizes.emplace_back(SmallVector<int>{1});
        option.loopType.emplace_back(
            OuterLoopGenerationOption::LoopType::ForOp);
        option.loopDim.emplace_back(SmallVector<int>{(int)dim});
      }
    }
    option.hasFillOp = hasFillOp;
    return generateOuterLoop(rewriter, linalgOp, option);
  }

  struct innerBodyGenerationOption {
    Operation *fillOp;
    SmallVector<LoopLikeOpInterface> KLoopHandles;
  };

  LogicalResult innerBodyGeneration(RewriterBase &rewriter,
                                    linalg::LinalgOp originOp,
                                    linalg::LinalgOp currentOp,
                                    innerBodyGenerationOption &option) const {
    mlir::easybuild::EasyBuilder eb{rewriter, originOp.getLoc()};
    auto operandDimTypes = getOprandDimType(originOp);
    MatmulConfig cfg = getDefaultMatmulConfig(originOp);
    auto AShape = originOp.getShape(originOp.getDpsInputOperand(0));
    auto BShape = originOp.getShape(originOp.getDpsInputOperand(1));
    auto CShape = originOp.getShape(originOp.getDpsInitOperand(0));
    bool useBlockedLayout = BShape.size() > 2;
    // TODO: support plain in/block out format
    SmallVector<int64_t> AInnermostDims, BInnermostDims, CInnermostDims;
    if (useBlockedLayout) {
      bool firstM = true, firstK = true, firstN = true;
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
      BInnermostDims = SmallVector<int64_t>{cfg.KBlock / cfg.innerMostKBlock *
                                                cfg.innerMostKBlock,
                                            cfg.innerMostNBlock};
      CInnermostDims =
          SmallVector<int64_t>{cfg.innerMostMBlock, cfg.innerMostNBlock};
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(currentOp);
    auto dataType =
        dyn_cast<mlir::RankedTensorType>(currentOp.getDpsInputs()[0].getType());
    auto weightType =
        dyn_cast<mlir::RankedTensorType>(currentOp.getDpsInputs()[1].getType());
    auto resultType =
        dyn_cast<mlir::RankedTensorType>(currentOp.getDpsInits()[0].getType());
    // use shrink layout when it is able to be converted to brgemm
    bool useShrinkedLayout = (BInnermostDims.size() == 4);

    // update the extractSlice to static size, replace it with
    // useBlockedLayout when
    if (failed(setStaticSizeForExtractSliceOp(
            rewriter, currentOp.getDpsInits()[0].getDefiningOp(), true,
            CInnermostDims, useShrinkedLayout ? 2 : 0)) ||
        failed(setStaticSizeForExtractSliceOp(
            rewriter, currentOp.getDpsInputs()[1].getDefiningOp(), true,
            BInnermostDims, useShrinkedLayout)) ||
        failed(setStaticSizeForExtractSliceOp(
            rewriter, currentOp.getDpsInputs()[0].getDefiningOp(), true,
            AInnermostDims, useShrinkedLayout))) {
      return failure();
    }

    // View the tensor to brgemm required format
    Value dataOprand = tensorViewRankedTensor(
        rewriter,
        mlir::RankedTensorType::get(
            useBlockedLayout
                ? SmallVector<int64_t>(AInnermostDims.begin() + 1,
                                       AInnermostDims.end())
                : SmallVector<int64_t>{1, AInnermostDims[0], AInnermostDims[1]},
            dataType.getElementType()),
        currentOp.getDpsInputs()[0]);
    Value weightOprand = tensorViewRankedTensor(
        rewriter,
        mlir::RankedTensorType::get(
            useBlockedLayout
                ? SmallVector<int64_t>(BInnermostDims.begin() + 1,
                                       BInnermostDims.end())
                : SmallVector<int64_t>{1, BInnermostDims[0], BInnermostDims[1]},
            weightType.getElementType()),
        currentOp.getDpsInputs()[1]);
    Value resultOprand = tensorViewRankedTensor(
        rewriter,
        mlir::RankedTensorType::get(
            SmallVector<int64_t>(CInnermostDims.begin() +
                                     (useBlockedLayout ? 2 : 0),
                                 CInnermostDims.end()),
            resultType.getElementType()),
        currentOp.getDpsInits()[0]);

    // Create the brgemm op and replace the origin linalg op
    linalg::LinalgOp matmul;
    if (BInnermostDims.size() == 4 || BInnermostDims.size() == 2) {
      matmul = rewriter.create<linalg::BatchReduceMatmulOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
    } else {
      IRMapping mapping;
      matmul = rewriter.create<linalgx::BatchReduceMatmulVnniOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
    }
    Value result = matmul.getOperation()->getResult(0);

    // Insert the result back to the original tensor
    for (Operation *user : currentOp->getResult(0).getUsers()) {
      if (failed(setStaticSizeForInsertSliceOp(rewriter, user, result,
                                               CInnermostDims))) {
        return failure();
      }
    }
    rewriter.replaceOp(currentOp, matmul.getOperation()->getResult(0));
    currentOp = matmul;

    // Fuse the fill op to the innermost body
    if (auto fillOp = llvm::dyn_cast_or_null<linalg::FillOp>(option.fillOp)) {
      auto fillValue = fillOp.getDpsInputs()[0];
      rewriter.replaceOp(fillOp, fillOp.getDpsInits()[0]);

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
    linalgOp = *linalg::generalizeNamedOp(rewriter, linalgOp);
    Operation *fillOp = findParentFillOp(linalgOp.getDpsInits()[0]);

    // Step 1. generate the outer loop
    MatmulConfig cfg = getDefaultMatmulConfig(linalgOp);
    auto outerLoopResult = outerLoopGeneration(rewriter, linalgOp, cfg,
                                               isa<linalg::FillOp>(fillOp));
    if (failed(outerLoopResult)) {
      return failure();
    }
    linalgOp = dyn_cast<linalg::LinalgOp>(outerLoopResult->tiledOps.back());
    // Step 2 index rolling
    // if (failed(indexRolling(rewriter, linalgOp.getLoc(),
    //                         outerLoopResult->reductionLoops[0].getInductionVar(),
    //                         linalgOp.getLoopRanges()[0].size, cfg.MBlock))
    //                         ||
    //     failed(indexRolling(rewriter, linalgOp.getLoc(),
    //                         linalgOp.getDpsInputOperand(1),
    //                         linalgOp.getLoopRanges()[1].size, cfg.KBlock)))
    // {
    //     return failure();
    // }

    // Step 3 generate inner loop body, convert the linalg.generic to brgemm
    auto option =
        innerBodyGenerationOption{fillOp, outerLoopResult->reductionLoops};
    if (failed(innerBodyGeneration(rewriter, originOp, linalgOp, option))) {
      return failure();
    }
    rewriter.eraseOp(originOp);
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