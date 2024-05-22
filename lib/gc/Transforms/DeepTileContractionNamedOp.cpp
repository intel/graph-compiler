//===----------------------------------------------------------------------===//
//===- DeepTileContractionNamedOp.cpp - the Fusion for any tilable MLIR
// operation --*- C++
//-*-=//
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./Tiling.hpp"
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
  cfg.NThreads = 1;
  cfg.KThreads = 1;
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
};

struct OuterLoopGenerationResult {
  /// Tiled operations that are generated during tiling. The order does not
  /// matter except the last op. The replacements are expected to be the results
  /// of the last op.
  SmallVector<Operation *> tiledOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  /// Values to use as replacements for the untiled op. Is the same size as the
  /// number of results of the untiled op.
  SmallVector<Value> replacements;
};

static FailureOr<OuterLoopGenerationResult>
generateOuterLoop(RewriterBase &b, linalg::LinalgOp linalgOp,
                  const OuterLoopGenerationOption &option) {
  // TODO: handle the return value
  OuterLoopGenerationResult result;
  auto nestedTileSizes = option.nestedTileSizes;
  auto loopType = option.loopType;
  auto loopDim = option.loopDim;

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
  for (auto iteratorType : llvm::enumerate(loopType)) {
    auto [i, type] = iteratorType;
    auto currentDim = loopDim[i];
    auto currentTileSize = nestedTileSizes[i];
    if (type == OuterLoopGenerationOption::LoopType::ForOp) {
      scf::SCFTilingOptions tileOption;
      SmallVector<OpFoldResult> TileSizes(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));

      for (auto [d, tile] : llvm::zip(currentDim, currentTileSize)) {
        TileSizes[d] = getAsIndexOpFoldResult(b.getContext(), tile);
      }
      tileOption.setTileSizes(TileSizes);
      tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(currentOp);
      auto tilingResult = scf::tileUsingSCF(
          b, cast<TilingInterface>(currentOp.getOperation()), tileOption);
      if (failed(tilingResult))
        return failure();
      b.replaceOp(currentOp, tilingResult->replacements);
      currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
    } else if (type == OuterLoopGenerationOption::LoopType::ForallOp) {
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

      SmallVector<OpFoldResult> numThreads;
      SmallVector<Range> loopRanges =
          cast<TilingInterface>(currentOp.getOperation()).getIterationDomain(b);
      unsigned nLoops = loopRanges.size();
      numThreads.reserve(nLoops);
      AffineExpr s0, s1;
      bindSymbols(b.getContext(), s0, s1);
      AffineExpr divExpr = s0.ceilDiv(s1);
      for (const auto &it : llvm::zip(tileSizes, loopRanges)) {
        OpFoldResult numTiles = std::get<0>(it);
        if (!isConstantIntValue(numTiles, 0))
          numTiles = mlir::affine::makeComposedFoldedAffineApply(
              b, currentOp.getLoc(), divExpr,
              {std::get<1>(it).size, std::get<0>(it)});
        numThreads.push_back(numTiles);
      }

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(currentOp);
      // TODO: add split reduction support here
      if (auto partialInterface =
              dyn_cast<PartialReductionOpInterface>(currentOp.getOperation())) {
        auto tilingResult = linalgX::tileAllUsingForall(
            b, cast<PartialReductionOpInterface>(currentOp.getOperation()),
            numThreads, tileSizes, std::nullopt);
        if (failed(tilingResult))
          return failure();
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOp);
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

enum DimType { Batch, M, N, K };

static FailureOr<SmallVector<SmallVector<DimType>>>
getOprandDimType(linalg::LinalgOp &linalgOp) {
  // TODO: add more support for other linalg named matmul
  if (isa<linalg::MatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N}};
  } else if (isa<linalg::GenericOp>(linalgOp)) {
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    if (iteratorTypes.size() == 7UL) {
      // 4Dx5D, brgemm vnni
      return SmallVector<SmallVector<DimType>>{
          SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
          SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                               DimType::K},
          SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
    } else if (iteratorTypes.size() == 6UL) {
      // 4Dx4D
      return SmallVector<SmallVector<DimType>>{
          SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
          SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N},
          SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
    }
  } else {
    return failure();
  }
  return failure();
}

/*
forall([PM, PN]: [MThreads, NThreads) {
  for(PK : KThreads) {
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
}
*/
struct deepTileMatmul : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  FailureOr<OuterLoopGenerationResult>
  outerLoopGeneration(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      MatmulConfig cfg) const {
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
    return generateOuterLoop(rewriter, linalgOp, option);
  }

  struct innerBodyGenerationOption {
    bool hasFillOp = false;
    Value fillValue;
  };

  LogicalResult
  innerBodyGeneration(RewriterBase &rewriter, linalg::LinalgOp originOp,
                      linalg::LinalgOp currentOp,
                      const innerBodyGenerationOption &option) const {
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

    // Create the brgemm op
    // TODO: use brgemm_vnni to replace generic when it is applicable
    linalg::LinalgOp matmul;
    if (BInnermostDims.size() == 4 || BInnermostDims.size() == 2) {
      matmul = rewriter.create<linalg::BatchReduceMatmulOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
    } else {
      IRMapping mapping;
      matmul = dyn_cast<linalg::LinalgOp>(
          *rewriter.clone(*(currentOp.getOperation())));
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

    if (option.hasFillOp) {
      // TODO: support partial K in sinsngle threads, control flow may need
      // easy builder support
      rewriter.setInsertionPointAfter(currentOp);
      auto fillOp = rewriter.create<linalg::FillOp>(
          currentOp->getLoc(), option.fillValue, currentOp.getDpsInits()[0]);
      IRMapping mapping;
      mapping.map(currentOp.getDpsInits()[0], fillOp.getResult(0));
      auto res = rewriter.clone(*(currentOp.getOperation()), mapping);
      rewriter.replaceOp(currentOp, res);
      currentOp = dyn_cast<linalg::LinalgOp>(res);
    }
    currentOp.getOperation()->getParentOfType<func::FuncOp>().dump();
    return success();
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (matmulOp.hasPureBufferSemantics())
      return failure();
    linalg::LinalgOp linalgOp;
    linalgOp = dyn_cast<linalg::LinalgOp>(matmulOp.getOperation());
    if (linalgOp.getOperation()->getParentOfType<scf::ForallOp>())
      return failure();

    // Step 1. Match and remove the init/fill operation
    // Fuse the fill op manually before fusion support this case(fuse it into
    // if-else block)
    bool hasFillOp = false;
    Value fillValue;
    SmallVector<LoopLikeOpInterface> KLoopHandle;
    if (auto op = dyn_cast<linalg::FillOp>(
            linalgOp.getDpsInits()[0].getDefiningOp())) {
      hasFillOp = true;
      fillValue = op.getDpsInputs()[0];
      rewriter.replaceOp(op, op.getDpsInits()[0]);
    }

    // Step 2. The processes of outer Loop Generation
    // 2.0 Get the iteration infomation first
    MatmulConfig cfg = getDefaultMatmulConfig(linalgOp);
    // TODO: move the reduction dim to the front. (M, N, threads) ->
    // (threads, M, N)
    auto outerLoopResult = outerLoopGeneration(rewriter, linalgOp, cfg);
    if (failed(outerLoopResult)) {
      return failure();
    }
    linalgOp = dyn_cast<linalg::LinalgOp>(outerLoopResult->tiledOps.back());

    // Step 3 inner loop generation, convert the linalg.generic to brgemm
    if (failed(innerBodyGeneration(
            rewriter, matmulOp, linalgOp,
            innerBodyGenerationOption{hasFillOp, fillValue}))) {
      return failure();
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