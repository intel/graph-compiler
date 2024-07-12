//===-- FlashAttentionConversion.cpp ----------------------------*- C++ -*-===//
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
#define GEN_PASS_DEF_FLASHATTENTIONCONVERSION
#include "gc/Transforms/Passes.h.inc"

namespace {

struct FlashAttentionConfig {
  int RowBlockSize, ColumnBlockSize;
};

static FlashAttentionConfig
getDefaultFlashAttentionConfig(linalgx::ScaledDotProductAttentionOp &sdpaOp) {
  // TODO: allow tuning
  FlashAttentionConfig cfg;
  cfg.RowBlockSize = 32;
  cfg.ColumnBlockSize = 32;
  return cfg;
}

static LogicalResult verifyAndAppend(SmallVector<Operation *> &decomposedOps,
                                     Value curVal) {
  return success();
}

struct MHAToFlashAttention
    : public OpRewritePattern<linalgx::ScaledDotProductAttentionOp> {
  using OpRewritePattern<
      linalgx::ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgx::ScaledDotProductAttentionOp sdpaOp,
                                PatternRewriter &rewriter) const override {
    FlashAttentionConfig cfg = getDefaultFlashAttentionConfig(sdpaOp);
    Location loc = sdpaOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<LoopLikeOpInterface> loops;
    SmallVector<Value> ivs;
    auto shape =
        dyn_cast<RankedTensorType>(sdpaOp.getOperand(0).getType()).getShape();
    auto dtype = dyn_cast<RankedTensorType>(sdpaOp.getOperand(0).getType())
                     .getElementType();
    int64_t seqLen = shape[2], headDim = shape[3];
    auto Q = sdpaOp.getOperand(0), K = sdpaOp.getOperand(1),
         V = sdpaOp.getOperand(2), mask = sdpaOp.getOperand(3);
    // construct 2 parallel outermost loops for batchSize and numHeads
    SmallVector<Range> loopRanges;
    for (size_t i = 0; i < 2; ++i) {
      Range curRange;
      curRange.offset = getAsIndexOpFoldResult(rewriter.getContext(), 0UL);
      curRange.size = getAsIndexOpFoldResult(rewriter.getContext(), shape[i]);
      curRange.stride = getAsIndexOpFoldResult(rewriter.getContext(), 1UL);
      loopRanges.push_back(curRange);
    }
    SmallVector<OpFoldResult> tileSizes(
        2, getAsIndexOpFoldResult(rewriter.getContext(), 1UL));
    SmallVector<Value> destinationTensors;
    tensor::getOrCreateDestinations(rewriter, sdpaOp.getLoc(), sdpaOp,
                                    destinationTensors);
    for (auto [loopRange, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
      Value lb =
          getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
      Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
      Value step = getValueOrCreateConstantIndexOp(rewriter, loc, tileSize);
      auto loop = rewriter.create<scf::ForOp>(
          loc, lb, ub, step, destinationTensors,
          [](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
             ValueRange /*iterArgs*/) {});
      loops.push_back(loop);
      rewriter.setInsertionPointToEnd(loop.getBody());
      ivs.push_back(loop.getInductionVar());
      destinationTensors.clear();
      destinationTensors.insert(destinationTensors.begin(),
                                loop.getRegionIterArgs().begin(),
                                loop.getRegionIterArgs().end());
    }
    // create rowBlockLoop
    auto rowBlockLoop = rewriter.create<scf::ForOp>(
        loc,
        getValueOrCreateConstantIndexOp(
            rewriter, loc, getAsIndexOpFoldResult(rewriter.getContext(), 0UL)),
        getValueOrCreateConstantIndexOp(
            rewriter, loc,
            getAsIndexOpFoldResult(rewriter.getContext(), seqLen)),
        getValueOrCreateConstantIndexOp(
            rewriter, loc,
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.RowBlockSize)),
        destinationTensors,
        [](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
           ValueRange /*iterArgs*/) {});
    loops.push_back(rowBlockLoop);
    ivs.push_back(rowBlockLoop.getInductionVar());
    rewriter.setInsertionPointToEnd(rowBlockLoop.getBody());
    // inserting body for rowBlockLoop
    SmallVector<OpFoldResult> offsets;
    offsets.push_back(getAsOpFoldResult(ivs[0]));
    offsets.push_back(getAsOpFoldResult(ivs[1]));
    offsets.push_back(getAsOpFoldResult(ivs[2]));
    offsets.push_back(rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(4, rewriter.getIndexAttr(1));
    sizes[2] = rewriter.getIndexAttr(cfg.RowBlockSize);
    sizes[3] = rewriter.getIndexAttr(headDim);
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    Value QSlice = rewriter.create<tensor::ExtractSliceOp>(loc, Q, offsets,
                                                           sizes, strides);
    SmallVector<ReassociationIndices> reassocIndices{{0, 1, 2}, {3}};
    Value collapsedQSlice =
        rewriter.create<tensor::CollapseShapeOp>(loc, QSlice, reassocIndices);
    Value OSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, destinationTensors[0], offsets, sizes, strides);
    Value collapsedOSlice =
        rewriter.create<tensor::CollapseShapeOp>(loc, OSlice, reassocIndices);
    SmallVector<int64_t> blockShape(1, cfg.RowBlockSize);
    Value maxSlice = rewriter.create<tensor::EmptyOp>(loc, blockShape, dtype);
    Value sumSlice = rewriter.create<tensor::EmptyOp>(loc, blockShape, dtype);
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dtype));
    Value minusInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(
                 dtype, APFloat::getLargest(
                            cast<FloatType>(dtype).getFloatSemantics(), true)));
    Value maxSliceFilled =
        rewriter.create<linalg::FillOp>(loc, minusInf, maxSlice).getResult(0);
    Value sumSliceFilled =
        rewriter.create<linalg::FillOp>(loc, zero, sumSlice).getResult(0);
    Value collapsedOSliceFilled =
        rewriter.create<linalg::FillOp>(loc, zero, collapsedOSlice)
            .getResult(0);
    // create the innermost for loop for columnBlock
    SmallVector<Value> innermostDestinationTensors{
        collapsedOSliceFilled, maxSliceFilled, sumSliceFilled};
    auto columnBlockLoop = rewriter.create<scf::ForOp>(
        loc,
        getValueOrCreateConstantIndexOp(
            rewriter, loc, getAsIndexOpFoldResult(rewriter.getContext(), 0UL)),
        getValueOrCreateConstantIndexOp(
            rewriter, loc,
            getAsIndexOpFoldResult(rewriter.getContext(), seqLen)),
        getValueOrCreateConstantIndexOp(
            rewriter, loc,
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.ColumnBlockSize)),
        innermostDestinationTensors,
        [](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
           ValueRange /*iterArgs*/) {});
    ivs.push_back(columnBlockLoop.getInductionVar());
    rewriter.setInsertionPointToEnd(columnBlockLoop.getBody());
    // innermost computations
    Value prevOSlice = columnBlockLoop.getRegionIterArgs()[0],
          prevMaxSlice = columnBlockLoop.getRegionIterArgs()[1],
          prevSumSlice = columnBlockLoop.getRegionIterArgs()[2];
    // adjust offsets and sizes
    offsets[2] = getAsOpFoldResult(ivs[3]);
    sizes[2] = rewriter.getIndexAttr(cfg.ColumnBlockSize);
    Value KSlice = rewriter.create<tensor::ExtractSliceOp>(loc, K, offsets,
                                                           sizes, strides);
    Value VSlice = rewriter.create<tensor::ExtractSliceOp>(loc, V, offsets,
                                                           sizes, strides);
    offsets[2] = getAsOpFoldResult(ivs[2]);
    offsets[3] = getAsOpFoldResult(ivs[3]);
    sizes[2] = rewriter.getIndexAttr(cfg.RowBlockSize);
    sizes[3] = rewriter.getIndexAttr(cfg.ColumnBlockSize);
    Value maskSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, mask, offsets, sizes, strides);
    // collapse
    Value collapsedKSlice =
        rewriter.create<tensor::CollapseShapeOp>(loc, KSlice, reassocIndices);
    Value collapsedVSlice =
        rewriter.create<tensor::CollapseShapeOp>(loc, VSlice, reassocIndices);
    Value collapsedMaskSlice = rewriter.create<tensor::CollapseShapeOp>(
        loc, maskSlice, reassocIndices);
    // transpose K
    SmallVector<int64_t> transposedShape{headDim, cfg.RowBlockSize};
    Value transposedShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, transposedShape, dtype);
    SmallVector<int64_t> transPerm{1, 0};
    Value transposedK =
        rewriter
            .create<linalg::TransposeOp>(loc, collapsedKSlice,
                                         transposedShapeOut, transPerm)
            ->getResult(0);
    // matmul QK
    SmallVector<int64_t> QKShape{cfg.RowBlockSize, cfg.ColumnBlockSize};
    Value QKShapeOut = rewriter.create<tensor::EmptyOp>(loc, QKShape, dtype);
    Value matmulQKOutFilled =
        rewriter.create<linalg::FillOp>(loc, zero, QKShapeOut).getResult(0);
    Value matmulQK =
        rewriter
            .create<linalg::MatmulOp>(loc, matmulQKOutFilled.getType(),
                                      ValueRange{collapsedQSlice, transposedK},
                                      ValueRange{matmulQKOutFilled})
            .getResult(0);
    // scale & add mask
    float rsqrtHead = 1 / sqrt(headDim);
    SmallVector<AffineMap, 2> indexingMaps;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(2));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(2));
    Value mul =
        rewriter
            .create<linalg::GenericOp>(
                loc, QKShapeOut.getType(), ValueRange{matmulQK},
                ValueRange{QKShapeOut}, indexingMaps,
                SmallVector<utils::IteratorType>(2,
                                                 utils::IteratorType::parallel),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  Value constant = nestedBuilder.create<arith::ConstantOp>(
                      loc, nestedBuilder.getFloatAttr(dtype, rsqrtHead));
                  Value scaled = nestedBuilder.create<arith::MulFOp>(
                      loc, args[0], constant);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, scaled);
                })
            .getResult(0);
    Value add = rewriter
                    .create<linalg::AddOp>(loc, QKShapeOut.getType(),
                                           ValueRange{mul, collapsedMaskSlice},
                                           ValueRange{QKShapeOut})
                    .getResult(0);
    // tiling softmax
    SmallVector<int64_t> reducedShape{cfg.RowBlockSize};
    Value reducedShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, reducedShape, dtype);
    Value reduceMaxFilled =
        rewriter.create<linalg::FillOp>(loc, minusInf, reducedShapeOut)
            .getResult(0);
    Value curMaxSlice =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{add}, ValueRange{reduceMaxFilled}, 1,
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange blockArgs) {
                  Value result = nestedBuilder.create<arith::MaximumFOp>(
                      nestedLoc, blockArgs[0], blockArgs[1]);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
                })
            .getResult(0);
    Value newMaxSlice =
        rewriter
            .create<linalg::MaxOp>(loc, reducedShapeOut.getType(),
                                   ValueRange{prevMaxSlice, curMaxSlice},
                                   ValueRange{reducedShapeOut})
            .getResult(0);
    Value newMaxSliceBroadcasted =
        rewriter
            .create<linalg::BroadcastOp>(loc, newMaxSlice, QKShapeOut,
                                         SmallVector<int64_t>{1})
            .getResults()[0];
    Value sub =
        rewriter
            .create<linalg::SubOp>(loc, QKShapeOut.getType(),
                                   ValueRange{add, newMaxSliceBroadcasted},
                                   ValueRange{QKShapeOut})
            .getResult(0);
    Value PSlice =
        rewriter
            .create<linalg::ExpOp>(loc, QKShapeOut.getType(), ValueRange{sub},
                                   ValueRange{QKShapeOut})
            .getResult(0);
    Value reduceSumFilled =
        rewriter.create<linalg::FillOp>(loc, zero, reducedShapeOut)
            .getResult(0);
    Value curSumSlice =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{PSlice}, ValueRange{reduceSumFilled}, 1,
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange blockArgs) {
                  Value result = nestedBuilder.create<arith::AddFOp>(
                      nestedLoc, blockArgs[0], blockArgs[1]);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
                })
            .getResult(0);
    Value maxDiff =
        rewriter
            .create<linalg::SubOp>(loc, reducedShapeOut.getType(),
                                   ValueRange{prevMaxSlice, newMaxSlice},
                                   ValueRange{reducedShapeOut})
            .getResult(0);
    Value expMaxDiff = rewriter
                           .create<linalg::ExpOp>(
                               loc, reducedShapeOut.getType(),
                               ValueRange{maxDiff}, ValueRange{reducedShapeOut})
                           .getResult(0);
    Value rescaledPrevSumSlice =
        rewriter
            .create<linalg::MulOp>(loc, reducedShapeOut.getType(),
                                   ValueRange{prevSumSlice, expMaxDiff},
                                   ValueRange{reducedShapeOut})
            .getResult(0);
    Value newSumSlice = rewriter
                            .create<linalg::AddOp>(
                                loc, reducedShapeOut.getType(),
                                ValueRange{curSumSlice, rescaledPrevSumSlice},
                                ValueRange{reducedShapeOut})
                            .getResult(0);
    Value newSumSliceRecip =
        rewriter
            .create<linalg::ReciprocalOp>(loc, reducedShapeOut.getType(),
                                          ValueRange{newSumSlice},
                                          ValueRange{reducedShapeOut})
            .getResult(0);
    SmallVector<int64_t> VShape{cfg.RowBlockSize, headDim};
    Value VShapeOut = rewriter.create<tensor::EmptyOp>(loc, VShape, dtype);
    Value matmulVOutFilled =
        rewriter.create<linalg::FillOp>(loc, zero, VShapeOut).getResult(0);
    Value matmulV =
        rewriter
            .create<linalg::MatmulOp>(loc, matmulVOutFilled.getType(),
                                      ValueRange{PSlice, collapsedVSlice},
                                      ValueRange{matmulVOutFilled})
            .getResult(0);
    Value newSumSliceRecipBroadcasted =
        rewriter
            .create<linalg::BroadcastOp>(loc, newSumSliceRecip, VShapeOut,
                                         SmallVector<int64_t>{1})
            .getResults()[0];
    Value rescaledPrevSumSliceBroadcasted =
        rewriter
            .create<linalg::BroadcastOp>(loc, rescaledPrevSumSlice, VShapeOut,
                                         SmallVector<int64_t>{1})
            .getResults()[0];
    Value rescaledMatmulV =
        rewriter
            .create<linalg::MulOp>(
                loc, matmulVOutFilled.getType(),
                ValueRange{matmulV, newSumSliceRecipBroadcasted},
                ValueRange{matmulVOutFilled})
            .getResult(0);
    Value sumSliceQuotient =
        rewriter
            .create<linalg::MulOp>(loc, matmulVOutFilled.getType(),
                                   ValueRange{rescaledPrevSumSliceBroadcasted,
                                              newSumSliceRecipBroadcasted},
                                   ValueRange{matmulVOutFilled})
            .getResult(0);
    Value rescaledOSlice =
        rewriter
            .create<linalg::MulOp>(loc, matmulVOutFilled.getType(),
                                   ValueRange{prevOSlice, sumSliceQuotient},
                                   ValueRange{matmulVOutFilled})
            .getResult(0);
    Value newOSlice =
        rewriter
            .create<linalg::AddOp>(loc, VShapeOut.getType(),
                                   ValueRange{rescaledOSlice, rescaledMatmulV},
                                   ValueRange{VShapeOut})
            .getResult(0);
    // yield all the results of the innermost loop.
    rewriter.create<scf::YieldOp>(
        loc, ValueRange{newOSlice, newMaxSlice, newSumSlice});
    // yield rowBlockLoop results
    rewriter.setInsertionPointToEnd(rowBlockLoop.getBody());
    auto innermostLoopResults = columnBlockLoop->getResults();
    Value OSliceFinal = innermostLoopResults[0];
    SmallVector<OpFoldResult> outputOffsets;
    outputOffsets.push_back(getAsOpFoldResult(ivs[0]));
    outputOffsets.push_back(getAsOpFoldResult(ivs[1]));
    outputOffsets.push_back(getAsOpFoldResult(ivs[2]));
    outputOffsets.push_back(rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> outputSizes(4, rewriter.getIndexAttr(1));
    outputSizes[2] = rewriter.getIndexAttr(cfg.RowBlockSize);
    outputSizes[3] = rewriter.getIndexAttr(headDim);
    Value insertedRescaledOSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, OSliceFinal, rowBlockLoop.getRegionIterArgs()[0], outputOffsets,
        outputSizes, strides);
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertedRescaledOSlice});
    // Add the scf.yield operations for all the outer loops.
    for (auto [outerLoop, innerLoop] :
         llvm::zip_equal(MutableArrayRef(loops).drop_back(),
                         MutableArrayRef(loops).drop_front())) {
      rewriter.setInsertionPointToEnd(
          cast<scf::ForOp>(outerLoop.getOperation()).getBody());
      rewriter.create<scf::YieldOp>(outerLoop.getLoc(),
                                    innerLoop->getResults());
    }
    rewriter.replaceOp(sdpaOp, loops.front()->getResults());
    return success();
  }
};

struct FlashAttentionConversion
    : public impl::FlashAttentionConversionBase<FlashAttentionConversion> {
public:
  void runOnOperation() final {
    auto &ctx = getContext();
    IRRewriter rewriter(&ctx);
    RewritePatternSet patterns(&ctx);
    patterns.add<MHAToFlashAttention>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir
