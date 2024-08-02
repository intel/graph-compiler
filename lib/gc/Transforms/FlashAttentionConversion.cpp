//===-- FlashAttentionConversion.cpp ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Transforms/Passes.h"

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
  cfg.RowBlockSize = 64;
  cfg.ColumnBlockSize = 64;
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
    auto shape =
        dyn_cast<RankedTensorType>(sdpaOp.getOperand(0).getType()).getShape();
    auto dtype = dyn_cast<RankedTensorType>(sdpaOp.getOperand(0).getType())
                     .getElementType();
    int64_t seqLen = shape[2], headDim = shape[3];
    auto Q = sdpaOp.getOperand(0), K = sdpaOp.getOperand(1),
         V = sdpaOp.getOperand(2), mask = sdpaOp.getOperand(3);
    // construct 3 parallel outermost loops for
    // batchSize/numHeads/(seqLen/rowBlockSize)
    SmallVector<Value> destinationTensors;
    tensor::getOrCreateDestinations(rewriter, sdpaOp.getLoc(), sdpaOp,
                                    destinationTensors);
    SmallVector<OpFoldResult> lbs, ubs, tileSizes;
    for (size_t i = 0; i < 3; ++i) {
      lbs.push_back(getAsIndexOpFoldResult(rewriter.getContext(), 0));
      ubs.push_back(getAsIndexOpFoldResult(rewriter.getContext(), shape[i]));
      tileSizes.push_back(getAsIndexOpFoldResult(
          rewriter.getContext(), i == 2 ? cfg.RowBlockSize : 1));
    }
    // create forall loop
    auto forallOp = rewriter.create<scf::ForallOp>(
        loc, lbs, ubs, tileSizes, destinationTensors,
        /*mapping=*/std::nullopt,
        /*bodyBuilderFn =*/[](OpBuilder &, Location, ValueRange) {});
    rewriter.setInsertionPointToEnd(forallOp.getBody());
    SmallVector<Value> ivs = forallOp.getInductionVars();
    // inserting body for forall loop
    SmallVector<OpFoldResult> offsets;
    offsets.push_back(getAsOpFoldResult(ivs[0]));
    offsets.push_back(getAsOpFoldResult(ivs[1]));
    offsets.push_back(getAsOpFoldResult(ivs[2]));
    offsets.push_back(rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(4, rewriter.getIndexAttr(1));
    sizes[2] = rewriter.getIndexAttr(cfg.RowBlockSize);
    sizes[3] = rewriter.getIndexAttr(headDim);
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    SmallVector<int64_t> QSliceShape{1, cfg.RowBlockSize, headDim};
    SmallVector<int64_t> KVSliceShape{1, cfg.ColumnBlockSize, headDim};
    Value QSliceShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, QSliceShape, dtype);
    Value KVSliceShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, KVSliceShape, dtype);
    Value QSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, cast<RankedTensorType>(QSliceShapeOut.getType()), Q, offsets,
        sizes, strides);
    Value OSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, destinationTensors[0], offsets, sizes, strides);
    Value collapsedOSlice = rewriter.create<tensor::CollapseShapeOp>(
        loc, OSlice, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
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
    rewriter.setInsertionPointToStart(columnBlockLoop.getBody());
    // innermost computations
    Value prevOSlice = columnBlockLoop.getRegionIterArgs()[0],
          prevMaxSlice = columnBlockLoop.getRegionIterArgs()[1],
          prevSumSlice = columnBlockLoop.getRegionIterArgs()[2];
    // adjust offsets and sizes
    offsets[2] = getAsOpFoldResult(ivs[3]);
    sizes[2] = rewriter.getIndexAttr(cfg.ColumnBlockSize);
    Value KSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, cast<RankedTensorType>(KVSliceShapeOut.getType()), K, offsets,
        sizes, strides);
    Value VSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, cast<RankedTensorType>(KVSliceShapeOut.getType()), V, offsets,
        sizes, strides);
    offsets[2] = getAsOpFoldResult(ivs[2]);
    offsets[3] = getAsOpFoldResult(ivs[3]);
    sizes[2] = rewriter.getIndexAttr(cfg.RowBlockSize);
    sizes[3] = rewriter.getIndexAttr(cfg.ColumnBlockSize);
    SmallVector<int64_t> maskSliceShape{cfg.RowBlockSize, cfg.ColumnBlockSize};
    Value QKShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, maskSliceShape, dtype);
    Value maskSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, cast<RankedTensorType>(QKShapeOut.getType()), mask, offsets, sizes,
        strides);
    // transpose K
    SmallVector<int64_t> transposedShape{1, headDim, cfg.RowBlockSize};
    Value transposedShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, transposedShape, dtype);
    SmallVector<int64_t> transPerm{0, 2, 1};
    Value transposedKSlice = rewriter
                                 .create<linalg::TransposeOp>(
                                     loc, KSlice, transposedShapeOut, transPerm)
                                 ->getResult(0);
    // matmul QK
    Value matmulQKOutFilled =
        rewriter.create<linalg::FillOp>(loc, zero, QKShapeOut).getResult(0);
    Value matmulQK = rewriter
                         .create<linalg::BatchReduceMatmulOp>(
                             loc, matmulQKOutFilled.getType(),
                             ValueRange{QSlice, transposedKSlice},
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
                                           ValueRange{mul, maskSlice},
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
    SmallVector<OpFoldResult> expandedPSliceShape{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(cfg.RowBlockSize),
        rewriter.getIndexAttr(cfg.ColumnBlockSize)};
    Value expandedPSliceShapeOut =
        rewriter.create<tensor::EmptyOp>(loc, expandedPSliceShape, dtype);
    Value expandedPSlice = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedPSliceShapeOut.getType(), PSlice,
        SmallVector<ReassociationIndices>{{0, 1}, {2}}, expandedPSliceShape);
    Value matmulV = rewriter
                        .create<linalg::BatchReduceMatmulOp>(
                            loc, matmulVOutFilled.getType(),
                            ValueRange{expandedPSlice, VSlice},
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
    // yield parallel loop results
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
    // Add the scf.forall.in_parallel operations for the forall op
    rewriter.setInsertionPointToEnd(forallOp.getBody());
    auto term = rewriter.create<scf::InParallelOp>(loc);
    rewriter.setInsertionPointToStart(term.getBody());
    rewriter.create<tensor::ParallelInsertSliceOp>(
        loc, OSliceFinal, forallOp.getRegionIterArgs()[0], outputOffsets,
        outputSizes, strides);
    rewriter.replaceOp(sdpaOp, forallOp->getResults());
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
