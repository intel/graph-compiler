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
  int RowBlock, ColumnBlock;
};

static FlashAttentionConfig
getDefaultFlashAttentionConfig(linalgx::ScaledDotProductAttentionOp &sdpaOp) {
  // TODO: allow tuning
  auto shape =
      dyn_cast<RankedTensorType>(sdpaOp.getOperand(0).getType()).getShape();
  int64_t seqLen = shape[2];

  FlashAttentionConfig cfg;
  cfg.RowBlock = seqLen / 64;
  cfg.ColumnBlock = seqLen / 64;
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

  struct OuterLoopGenerationResult {
    /// Tiled operations that are generated during tiling. The order does not
    /// matter except the last op. The replacements are expected to be the
    /// results of the last op.
    SmallVector<Operation *> tiledOps;
    /// The `scf.for` operations that iterate over the tiles.
    SmallVector<LoopLikeOpInterface> loops;
  };

  static FailureOr<OuterLoopGenerationResult>
  generateOuterLoop(RewriterBase &b,
                    linalgx::ScaledDotProductAttentionOp sdpaOp,
                    const FlashAttentionConfig &cfg) {
    // TODO: handle the return value
    OuterLoopGenerationResult result;

    auto decomposableOp =
        dyn_cast<mlir::linalg::AggregatedOpInterface>(sdpaOp.getOperation());
    FailureOr<SmallVector<Value>> maybeNewResults =
        decomposableOp.decomposeOperation(b);
    b.replaceOp(decomposableOp, *maybeNewResults);
    // collect decomposed ops: matmul + mul + add + softmax + matmul
    SmallVector<Operation *> decomposedOps;
    Value curVal = (*maybeNewResults)[0];
    decomposedOps.push_back(curVal.getDefiningOp());
    if (!mlir::isa<linalgx::MultiBatchMatmulOp>(decomposedOps.back()))
      return b.notifyMatchFailure(sdpaOp,
                                  "currentOp should be a batch reduce matmul");
    curVal = mlir::dyn_cast<linalg::LinalgOp>(decomposedOps.back())
                 .getDpsInputs()[0];
    decomposedOps.push_back(curVal.getDefiningOp());
    if (!mlir::isa<linalg::SoftmaxOp>(decomposedOps.back())) {
      return b.notifyMatchFailure(sdpaOp, "currentOp should be softmax op");
    }
    curVal = mlir::dyn_cast<linalg::SoftmaxOp>(decomposedOps.back())
                 .getDpsInputs()[0];
    decomposedOps.push_back(curVal.getDefiningOp());
    if (!mlir::isa<linalg::AddOp>(decomposedOps.back()))
      return b.notifyMatchFailure(sdpaOp, "currentOp should be add op");
    curVal = mlir::dyn_cast<linalg::LinalgOp>(decomposedOps.back())
                 .getDpsInputs()[0];
    decomposedOps.push_back(curVal.getDefiningOp());
    if (!mlir::isa<linalg::GenericOp>(decomposedOps.back()))
      return b.notifyMatchFailure(sdpaOp, "currentOp should be generic mul op");
    curVal = mlir::dyn_cast<linalg::LinalgOp>(decomposedOps.back())
                 .getDpsInputs()[0];
    decomposedOps.push_back(curVal.getDefiningOp());
    if (!mlir::isa<linalgx::MultiBatchMatmulOp>(decomposedOps.back()))
      return b.notifyMatchFailure(sdpaOp,
                                  "currentOp should be batch reduce matmul");
    if (decomposedOps.size() != 5)
      return b.notifyMatchFailure(sdpaOp,
                                  "currentOp should be a decomposed sdpa.");

    // construct outer Row parallel for and Col parallel for
    scf::SCFTilingOptions rowTileOption;
    SmallVector<OpFoldResult> rowTileSizes(
        4, getAsIndexOpFoldResult(b.getContext(), 0));
    rowTileSizes[0] = getAsIndexOpFoldResult(b.getContext(), 1UL);
    rowTileSizes[1] = getAsIndexOpFoldResult(b.getContext(), 1UL);
    rowTileSizes[2] = getAsIndexOpFoldResult(b.getContext(), 32UL);
    rowTileOption.setTileSizes(rowTileSizes);
    rowTileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    for (auto &op : llvm::reverse(decomposedOps)) {
      OpBuilder::InsertionGuard guard(b);
      if (mlir::isa<linalg::LinalgOp>(op)) {
        linalg::LinalgOp linalgOp = mlir::dyn_cast<linalg::LinalgOp>(op);
        b.setInsertionPoint(linalgOp);
        if (mlir::isa<linalgx::MultiBatchMatmulOp>(op))
          linalgOp = *linalg::generalizeNamedOp(b, linalgOp);
        auto tilingInterfaceOp =
            dyn_cast<TilingInterface>(linalgOp.getOperation());
        if (!tilingInterfaceOp)
          return b.notifyMatchFailure(sdpaOp,
                                      "dyn_cast to tilingInterface failed");
        auto tilingResult =
            scf::tileUsingSCF(b, tilingInterfaceOp, rowTileOption);
        std::cout << "get tilingResult" << std::endl;
        if (failed(tilingResult))
          return failure();
        std::cout << "get tilingResult succeed" << std::endl;
        b.replaceOp(linalgOp, tilingResult->replacements);
      } else {
        linalg::SoftmaxOp softmaxOp = mlir::dyn_cast<linalg::SoftmaxOp>(op);
      }
    }
    return result;
  }

  LogicalResult matchAndRewrite(linalgx::ScaledDotProductAttentionOp sdpaOp,
                                PatternRewriter &rewriter) const override {
    FlashAttentionConfig cfg = getDefaultFlashAttentionConfig(sdpaOp);
    FailureOr<OuterLoopGenerationResult> result =
        generateOuterLoop(rewriter, sdpaOp, cfg);
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
