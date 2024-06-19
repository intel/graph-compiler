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

struct MHAToFlashAttention
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa<linalgx::ScaledDotProductAttentionOp>(linalgOp))
      return failure();
    if (linalgOp.hasPureBufferSemantics())
      return failure();
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
    // linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    // linalg::ControlDropUnitDims options;
    // options.rankReductionStrategy =
    //     linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
    // linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    // tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

    // for (auto *dialect : ctx.getLoadedDialects())
    //   dialect->getCanonicalizationPatterns(patterns);
    // for (RegisteredOperationName op : ctx.getRegisteredOperations())
    //   op.getCanonicalizationPatterns(patterns, &ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir
