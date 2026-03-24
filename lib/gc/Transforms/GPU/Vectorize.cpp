//===--------- Vectorize.cpp - Vectorize structured ops ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "gc/Utils/Transform.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_VECTORIZE
#define GEN_PASS_DEF_VECTORIZE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rw) const override {
    if (!linalg::hasVectorizationImpl(op))
      return rw.notifyMatchFailure(op, "Unsupported Op, cannot vectorize");

    FailureOr<linalg::VectorizationResult> vectorResults =
        linalg::vectorize(rw, op, /*inputVectorSizes=*/{},
                          /*inputScalableVecDims=*/{});
    if (failed(vectorResults))
      return failure();

    rw.replaceOp(op, vectorResults->replacements);
    return success();
  }
};

struct Vectorize final : gc::impl::VectorizeBase<Vectorize> {

  void runOnOperation() override {
    auto fn = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<VectorizationPattern>(ctx);

    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);

    patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                 linalg::LinalgCopyVTWForwardingPattern>(ctx, /*benefit=*/2);

    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

    patterns.add<linalg::CopyVectorizationPattern>(ctx);

    vector::populateFoldArithExtensionPatterns(patterns);

    linalg::populatePadOpVectorizationPatterns(patterns);
    linalg::populateDecomposePadPatterns(patterns);

    vector::populateVectorStepLoweringPatterns(patterns);

    if (failed(applyPatternsGreedily(fn, std::move(patterns),
                                     GreedyRewriteConfig()))) {
      signalPassFailure();
      return;
    }

    // Inner loops hoisting
    auto isKernelLoop = [](LoopLikeOpInterface loop) {
      return isa<scf::ForallOp>(loop) && loop->hasAttr(GC_ATTR_KERNEL_NAME);
    };
    fn.walk([&](LoopLikeOpInterface loop) {
      if (!isKernelLoop(loop))
        moveLoopInvariantCode(loop);
    });
    OpRewriter rw(fn);
    fn.walk([&](LoopLikeOpInterface loop) {
      if (!isKernelLoop(loop))
        (void)hoistLoopInvariantSubsets(rw, loop);
    });
  }
};

} // namespace
