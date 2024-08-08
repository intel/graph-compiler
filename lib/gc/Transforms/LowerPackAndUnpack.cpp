//===-- LowerPackAndUnpack.cpp - Lower PackOp and UnpackOp ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_LOWERPACKANDUNPACK
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {
class LowerPackAndUnpack
    : public mlir::gc::impl::LowerPackAndUnpackBase<LowerPackAndUnpack> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    IRRewriter rewriter(ctx);

    getOperation()->walk([&](tensor::PackOp target) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(target);

      FailureOr<linalg::LowerPackResult> res =
          linalg::lowerPack(rewriter, target);
      if (failed(res)) {
        llvm::dbgs()
            << "cannot lower tensor.pack to pad + expand + transpose\n";
        return signalPassFailure();
      }
    });

    getOperation()->walk([&](tensor::UnPackOp target) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(target);

      FailureOr<linalg::LowerUnPackOpResult> res =
          linalg::lowerUnPack(rewriter, target);
      if (failed(res)) {
        llvm::dbgs()
            << "cannot lower tensor.unpack to transpose + collapse + extract\n";
        return signalPassFailure();
      }
    });

    RewritePatternSet patterns(ctx);
    tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
    ctx->getLoadedDialect<linalg::LinalgDialect>()->getCanonicalizationPatterns(
        patterns);
    ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace