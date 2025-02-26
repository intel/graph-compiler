//===-- LowerPackUnpack.cpp - Lower pack unpack into linalg ops -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <numeric>

#include "gc/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Passes.h"
namespace mlir {
namespace gc {
#define GEN_PASS_DEF_LOWERPACKUNPACK
#include "gc/Transforms/Passes.h.inc"

#define DEBUG_TYPE "lower-pack-unpack"

using namespace mlir;

// copied from tpp
// A wrapper pattern that calls linalg::lowerPack on tensor::PackOp. It lowers
// a tensor.pack op to tensor.pad + tensor.expand_shape + linalg.transpose ops.
struct LowerPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerPackResult> res = linalg::lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

// A wrapper pattern that calls linalg::lowerUnPack on tensor::UnPackOp. It
// lowers a tensor.unpack op to tensor.empty + linalg.transpose +
// tensor.collapse_shape + tensor.extract_slice ops.
struct LowerUnPackPattern : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(linalg::lowerUnPack(rewriter, op))) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to empty + transpose + reshape + extract_slice");
    }
    return success();
  }
};

class LowerPackUnpack : public impl::LowerPackUnpackBase<LowerPackUnpack> {
public:
  using impl::LowerPackUnpackBase<LowerPackUnpack>::LowerPackUnpackBase;
  void runOnOperation() final;
};

void LowerPackUnpack::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace gc
} // namespace mlir
