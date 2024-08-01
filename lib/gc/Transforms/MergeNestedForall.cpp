//===-- MergeNestedForall.cpp - Merge nested scf.forall op ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_MERGENESTEDFORALL
#include "gc/Transforms/Passes.h.inc"

namespace {

struct MergeNestedForallLoops : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = *op.getBody();
    if (!llvm::hasSingleElement(outerBody.without_terminator()))
      return failure();

    scf::ForallOp innerOp = dyn_cast<scf::ForallOp>(outerBody.front());
    if (!innerOp)
      return failure();

    for (auto val : outerBody.getArguments())
      if (llvm::is_contained(innerOp.getDynamicLowerBound(), val) ||
          llvm::is_contained(innerOp.getDynamicUpperBound(), val) ||
          llvm::is_contained(innerOp.getDynamicStep(), val))
        return failure();

    // Reductions are not supported yet.
    if (!op.getInits().empty() || !innerOp.getInits().empty())
      return failure();

    auto bodyBuilder = [&](OpBuilder &builder, Location /*loc*/,
                           ValueRange iterVals) {
      Block &innerBody = *innerOp.getBody();
      assert(iterVals.size() ==
             (outerBody.getNumArguments() + innerBody.getNumArguments()));
      IRMapping mapping;
      mapping.map(outerBody.getArguments(),
                  iterVals.take_front(outerBody.getNumArguments()));
      mapping.map(innerBody.getArguments(),
                  iterVals.take_back(innerBody.getNumArguments()));
      for (Operation &op : innerBody)
        builder.clone(op, mapping);
    };

    auto concatValues = [](const auto &first, const auto &second) {
      SmallVector<OpFoldResult> ret;
      ret.reserve(first.size() + second.size());
      ret.assign(first.begin(), first.end());
      ret.append(second.begin(), second.end());
      return ret;
    };

    auto newLowerBounds =
        concatValues(op.getMixedLowerBound(), innerOp.getMixedLowerBound());
    auto newUpperBounds =
        concatValues(op.getMixedUpperBound(), innerOp.getMixedUpperBound());
    auto newSteps = concatValues(op.getMixedStep(), innerOp.getMixedStep());
    rewriter.replaceOpWithNewOp<scf::ForallOp>(
        op, newLowerBounds, newUpperBounds, newSteps, ValueRange{},
        std::nullopt, bodyBuilder);
    return success();
  }
};

struct MergeNestedForall
    : public impl::MergeNestedForallBase<MergeNestedForall> {
public:
  void runOnOperation() final {
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);

    patterns.add<MergeNestedForallLoops>(patterns.getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace gc
} // namespace mlir
