//===- SetLayouts.cpp - Set XeGPU layouts on DPAS ops --------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Utils/Transform.h"
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SETLAYOUTS
#define GEN_PASS_DEF_SETLAYOUTS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

using namespace mlir::gc;

xegpu::LayoutAttr makeLayout(MLIRContext *ctx, ArrayRef<int32_t> sgLayout,
                             ArrayRef<int32_t> sgData) {
  return xegpu::LayoutAttr::get(
      ctx, /*sg_layout=*/DenseI32ArrayAttr::get(ctx, sgLayout),
      /*sg_data=*/DenseI32ArrayAttr::get(ctx, sgData),
      /*inst_data=*/nullptr,
      /*lane_layout=*/nullptr,
      /*lane_data=*/nullptr, /*order=*/nullptr);
}

std::optional<SmallVector<int32_t>> adjustLayout(Operation *op,
                                                 SmallVector<int32_t> &shape) {
  auto fn = op->getParentOfType<gpu::GPUFuncOp>();
  if (!fn) return std::nullopt;
  auto rank = shape.size();
  if (rank < 2) return std::nullopt;
  auto sgLayout = KernelAttrs(op, fn.getName().str()).getSgLayout();
  if (!sgLayout) return std::nullopt;

  // Only the innermost 2 dims are tiled, the leading ones are always 1.
  auto s = sgLayout->size();
  int32_t n = s ? static_cast<int32_t>((*sgLayout)[s - 1]) : 1;
  int32_t m = s > 1 ? static_cast<int32_t>((*sgLayout)[s - 2]) : 1;
  if (shape[rank - 2] % m || shape[rank - 1] % n) return std::nullopt;
  SmallVector<int32_t> layout(rank, 1);
  layout[rank - 2] = m;
  layout[rank - 1] = n;
  shape[rank - 2] /= m;
  shape[rank - 1] /= n;
  return layout;
}

struct DpasPattern : OpRewritePattern<xegpu::DpasOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::DpasOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLayoutA()) return failure();
    SmallVector<int32_t> dataC(
        cast<VectorType>(op.getAcc().getType()).getShape());
    auto layoutC = adjustLayout(op, dataC);
    if (!layoutC) return failure();
    SmallVector<int32_t> layoutA(*layoutC);
    SmallVector<int32_t> layoutB(*layoutC);
    SmallVector<int32_t> dataA(
        cast<VectorType>(op.getLhs().getType()).getShape());
    SmallVector<int32_t> dataB(
        cast<VectorType>(op.getRhs().getType()).getShape());
    dataA[dataA.size() - 2] /= layoutA[dataA.size() - 2];
    dataB[dataB.size() - 1] /= layoutB[dataB.size() - 1];
    auto ctx = op.getContext();
    op.setLayoutAAttr(makeLayout(ctx, layoutA, dataA));
    op.setLayoutBAttr(makeLayout(ctx, layoutB, dataB));
    op.setLayoutCdAttr(makeLayout(ctx, *layoutC, dataC));
    return success();
  }
};

template <typename OpTy> struct StorePattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op.getLayout()) return failure();
    auto shape = op.getValueType().getShape();
    if (shape.size() < 2) return failure();
    SmallVector<int32_t> data(shape.begin(), shape.end());
    auto layout = adjustLayout(op, data);
    if (!layout) return failure();
    op.setLayoutAttr(makeLayout(op.getContext(), *layout, data));
    return success();
  }
};

struct SetLayouts final : ::mlir::gc::impl::SetLayoutsBase<SetLayouts> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DpasPattern>(&getContext());
    patterns.add<StorePattern<xegpu::StoreNdOp>>(&getContext());
    patterns.add<StorePattern<xegpu::StoreScatterOp>>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
