//===--- SetAttentionLayouts.cpp - Set XeGPU layouts for attention --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SETATTENTIONLAYOUTS
#define GEN_PASS_DEF_SETATTENTIONLAYOUTS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

static xegpu::LayoutAttr makeLayout(MLIRContext *ctx,
                                    ArrayRef<int32_t> sgLayout,
                                    ArrayRef<int32_t> sgData) {
  return xegpu::LayoutAttr::get(ctx, DenseI32ArrayAttr::get(ctx, sgLayout),
                                DenseI32ArrayAttr::get(ctx, sgData),
                                /*inst_data=*/nullptr, /*lane_layout=*/nullptr,
                                /*lane_data=*/nullptr, /*order=*/nullptr);
}

// Compute sg_data = shape / sg_layout element-wise.
// Returns failure if any dimension is not evenly divisible.
static FailureOr<SmallVector<int32_t>>
computeSgData(ArrayRef<int64_t> shape, ArrayRef<int32_t> sgLayout) {
  if (shape.size() != sgLayout.size())
    return failure();
  SmallVector<int32_t> sgData;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (sgLayout[i] == 0 || shape[i] % sgLayout[i] != 0)
      return failure();
    sgData.push_back(static_cast<int32_t>(shape[i] / sgLayout[i]));
  }
  return sgData;
}

// Set layouts on a DpasOp.
// layout_a:  sg_layout = [8, 1], sg_data = lhsShape / [8, 1]
// layout_b:  sg_layout = [1, 1], sg_data = rhsShape / [1, 1]
// layout_cd: sg_layout = [8, 1], sg_data = resultShape / [8, 1]
static LogicalResult setDpasLayouts(xegpu::DpasOp dpas) {
  MLIRContext *ctx = dpas.getContext();

  // Already has layouts — skip.
  if (dpas.getLayoutA() && dpas.getLayoutB() && dpas.getLayoutCd())
    return failure();

  VectorType lhsTy = dpas.getLhsType();
  VectorType rhsTy = dpas.getRhsType();
  VectorType resTy = dpas.getResultType();

  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || resTy.getRank() != 2)
    return failure();

  SmallVector<int32_t> sgLayoutA = {8, 1};
  SmallVector<int32_t> sgLayoutB = {1, 1};
  SmallVector<int32_t> sgLayoutCD = {8, 1};

  auto sgDataA = computeSgData(lhsTy.getShape(), sgLayoutA);
  auto sgDataB = computeSgData(rhsTy.getShape(), sgLayoutB);
  auto sgDataCD = computeSgData(resTy.getShape(), sgLayoutCD);

  if (failed(sgDataA) || failed(sgDataB) || failed(sgDataCD))
    return failure();

  dpas.setLayoutAAttr(makeLayout(ctx, sgLayoutA, *sgDataA));
  dpas.setLayoutBAttr(makeLayout(ctx, sgLayoutB, *sgDataB));
  dpas.setLayoutCdAttr(makeLayout(ctx, sgLayoutCD, *sgDataCD));

  return success();
}

// Set layout on a StoreNdOp.
// layout: sg_layout = [8, 1], sg_data = dataShape / [8, 1]
static LogicalResult setStoreLayout(xegpu::StoreNdOp store) {
  MLIRContext *ctx = store.getContext();

  // Already has a layout — skip.
  if (store.getLayoutAttr())
    return failure();

  auto valueTy = cast<VectorType>(store.getValue().getType());
  if (valueTy.getRank() != 2)
    return failure();

  SmallVector<int32_t> sgLayout = {8, 1};
  auto sgData = computeSgData(valueTy.getShape(), sgLayout);
  if (failed(sgData))
    return failure();

  store.setLayoutAttr(makeLayout(ctx, sgLayout, *sgData));
  return success();
}

struct SetAttentionLayouts final
    : gc::impl::SetAttentionLayoutsBase<SetAttentionLayouts> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool changed = false;

    moduleOp->walk([&](scf::ForOp forOp) {
      // Collect all DpasOps inside the loop body.
      SmallVector<xegpu::DpasOp> dpasOps;
      forOp.getBody()->walk(
          [&](xegpu::DpasOp dpas) { dpasOps.push_back(dpas); });

      // We expect at least 2 dpas ops in the attention pattern.
      if (dpasOps.size() < 2)
        return;

      // Set layouts on all dpas ops inside the loop.
      for (auto dpas : dpasOps) {
        if (succeeded(setDpasLayouts(dpas)))
          changed = true;
      }

      // Look for store_nd ops that consume the scf.for results
      // (i.e., appear after the loop and use its results, possibly
      // through intermediate ops).
      for (auto result : forOp.getResults()) {
        SmallVector<Operation *> worklist;
        for (auto *user : result.getUsers())
          worklist.push_back(user);

        // Walk transitively through users to find store_nd.
        SmallVector<Operation *> visited;
        while (!worklist.empty()) {
          Operation *op = worklist.pop_back_val();
          if (llvm::is_contained(visited, op))
            continue;
          visited.push_back(op);

          if (auto store = dyn_cast<xegpu::StoreNdOp>(op)) {
            if (succeeded(setStoreLayout(store)))
              changed = true;
          } else {
            for (auto res : op->getResults())
              for (auto *user : res.getUsers())
                worklist.push_back(user);
          }
        }
      }
    });

    // Signal no change if nothing was modified (for pattern convergence).
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
