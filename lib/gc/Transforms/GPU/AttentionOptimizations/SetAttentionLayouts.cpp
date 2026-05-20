//===--- SetAttentionLayouts.cpp - Set XeGPU layouts for attention --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/GPU/AttentionOptimizations/Utils.h"
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

// For Q-like tensors, derive sg_layout = [shape[0] / 16, 1].
// This generalizes 128x64 -> [8, 1], 256x64 -> [16, 1], etc.
static FailureOr<SmallVector<int32_t>>
computeQLikeSgLayout(ArrayRef<int64_t> shape) {
  if (shape.size() != 2)
    return failure();
  if (shape[0] <= 0 || shape[0] % 16 != 0)
    return failure();
  return SmallVector<int32_t>{static_cast<int32_t>(shape[0] / 16), 1};
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

  auto sgLayoutA = computeQLikeSgLayout(lhsTy.getShape());
  if (failed(sgLayoutA))
    return failure();
  SmallVector<int32_t> sgLayoutB = {1, 1};
  auto sgLayoutCD = computeQLikeSgLayout(resTy.getShape());
  if (failed(sgLayoutCD))
    return failure();

  auto sgDataA = gc::attention::computeSgData(lhsTy.getShape(), *sgLayoutA);
  auto sgDataB = gc::attention::computeSgData(rhsTy.getShape(), sgLayoutB);
  auto sgDataCD = gc::attention::computeSgData(resTy.getShape(), *sgLayoutCD);

  if (failed(sgDataA) || failed(sgDataB) || failed(sgDataCD))
    return failure();

  dpas.setLayoutAAttr(gc::attention::makeLayout(ctx, *sgLayoutA, *sgDataA));
  dpas.setLayoutBAttr(gc::attention::makeLayout(ctx, sgLayoutB, *sgDataB));
  dpas.setLayoutCdAttr(gc::attention::makeLayout(ctx, *sgLayoutCD, *sgDataCD));

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

  auto sgLayout = computeQLikeSgLayout(valueTy.getShape());
  if (failed(sgLayout))
    return failure();

  auto sgData = gc::attention::computeSgData(valueTy.getShape(), *sgLayout);
  if (failed(sgData))
    return failure();

  store.setLayoutAttr(gc::attention::makeLayout(ctx, *sgLayout, *sgData));
  return success();
}

// Set layout on a LoadNdOp.
// isLhs = true  → Q-like load:  sg_layout = [shape[0]/16, 1], inst_data = [16,
// 32] isLhs = false → K/V load:     sg_layout = [1, 1]
//   K (has transpose user): order = [0, 1]
//   V (no transpose user):  inst_data = [32, 32]
static LogicalResult setLoadLayout(xegpu::LoadNdOp load, bool isLhs) {
  MLIRContext *ctx = load.getContext();

  // Already has a layout — skip.
  if (load.getLayout())
    return failure();

  VectorType valueTy = load.getType();
  if (valueTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = valueTy.getShape();
  SmallVector<int32_t> sgLayout;
  SmallVector<int32_t> instData;
  SmallVector<int32_t> order;

  if (isLhs) {
    // Q-like load: sg_layout = [shape[0]/16, 1], inst_data = [16, 32].
    if (shape[0] <= 0 || shape[0] % 16 != 0 || shape[1] % 32 != 0)
      return failure();
    sgLayout = {static_cast<int32_t>(shape[0] / 16), 1};
    instData = {16, 32};
  } else {
    // K/V load: sg_layout = [1, 1].
    sgLayout = {1, 1};

    // K path is consumed by vector.transpose and uses ordered layout.
    bool hasTransposeUser = llvm::any_of(load->getUsers(), [](Operation *user) {
      return isa<vector::TransposeOp>(user);
    });
    if (hasTransposeUser) {
      order = {0, 1};
    } else {
      if (shape[0] % 32 != 0 || shape[1] % 32 != 0)
        return failure();
      instData = {32, 32};
    }
  }

  auto sgData = gc::attention::computeSgData(shape, sgLayout);
  if (failed(sgData))
    return failure();

  load.setLayoutAttr(
      gc::attention::makeLayout(ctx, sgLayout, *sgData, instData, order));
  return success();
}

static LogicalResult setPrefetchLayout(xegpu::PrefetchNdOp prefetch) {
  MLIRContext *ctx = prefetch.getContext();

  // Already has a layout — skip.
  if (prefetch.getLayout())
    return failure();

  auto tdescTy = prefetch.getTensorDescType();
  if (tdescTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = tdescTy.getShape();

  SmallVector<int32_t> sgLayout = {2, 4};
  auto sgData = gc::attention::computeSgData(shape, sgLayout);
  if (failed(sgData))
    return failure();

  // inst_data = sg_data for prefetch operations.
  SmallVector<int32_t> instData = *sgData;
  prefetch.setLayoutAttr(
      gc::attention::makeLayout(ctx, sgLayout, *sgData, instData));
  return success();
}

struct SetAttentionLayouts final
    : gc::impl::SetAttentionLayoutsBase<SetAttentionLayouts> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool changed = false;

    moduleOp->walk([&](scf::ForOp forOp) {
      // Collect all DpasOps inside the loop body.
      auto dpasOps = gc::attention::collectDpasOps(forOp);

      // We expect at least 2 dpas ops in the attention pattern.
      if (dpasOps.size() < 2)
        return;

      // Set layouts on all dpas ops inside the loop.
      for (auto dpas : dpasOps) {
        if (succeeded(setDpasLayouts(dpas)))
          changed = true;

        // Set layouts on producer loads used by dpas operands.
        auto trySetLoadLayout = [&](Value val, bool isLhs) {
          if (auto load = val.getDefiningOp<xegpu::LoadNdOp>()) {
            if (succeeded(setLoadLayout(load, isLhs)))
              changed = true;
            return;
          }
          if (auto transpose = val.getDefiningOp<vector::TransposeOp>()) {
            if (auto load =
                    transpose.getVector().getDefiningOp<xegpu::LoadNdOp>()) {
              if (succeeded(setLoadLayout(load, /*isLhs=*/false)))
                changed = true;
            }
          }
        };

        trySetLoadLayout(dpas.getLhs(), /*isLhs=*/true);
        trySetLoadLayout(dpas.getRhs(), /*isLhs=*/false);
      }

      // Set layouts on prefetch_nd ops in the surrounding function. Prefetches
      // are optional in attention kernels.
      if (auto func = forOp->getParentOfType<FunctionOpInterface>()) {
        func.walk([&](xegpu::PrefetchNdOp prefetch) {
          if (succeeded(setPrefetchLayout(prefetch)))
            changed = true;
        });
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
