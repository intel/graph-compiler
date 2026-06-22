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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SETATTENTIONLAYOUTS
#define GEN_PASS_DEF_SETATTENTIONLAYOUTS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// For Q-like tensors, derive sg_layout = [flatBlockSize / 16, 1].
static FailureOr<SmallVector<int32_t>>
computeQLikeSgLayout(ArrayRef<int64_t> shape, int flatBlockSize) {
  if (shape.size() != 2) return failure();
  if (flatBlockSize <= 0 || flatBlockSize % 16 != 0) return failure();
  return SmallVector<int32_t>{static_cast<int32_t>(flatBlockSize / 16), 1};
}

// Set layouts on a DpasOp.
static LogicalResult setDpasLayouts(xegpu::DpasOp dpas, int flatBlockSize) {
  MLIRContext *ctx = dpas.getContext();

  // Already has layouts — skip.
  if (dpas.getLayoutA() && dpas.getLayoutB() && dpas.getLayoutCd())
    return failure();

  VectorType lhsTy = dpas.getLhsType();
  VectorType rhsTy = dpas.getRhsType();
  VectorType resTy = dpas.getResultType();

  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || resTy.getRank() != 2)
    return failure();

  auto sgLayoutA = computeQLikeSgLayout(lhsTy.getShape(), flatBlockSize);
  if (failed(sgLayoutA)) return failure();
  SmallVector<int32_t> sgLayoutB = {1, 1};
  auto sgLayoutCD = computeQLikeSgLayout(resTy.getShape(), flatBlockSize);
  if (failed(sgLayoutCD)) return failure();

  auto sgDataA = gc::attention::computeSgData(lhsTy.getShape(), *sgLayoutA);
  auto sgDataB = gc::attention::computeSgData(rhsTy.getShape(), sgLayoutB);
  auto sgDataCD = gc::attention::computeSgData(resTy.getShape(), *sgLayoutCD);

  if (failed(sgDataA) || failed(sgDataB) || failed(sgDataCD)) return failure();

  dpas.setLayoutAAttr(gc::attention::makeLayout(ctx, *sgLayoutA, *sgDataA));
  dpas.setLayoutBAttr(gc::attention::makeLayout(ctx, sgLayoutB, *sgDataB));
  dpas.setLayoutCdAttr(gc::attention::makeLayout(ctx, *sgLayoutCD, *sgDataCD));

  return success();
}

// Set layout on a StoreNdOp.
static LogicalResult setStoreLayout(xegpu::StoreNdOp store, int flatBlockSize) {
  MLIRContext *ctx = store.getContext();

  // Already has a layout — skip.
  if (store.getLayoutAttr()) return failure();

  auto valueTy = cast<VectorType>(store.getValue().getType());
  if (valueTy.getRank() != 2) return failure();

  auto sgLayout = computeQLikeSgLayout(valueTy.getShape(), flatBlockSize);
  if (failed(sgLayout)) return failure();

  auto sgData = gc::attention::computeSgData(valueTy.getShape(), *sgLayout);
  if (failed(sgData)) return failure();

  store.setLayoutAttr(gc::attention::makeLayout(ctx, *sgLayout, *sgData));
  return success();
}

// Set layout on a LoadNdOp.
// isLhs = true  → Q-like load:  sg_layout = [flatBlockSize/16, 1],
//                               inst_data = [sgData[0], 32]
// isLhs = false → K/V load:     sg_layout = [1, 1]
//   K (has transpose user): order = [0, 1]
//   V (no transpose user):  inst_data = [32, 32]
static LogicalResult setLoadLayout(xegpu::LoadNdOp load, bool isLhs,
                                   int flatBlockSize) {
  MLIRContext *ctx = load.getContext();

  // Already has a layout — skip.
  if (load.getLayout()) return failure();

  VectorType valueTy = load.getType();
  if (valueTy.getRank() != 2) return failure();

  ArrayRef<int64_t> shape = valueTy.getShape();
  SmallVector<int32_t> sgLayout;
  SmallVector<int32_t> instData;
  SmallVector<int32_t> order;

  if (isLhs) {
    if (flatBlockSize <= 0 || flatBlockSize % 16 != 0 || shape[1] % 32 != 0)
      return failure();
    sgLayout = {static_cast<int32_t>(flatBlockSize / 16), 1};
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
      if (shape[0] % 32 != 0 || shape[1] % 32 != 0) return failure();
      instData = {32, 32};
    }
  }

  auto sgData = gc::attention::computeSgData(shape, sgLayout);
  if (failed(sgData)) return failure();

  if (isLhs) instData = {(*sgData)[0], 32};

  load.setLayoutAttr(
      gc::attention::makeLayout(ctx, sgLayout, *sgData, instData, order));
  return success();
}

static LogicalResult setPrefetchLayout(xegpu::PrefetchNdOp prefetch) {
  MLIRContext *ctx = prefetch.getContext();

  // Already has a layout — skip.
  if (prefetch.getLayout()) return failure();

  auto tdescTy = prefetch.getTensorDescType();
  if (tdescTy.getRank() != 2) return failure();

  ArrayRef<int64_t> shape = tdescTy.getShape();

  SmallVector<int32_t> sgLayout = {2, 4};
  auto sgData = gc::attention::computeSgData(shape, sgLayout);
  if (failed(sgData)) return failure();

  // inst_data = sg_data for prefetch operations.
  SmallVector<int32_t> instData = *sgData;
  prefetch.setLayoutAttr(
      gc::attention::makeLayout(ctx, sgLayout, *sgData, instData));
  return success();
}

// Set layouts on all DpasOps and their producer loads inside a region.
static bool setDpasAndLoadLayouts(Operation *root, int flatBlockSize) {
  bool changed = false;
  root->walk([&](xegpu::DpasOp dpas) {
    if (succeeded(setDpasLayouts(dpas, flatBlockSize))) changed = true;

    auto trySetLoad = [&](Value val, bool isLhs) {
      if (auto load = val.getDefiningOp<xegpu::LoadNdOp>()) {
        if (succeeded(setLoadLayout(load, isLhs, flatBlockSize)))
          changed = true;
        return;
      }
      if (auto transpose = val.getDefiningOp<vector::TransposeOp>()) {
        if (auto load =
                transpose.getVector().getDefiningOp<xegpu::LoadNdOp>()) {
          if (succeeded(setLoadLayout(load, /*isLhs=*/false, flatBlockSize)))
            changed = true;
        }
      }
    };

    trySetLoad(dpas.getLhs(), /*isLhs=*/true);
    trySetLoad(dpas.getRhs(), /*isLhs=*/false);
  });
  return changed;
}

// Find store_nd ops reachable transitively through users of the given values.
static bool setStoreLayoutsForResults(ArrayRef<Value> roots,
                                      int flatBlockSize) {
  bool changed = false;
  for (auto startVal : roots) {
    SmallVector<Operation *> worklist;
    for (auto *user : startVal.getUsers()) worklist.push_back(user);

    SmallVector<Operation *> visited;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (llvm::is_contained(visited, op)) continue;
      visited.push_back(op);

      if (auto store = dyn_cast<xegpu::StoreNdOp>(op)) {
        if (succeeded(setStoreLayout(store, flatBlockSize))) changed = true;
      } else {
        for (auto res : op->getResults())
          for (auto *user : res.getUsers()) worklist.push_back(user);
      }
    }
  }
  return changed;
}

// Collect scf.if ops that consume results of forOp (peeled remainders).
static SmallVector<scf::IfOp> collectPeeledIfs(scf::ForOp forOp) {
  SmallVector<scf::IfOp> result;
  for (auto val : forOp.getResults()) {
    for (auto *user : val.getUsers()) {
      auto *parentOp = user->getParentOp();
      while (parentOp && parentOp != forOp->getParentOp()) {
        if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
          if (!llvm::is_contained(result, ifOp)) result.push_back(ifOp);
          break;
        }
        parentOp = parentOp->getParentOp();
      }
    }
  }
  return result;
}

struct SetAttentionLayouts final
    : gc::impl::SetAttentionLayoutsBase<SetAttentionLayouts> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool changed = false;

    moduleOp->walk([&](scf::ForOp forOp) {
      auto gpuFunc = forOp->getParentOfType<gpu::GPUFuncOp>();
      if (!gpuFunc) return;
      auto knownBlockSize = gpuFunc.getKnownBlockSize();
      if (!knownBlockSize.has_value()) return;
      const int flatBlockSize =
          static_cast<int>(llvm::product_of(knownBlockSize.value()));

      auto dpasOps = gc::attention::collectDpasOps(forOp);
      if (dpasOps.size() < 2) return;

      // Set layouts on dpas and loads inside the for loop.
      if (setDpasAndLoadLayouts(forOp, flatBlockSize)) changed = true;

      // Set layouts on prefetch_nd ops in the surrounding function.
      if (auto func = forOp->getParentOfType<FunctionOpInterface>()) {
        func.walk([&](xegpu::PrefetchNdOp prefetch) {
          if (succeeded(setPrefetchLayout(prefetch))) changed = true;
        });
      }

      // Handle peeled remainder iterations (scf.if using for results).
      auto peeledIfs = collectPeeledIfs(forOp);
      for (auto ifOp : peeledIfs) {
        if (setDpasAndLoadLayouts(ifOp, flatBlockSize)) changed = true;
      }

      // Find store_nd ops reachable from for/if results.
      SmallVector<Value> storeSearchRoots;
      for (auto result : forOp.getResults()) storeSearchRoots.push_back(result);
      for (auto ifOp : peeledIfs)
        for (auto result : ifOp.getResults())
          storeSearchRoots.push_back(result);

      if (setStoreLayoutsForResults(storeSearchRoots, flatBlockSize))
        changed = true;
    });

    if (!changed) markAllAnalysesPreserved();
  }
};

} // namespace
