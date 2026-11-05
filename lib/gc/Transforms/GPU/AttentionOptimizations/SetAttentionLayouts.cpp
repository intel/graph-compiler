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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SETATTENTIONLAYOUTS
#define GEN_PASS_DEF_SETATTENTIONLAYOUTS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Tile sizes used for 2D block loads (inst_data) of f16 operands.
constexpr int32_t kBlockRows = 32;
constexpr int32_t kBlockCols = 32;

// Layouts of the three dpas operands with query rows divided across subgroups.
struct DpasLayouts {
  xegpu::LayoutAttr a, b, cd;
  SmallVector<int32_t> sgDataA, sgDataB, sgDataCD;
};

static FailureOr<DpasLayouts> makeDpasLayouts(xegpu::DpasOp dpas,
                                              ArrayRef<int32_t> grid) {
  MLIRContext *ctx = dpas.getContext();
  ArrayRef<int64_t> aShape = dpas.getLhsType().getShape();
  ArrayRef<int64_t> bShape = dpas.getRhsType().getShape();
  ArrayRef<int64_t> cShape = dpas.getResultType().getShape();
  if (aShape.size() != 2 || bShape.size() != 2 || cShape.size() != 2)
    return failure();
  int64_t m = cShape[0], n = cShape[1], k = aShape[1];
  if (m % grid[0] != 0 || n % grid[1] != 0) return failure();

  DpasLayouts l;
  l.sgDataA = {static_cast<int32_t>(m / grid[0]), static_cast<int32_t>(k)};
  l.sgDataB = {static_cast<int32_t>(k), static_cast<int32_t>(n / grid[1])};
  l.sgDataCD = {static_cast<int32_t>(m / grid[0]),
                static_cast<int32_t>(n / grid[1])};
  l.a = gc::attention::makeLayout(ctx, grid, l.sgDataA);
  l.b = gc::attention::makeLayout(ctx, grid, l.sgDataB);
  l.cd = gc::attention::makeLayout(ctx, grid, l.sgDataCD);
  return l;
}

static xegpu::LayoutAttr withInstData(xegpu::LayoutAttr layout,
                                      ArrayRef<int32_t> instData) {
  MLIRContext *ctx = layout.getContext();
  return xegpu::LayoutAttr::get(ctx, layout.getSgLayout(), layout.getSgData(),
                                DenseI32ArrayAttr::get(ctx, instData),
                                layout.getLaneLayout(), layout.getLaneData(),
                                layout.getOrder());
}

// The Q K^T dpas consumes a transposed load (K); the P V dpas consumes the V
// load directly.
static bool isQKDpas(xegpu::DpasOp dpas) {
  return dpas.getRhs().getDefiningOp<vector::TransposeOp>() != nullptr;
}

// Set the layouts on one dpas and on the loads feeding it. isTail marks dpas
// ops of a peeled remainder iteration: its V load gets no inst_data. The
// remainder is straight-line code where a 32x32 load tile keeps the whole V
// block live and causes heavy register spilling, so blocking is left to pick
// the tile from the dpas operand shape.
static bool setDpasLayouts(xegpu::DpasOp dpas, ArrayRef<int32_t> grid,
                           bool isTail) {
  if (dpas.getLayoutA() && dpas.getLayoutB() && dpas.getLayoutCd())
    return false;
  auto layouts = makeDpasLayouts(dpas, grid);
  if (failed(layouts)) return false;

  dpas.setLayoutAAttr(layouts->a);
  dpas.setLayoutBAttr(layouts->b);
  dpas.setLayoutCdAttr(layouts->cd);

  // A operand: Q (loaded once outside the loop) or P (computed in the loop).
  if (auto load = dpas.getLhs().getDefiningOp<xegpu::LoadNdOp>();
      load && !load.getLayout()) {
    int32_t rows = std::min(layouts->sgDataA[0], kBlockRows);
    load.setLayoutAttr(withInstData(layouts->a, {rows, kBlockCols}));
  }

  // B operand: K through a transpose, or V directly.
  Value rhs = dpas.getRhs();
  if (auto transpose = rhs.getDefiningOp<vector::TransposeOp>()) {
    if (auto load = transpose.getVector().getDefiningOp<xegpu::LoadNdOp>();
        load && !load.getLayout()) {
      auto kLayout = cast<xegpu::LayoutAttr>(
          layouts->b.transposeDims(transpose.getPermutation()));
      load.setLayoutAttr(kLayout);
    }
  } else if (auto load = rhs.getDefiningOp<xegpu::LoadNdOp>();
             load && !load.getLayout()) {
    xegpu::LayoutAttr vLayout = layouts->b;
    if (!isTail && layouts->sgDataB[0] % kBlockRows == 0 &&
        layouts->sgDataB[1] % kBlockCols == 0)
      vLayout = withInstData(vLayout, {kBlockRows, kBlockCols});
    load.setLayoutAttr(vLayout);
  }
  return true;
}

// Set the output layout on a store.
static bool setStoreLayout(xegpu::StoreNdOp store, ArrayRef<int32_t> grid) {
  if (store.getLayoutAttr()) return false;
  auto valueTy = cast<VectorType>(store.getValue().getType());
  if (valueTy.getRank() != 2) return false;
  auto sgData = gc::attention::computeSgData(valueTy.getShape(), grid);
  if (failed(sgData)) return false;
  store.setLayoutAttr(
      gc::attention::makeLayout(store.getContext(), grid, *sgData));
  return true;
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

// Re-loading Q per iteration lets the backend stream it from L1 instead of
// keeping it live alongside the accumulator across the whole loop.
static bool sinkQLoadIntoLoop(scf::ForOp forOp) {
  bool changed = false;
  forOp.getBody()->walk([&](xegpu::DpasOp dpas) {
    if (!isQKDpas(dpas)) return;
    auto load = dpas.getLhs().getDefiningOp<xegpu::LoadNdOp>();
    if (!load || forOp.getBody()->findAncestorOpInBlock(*load)) return;
    auto desc = load.getTensorDesc().getDefiningOp<xegpu::CreateNdDescOp>();
    if (!desc) return;
    OpBuilder b(dpas);
    IRMapping map;
    Operation *newDesc = b.clone(*desc, map);
    map.map(desc.getResult(), newDesc->getResult(0));
    Operation *newLoad = b.clone(*load, map);
    dpas.getLhsMutable().assign(newLoad->getResult(0));
    changed = true;
  });
  return changed;
}

// Set layouts on all dpas ops (and their loads) inside a region. isTail marks
// a peeled remainder iteration (see setDpasLayouts).
static bool setDpasAndLoadLayouts(Operation *root, ArrayRef<int32_t> grid,
                                  bool isTail) {
  bool changed = false;
  root->walk([&](xegpu::DpasOp dpas) {
    if (setDpasLayouts(dpas, grid, isTail)) changed = true;
  });
  return changed;
}

// Find store_nd ops reachable transitively through users of the given values.
static bool setStoreLayoutsForResults(ArrayRef<Value> roots,
                                      ArrayRef<int32_t> grid) {
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
        if (setStoreLayout(store, grid)) changed = true;
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

// Collect dpas ops of a peeled remainder iteration that was inlined into the
// block of forOp. With static shapes the canonicalizer folds the guarding
// scf.if away, so the remainder appears as plain ops after the loop. Walk
// forward from the loop results and from the loop-invariant Q operand `q`,
// staying within the same block and after forOp.
static SmallVector<xegpu::DpasOp> collectInlinedPeeledDpas(scf::ForOp forOp,
                                                           Value q) {
  Block *block = forOp->getBlock();
  SmallVector<Value> roots(forOp.getResults());
  if (q && !forOp.getRegion().isAncestor(q.getParentRegion()))
    roots.push_back(q);

  SmallVector<xegpu::DpasOp> result;
  DenseSet<Operation *> visited;
  SmallVector<Operation *> worklist;
  auto pushUsers = [&](Value val) {
    for (auto *user : val.getUsers())
      if (user->getBlock() == block && forOp->isBeforeInBlock(user))
        worklist.push_back(user);
  };
  for (auto root : roots) pushUsers(root);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second) continue;
    if (auto dpas = dyn_cast<xegpu::DpasOp>(op)) result.push_back(dpas);
    for (auto res : op->getResults()) pushUsers(res);
  }
  return result;
}

struct SetAttentionLayouts final
    : gc::impl::SetAttentionLayoutsBase<SetAttentionLayouts> {
  using SetAttentionLayoutsBase::SetAttentionLayoutsBase;

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
      if (flatBlockSize <= 0 || flatBlockSize % 16 != 0) return;
      const int32_t numSg = flatBlockSize / 16;

      auto dpasOps = gc::attention::collectDpasOps(forOp);
      if (dpasOps.size() < 2) return;

      SmallVector<int32_t> grid = {numSg, 1};

      // The loop-invariant Q operand, taken before sinkQLoadIntoLoop rewires
      // the Q K^T dpas to an in-loop copy: a peeled remainder inlined after
      // the loop still consumes the original.
      Value q;
      for (auto dpas : dpasOps)
        if (isQKDpas(dpas)) {
          q = dpas.getLhs();
          break;
        }

      // Set layouts on dpas and loads inside the for loop.
      if (setDpasAndLoadLayouts(forOp, grid, /*isTail=*/false)) changed = true;

      if (qInLoop && sinkQLoadIntoLoop(forOp)) changed = true;

      // Set layouts on prefetch_nd ops in the surrounding function.
      if (auto func = forOp->getParentOfType<FunctionOpInterface>()) {
        func.walk([&](xegpu::PrefetchNdOp prefetch) {
          if (succeeded(setPrefetchLayout(prefetch))) changed = true;
        });
      }

      // Handle peeled remainder iterations (scf.if using for results).
      auto peeledIfs = collectPeeledIfs(forOp);
      for (auto ifOp : peeledIfs) {
        if (setDpasAndLoadLayouts(ifOp, grid, /*isTail=*/true)) changed = true;
      }

      // Handle peeled remainder iterations inlined after the loop.
      for (auto dpas : collectInlinedPeeledDpas(forOp, q)) {
        if (setDpasLayouts(dpas, grid, /*isTail=*/true)) changed = true;
      }

      // Find store_nd ops reachable from for/if results.
      SmallVector<Value> storeSearchRoots;
      for (auto result : forOp.getResults()) storeSearchRoots.push_back(result);
      for (auto ifOp : peeledIfs)
        for (auto result : ifOp.getResults())
          storeSearchRoots.push_back(result);

      if (setStoreLayoutsForResults(storeSearchRoots, grid)) changed = true;
    });

    if (!changed) markAllAnalysesPreserved();
  }
};

} // namespace
