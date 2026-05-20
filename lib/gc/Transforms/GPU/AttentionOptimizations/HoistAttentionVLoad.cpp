//===--- HoistAttentionVLoad.cpp - Hoist V-load in flash-attention --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/GPU/AttentionOptimizations/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_HOISTATTENTIONVLOAD
#define GEN_PASS_DEF_HOISTATTENTIONVLOAD
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

/// Find the xegpu.load_nd operation chain that feeds into the second dpas.
/// Collects all operations in the backward slice of the load.
static std::optional<SmallVector<Operation *>>
findVLoadChain(xegpu::DpasOp secondDpas, Region *loopBodyRegion) {
  // The second operand (rhs) of dpas is the V matrix
  Value vOperand = secondDpas.getRhs();

  // V operand might be the direct result of load_nd
  auto vLoad = dyn_cast_or_null<xegpu::LoadNdOp>(vOperand.getDefiningOp());
  if (!vLoad)
    return std::nullopt;

  // Collect the full backward slice of the load operation
  DenseSet<Operation *> visited;
  SmallVector<Operation *> chain;
  gc::attention::collectDepsInRegion(vLoad.getResult(), loopBodyRegion, chain,
                                     visited);
  return chain;
}

/// Find the K-load operation (xegpu.load_nd that feeds the first dpas)
static xegpu::LoadNdOp findKLoad(xegpu::DpasOp firstDpas) {
  // K is the rhs of the first dpas (after potential transpose)
  Value kOperand = firstDpas.getRhs();

  // K operand might go through a transpose
  if (auto transposeOp = kOperand.getDefiningOp<vector::TransposeOp>()) {
    kOperand = transposeOp.getVector();
  }

  if (auto loadOp = kOperand.getDefiningOp<xegpu::LoadNdOp>()) {
    return loadOp;
  }

  return nullptr;
}

/// Check if any operation in the chain depends on values computed after kLoad.
static bool dependsOnOpsAfterKLoad(const SmallVector<Operation *> &vLoadChain,
                                   Operation *kLoad, Block *loopBody) {
  // Collect all operations after kLoad in the loop body
  llvm::DenseSet<Operation *> opsAfterKLoad;
  bool foundKLoad = false;
  for (Operation &op : loopBody->getOperations()) {
    if (&op == kLoad) {
      foundKLoad = true;
      continue;
    }
    if (foundKLoad) {
      opsAfterKLoad.insert(&op);
    }
  }

  // Check if any op in vLoadChain uses a result from opsAfterKLoad
  for (Operation *op : vLoadChain) {
    // Skip if it's in the chain itself
    if (llvm::is_contained(vLoadChain, op))
      continue;

    for (Value operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        if (opsAfterKLoad.contains(defOp))
          return true;
      }
    }
  }

  return false;
}

/// Hoist the V-load chain right after the K-load.
static bool hoistVLoadChain(xegpu::DpasOp firstDpas,
                            const SmallVector<Operation *> &vLoadChain) {
  // Find K load operation
  auto kLoad = findKLoad(firstDpas);
  if (!kLoad) {
    // Cannot find K load, skip hoisting
    return false;
  }

  Block *loopBody = firstDpas->getBlock();

  // Check if we can safely move the chain (no dependencies on ops after kLoad)
  if (dependsOnOpsAfterKLoad(vLoadChain, kLoad, loopBody)) {
    return false;
  }

  // Move each operation in the chain right after the K load
  // We need to insert them in order, so each one goes after the previous
  Operation *insertAfter = kLoad;
  for (Operation *op : vLoadChain) {
    op->moveAfter(insertAfter);
    insertAfter = op;
  }

  return true;
}

struct HoistAttentionVLoad final
    : gc::impl::HoistAttentionVLoadBase<HoistAttentionVLoad> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool changed = false;

    moduleOp->walk([&](scf::ForOp forOp) {
      // Collect all xegpu.dpas operations in the loop
      auto dpasOps = gc::attention::collectDpasOps(forOp);

      // We're looking for flash-attention pattern with at least 2 dpas ops
      if (dpasOps.size() < 2)
        return;

      // Take the first and second dpas operations
      xegpu::DpasOp firstDpas = dpasOps[0];
      xegpu::DpasOp secondDpas = dpasOps[1];

      // Find the V-load chain before the second dpas
      auto vLoadChain = findVLoadChain(secondDpas, &forOp.getRegion());
      if (!vLoadChain.has_value() || vLoadChain->empty())
        return;

      // Hoist the V-load chain to after the K-load
      if (hoistVLoadChain(firstDpas, *vLoadChain)) {
        changed = true;
      }
    });

    if (!changed)
      return;
  }
};

} // namespace
