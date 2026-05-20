//===--- SetAttentionPrefetch.cpp - Insert prefetches for attention -------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/GPU/AttentionOptimizations/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SETATTENTIONPREFETCH
#define GEN_PASS_DEF_SETATTENTIONPREFETCH
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

/// Collect the full transitive dependency chain inside the loop body for a
/// create_nd_tdesc op. Returns ops in topological order.
static SmallVector<Operation *> collectDescDeps(xegpu::CreateNdDescOp descOp,
                                                Region *loopBodyRegion) {
  SmallVector<Operation *> deps;
  DenseSet<Operation *> visited;
  gc::attention::collectDepsInRegion(descOp.getResult(), loopBodyRegion, deps,
                                     visited);
  return deps;
}

/// Clone the full dependency chain, remapping `origIV` -> `newIV`.
/// Returns the cloned create_nd_tdesc result (last op in deps).
static Value cloneDepsWithNewIV(OpBuilder &builder,
                                const SmallVectorImpl<Operation *> &deps,
                                Value origIV, Value newIV) {
  IRMapping mapping;
  mapping.map(origIV, newIV);
  for (Operation *op : deps)
    builder.clone(*op, mapping);
  return mapping.lookup(deps.back()->getResult(0));
}

/// Create a prefetch_nd op for the given tensor descriptor.
static void emitPrefetch(OpBuilder &builder, Location loc, Value tdesc) {
  auto ctx = builder.getContext();
  auto cachedHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);

  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0),
                                       builder.getIndexAttr(0)};
  xegpu::PrefetchNdOp::create(builder, loc, tdesc, offsets, cachedHint,
                              cachedHint, cachedHint, /*layout=*/nullptr);
}

struct LoadInfo {
  xegpu::LoadNdOp loadOp;
  xegpu::CreateNdDescOp descOp;
  SmallVector<Operation *> deps; // topologically ordered deps inside loop
};

static SmallVector<LoadInfo> collectLoads(scf::ForOp forOp) {
  SmallVector<LoadInfo> loads;
  Region *body = &forOp.getRegion();
  forOp.getBody()->walk([&](xegpu::LoadNdOp loadOp) {
    auto descOp = dyn_cast_or_null<xegpu::CreateNdDescOp>(
        loadOp.getTensorDesc().getDefiningOp());
    if (!descOp)
      return;
    LoadInfo info;
    info.loadOp = loadOp;
    info.descOp = descOp;
    info.deps = collectDescDeps(descOp, body);
    if (!info.deps.empty())
      loads.push_back(std::move(info));
  });
  return loads;
}

struct SetAttentionPrefetch final
    : gc::impl::SetAttentionPrefetchBase<SetAttentionPrefetch> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool changed = false;

    moduleOp->walk([&](scf::ForOp forOp) {
      auto dpasOps = gc::attention::collectDpasOps(forOp);

      if (dpasOps.size() < 2)
        return;

      auto loads = collectLoads(forOp);
      if (loads.empty())
        return;

      Value loopIV = forOp.getInductionVar();
      Value loopStep = forOp.getStep();
      Value loopLB = forOp.getLowerBound();
      Location loc = forOp.getLoc();

      // Only prefetch loads whose address depends on the loop IV.
      SmallVector<LoadInfo *> ivLoads;
      for (auto &info : loads) {
        if (gc::attention::usesValue(info.deps, loopIV))
          ivLoads.push_back(&info);
      }
      if (ivLoads.empty())
        return;

      OpBuilder builder(forOp.getContext());

      // === Pre-loop prefetches (for the first iteration) ===
      builder.setInsertionPoint(forOp);
      for (auto *info : ivLoads) {
        Value prefetchDesc =
            cloneDepsWithNewIV(builder, info->deps, loopIV, loopLB);
        emitPrefetch(builder, loc, prefetchDesc);
      }

      // === In-loop prefetches (for the next iteration) ===
      builder.setInsertionPointToStart(forOp.getBody());

      Value nextIV = arith::AddIOp::create(builder, loc, loopIV, loopStep);

      for (auto *info : ivLoads) {
        Value prefetchDesc =
            cloneDepsWithNewIV(builder, info->deps, loopIV, nextIV);
        emitPrefetch(builder, loc, prefetchDesc);
      }

      changed = true;
    });

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
