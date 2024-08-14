//===- MergeAlloc.cpp - General framework for merge-allocation ------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "gc/Transforms/MergeAllocTickBased.h"
#include "gc/Transforms/Passes.h"
#include "gc/Transforms/StaticMemoryPlanning.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_MERGEALLOC
#include "gc/Transforms/Passes.h.inc"

namespace {

LogicalResult passDriver(Operation *op, const gc::MergeAllocationOptions &o) {
  BufferViewFlowAnalysis aliasAnaly{op};
  auto tracesOrFail = o.tracer(op, aliasAnaly, o);
  if (failed(tracesOrFail)) {
    return failure();
  }
  if (o.checkOnly) {
    return success();
  }
  for (auto &traces : (*tracesOrFail).scopeTraces) {
    auto schedule = o.planner(op, *traces, o);
    if (failed(schedule)) {
      return failure();
    }
    if (failed(o.mutator(op, traces->getAllocScope(), *schedule, o))) {
      return failure();
    }
  }
  return success();
}

} // namespace
} // namespace gc

namespace {
using namespace mlir;
class MergeAllocPass : public gc::impl::MergeAllocBase<MergeAllocPass> {
  using parent = gc::impl::MergeAllocBase<MergeAllocPass>;
  void runOnOperation() override {
    gc::MergeAllocationOptions opt;
    if (!options) {
      opt.checkOnly = optionAnalysisOnly;
      opt.plannerOptions = plannerOptions;
      opt.alignment = optionAlignment;
      opt.tracer = gc::TickCollecter();
      opt.planner = gc::tickBasedPlanMemory;
      opt.mutator = gc::MergeAllocDefaultMutator();
    } else {
      opt = options.value();
      if (!opt.tracer)
        opt.tracer = gc::TickCollecter();
      if (!opt.planner)
        opt.planner = gc::tickBasedPlanMemory;
      if (!opt.mutator)
        opt.mutator = gc::MergeAllocDefaultMutator();
    }
    if (opt.alignment <= 0) {
      signalPassFailure();
    }
    auto op = getOperation();
    if (failed(gc::passDriver(op, opt))) {
      signalPassFailure();
    }
  }

  std::optional<gc::MergeAllocationOptions> options;

public:
  MergeAllocPass() = default;
  explicit MergeAllocPass(const gc::MergeAllocationOptions &o)
      : options{std::move(o)} {}
};
} // namespace
} // namespace mlir

std::unique_ptr<mlir::Pass>
mlir::gc::createMergeAllocPass(const gc::MergeAllocationOptions &o) {
  return std::make_unique<MergeAllocPass>(o);
}

std::unique_ptr<mlir::Pass> mlir::gc::createMergeAllocPass() {
  return std::make_unique<MergeAllocPass>();
}