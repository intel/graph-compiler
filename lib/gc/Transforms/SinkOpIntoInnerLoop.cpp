//===-- SinkOpIntoInnerLoop.cpp - sink op to inner if possible --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_SINKOPINTOINNERLOOP
#include "gc/Transforms/Passes.h.inc"

namespace {

struct SinkOpIntoInnerLoop
    : public impl::SinkOpIntoInnerLoopBase<SinkOpIntoInnerLoop> {
public:
  void runOnOperation() final {
    auto &domInfo = getAnalysis<DominanceInfo>();
    getOperation()->walk([&](LoopLikeOpInterface loop) {
      SmallVector<Region *> regionsToSink;
      // Get the regions are that known to be executed at most once.
      for (auto &it : loop->getRegions()) {
        regionsToSink.push_back(&it);
      }
      // Sink side-effect free operations.
      controlFlowSink(
          regionsToSink, domInfo,
          [](Operation *op, Region *) { return isMemoryEffectFree(op); },
          [](Operation *op, Region *region) {
            // Move the operation to the beginning of the region's entry block.
            // This guarantees the preservation of SSA dominance of all of the
            // operation's uses are in the region.
            op->moveBefore(&region->front(), region->front().begin());
          });
    });
  }
};

} // namespace
} // namespace gc
} // namespace mlir