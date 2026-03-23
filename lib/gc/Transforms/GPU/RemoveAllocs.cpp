//===--------- RemoveAllocs.cpp - Remove unnecessary allocs --------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Utils/Transform.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_REMOVEALLOCS
#define GEN_PASS_DEF_REMOVEALLOCS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct RemoveAllocs final : gc::impl::RemoveAllocsBase<RemoveAllocs> {

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    // If the target of the copy operation is a function argument and the only
    // use of the argument is this operation, replace the uses of the source
    // with the argument.
    fn.walk([&](memref::CopyOp copy) {
      auto target = copy.getTarget();
      if (!isa<BlockArgument>(target) || !target.hasOneUse()) {
        return WalkResult::skip();
      }

      auto src = copy.getSource().getDefiningOp();
      if (isa<memref::AllocOp>(src)) {
        src->getResult(0).replaceAllUsesWith(target);
      } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(src);
                 collapse &&
                 isa<memref::AllocOp>(collapse.getSrc().getDefiningOp())) {
        OpRewriter rw(fn);
        rw.setInsertionPointToStart(&fn.front());
        auto expand = rw.create<memref::ExpandShapeOp>(
            collapse.getSrc().getType(), target,
            collapse.getReassociationIndices());
        collapse.getSrc().replaceAllUsesWith(expand.getResult());
      }

      return WalkResult::skip();
    });

    // Remove allocs that are only used by deallocs.
    fn.walk([&](memref::AllocOp alloc) {
      if (alloc.getResult().hasOneUse()) {
        if (auto dealloc = dyn_cast<memref::DeallocOp>(
                alloc.getResult().use_begin()->getOwner())) {
          dealloc.erase();
          alloc.erase();
        }
      }
    });
  }
};

} // namespace
