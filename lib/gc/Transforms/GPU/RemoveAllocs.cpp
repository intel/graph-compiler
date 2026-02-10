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

using namespace mlir;

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
      if (!isa<BlockArgument>(target) ||
          llvm::any_of(target.getUses(), [&](OpOperand &use) {
            return use.getOwner() != copy;
          })) {
        return WalkResult::skip();
      }
      if (auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>()) {
        for (auto &use : alloc.getResult().getUses()) {
          if (auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner())) {
            dealloc.erase();
            break;
          }
        }
      }
      copy.getSource().replaceAllUsesWith(target);
      return WalkResult::skip();
    });
  }
};

} // namespace
