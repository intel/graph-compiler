//===- TileNamed.cpp - Tile Named Linalg Ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_TILELINALGNAMED
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {
class TileLinalg : public mlir::gc::impl::TileLinalgNamedBase<TileLinalg> {

  void runOnOperation() override {
auto *ctx = &getContext();
IRRewriter rewriter(ctx);

llvm::SmallVector<Operation *> to_tile;
for (Operation &o : getOperation()->getRegion(0).front().getOperations()) {
  if (isa<linalg::MatmulOp>(o)) {
    to_tile.push_back(&o);
  }
}

for (Operation *o : to_tile) {
                                                                              llvm::errs() << "func op body to tile: " << *o << "\n";
}
  }
};

} // namespace
