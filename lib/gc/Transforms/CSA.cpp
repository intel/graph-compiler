//===- CSA.cpp - Constant Subgraph Analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a constant subgraph analysis
// in MLIR.
//
//===----------------------------------------------------------------------===//
#include "gc/Analysis/DataFlow/ConstantSubgraphAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CSA
#include "gc/Transforms/Passes.h.inc"
} // namespace gc

using namespace mlir;
using namespace mlir::dataflow;

namespace gc {

struct CSA : public impl::CSABase<CSA> {
  void runOnOperation() override;
};

void CSA::runOnOperation() {
  Operation *op = getOperation();
  auto &func =
      op->getRegions().front().getBlocks().front().getOperations().front();

  // Hard-code: set the #1 argument to be constant.
  // OpBuilder builder(op->getContext());
  // func.setAttr("onednn_graph.const_args",
  //     builder.getI32ArrayAttr({1,2,3,4}));

  RunConstantSubgraphAnalysis csa;
  (void)csa.run(&func);
}

std::unique_ptr<Pass> createCSAPass() { return std::make_unique<CSA>(); }

} // namespace gc
} // namespace mlir