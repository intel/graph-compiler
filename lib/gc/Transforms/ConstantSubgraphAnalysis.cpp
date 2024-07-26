//===-- ConstantSubgraphAnalysis.cpp - Constant Subgraph --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a constant subgraph analysis
// in MLIR.
//
//===----------------------------------------------------------------------===//
#include "gc/Analysis/DataFlow/ConstantSubgraphAnalyser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONSTANTSUBGRAPHANALYSIS
#include "gc/Transforms/Passes.h.inc"
} // namespace gc

using namespace mlir;
using namespace mlir::dataflow;

namespace gc {

struct ConstantSubgraphAnalysis
    : public impl::ConstantSubgraphAnalysisBase<ConstantSubgraphAnalysis> {
  void runOnOperation() override;
};

void ConstantSubgraphAnalysis::runOnOperation() {
  Operation *op = getOperation();
  auto &func =
      op->getRegions().front().getBlocks().front().getOperations().front();

  // Hard-code: set the #1 argument to be constant.
  // OpBuilder builder(op->getContext());
  // func.setAttr("runtime_const_args_index",
  //     builder.getI32ArrayAttr({1,2,3,4}));

  RunConstantSubgraphAnalyser runAnalyser;
  (void)runAnalyser.run(&func);
}

std::unique_ptr<Pass> createConstantSubgraphAnalysisPass() {
  return std::make_unique<ConstantSubgraphAnalysis>();
}

} // namespace gc
} // namespace mlir