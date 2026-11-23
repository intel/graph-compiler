//===- DropUnitDims.cpp - Drop unit dims ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_DROPUNITDIMS
#define GEN_PASS_DEF_DROPUNITDIMS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct DropUnitDims final : gc::impl::DropUnitDimsBase<DropUnitDims> {

  void runOnOperation() override {
    auto fn = getOperation();
    // Drop the inner most unit dims from vector transfers
    RewritePatternSet patterns(&getContext());
    vector::populateDropInnerMostUnitDimsXferOpPatterns(patterns);
    vector::ShapeCastOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(fn, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
