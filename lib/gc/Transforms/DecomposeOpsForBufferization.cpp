//===-- MemRefToCPURuntime.cpp - MemRef To CPURuntime Lowering --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <vector>

#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/Config/mlir-config.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_DECOMPOSEOPSFORBUFFERIZE
#include "gc/Transforms/Passes.h.inc"

namespace {

struct DecomposeOpsForBufferize
    : public impl::DecomposeOpsForBufferizeBase<DecomposeOpsForBufferize> {

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    ::mlir::tensor::populateDecomposeTensorConcatPatterns(patterns);

    FrozenRewritePatternSet patternSet(std::move(patterns));
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      if (isa<tensor::ConcatOp>(op))
        ops.push_back(op);
    });
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(applyOpPatternsAndFold(ops, patternSet, config)))
      signalPassFailure();
    return;
  }
};

} // namespace
} // namespace gc
} // namespace mlir