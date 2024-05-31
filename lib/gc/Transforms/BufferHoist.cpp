//===- BufferHoist.cpp - Buffer hoist in nested parallel loop --------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "gc/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir::gc {

#define GEN_PASS_DEF_BUFFERNESTEDPARALLELLOOPHOISTING
#include "gc/Transforms/Passes.h.inc"
namespace {

class BufferHoisting : public BufferPlacementTransformationBase {
public:
  BufferHoisting(Operation *op) : BufferPlacementTransformationBase(op) {}

  LogicalResult hoist() {
    // 1. Find all the buffers that are used in the nested parallel loop.
    // 2. Find the outermost parallel loop.
    // 3. Hoist the buffers to the outermost parallel loop.
    // 4. Update the uses of the buffers in the nested parallel loop.
    if (false)
      return failure();

    return success();
  }
};

static LogicalResult hoistBuffersFromNestedParallelLoop(Operation *op) {
  BufferHoisting optimizer(op);
  return optimizer.hoist();
};

class BufferNestedParallelLoopHoisting
    : public impl::BufferNestedParallelLoopHoistingBase<
          BufferNestedParallelLoopHoisting> {
public:
  friend struct PassHelper;
  using impl::BufferNestedParallelLoopHoistingBase<
      BufferNestedParallelLoopHoisting>::BufferNestedParallelLoopHoistingBase;

  void runOnOperation() final {
    auto op = getOperation();
    if (failed(hoistBuffersFromNestedParallelLoop(op)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createBufferNestedParallelLoopHoistingPass() {
  return std::make_unique<BufferNestedParallelLoopHoisting>();
}

} // namespace mlir::gc
