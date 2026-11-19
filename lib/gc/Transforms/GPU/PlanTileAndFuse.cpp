//===- PlanTileAndFuse.cpp - Plan tiling and fusion groups ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/IR/GCAttributes.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Transform.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizationDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_PLANTILEANDFUSE
#define GEN_PASS_DEF_PLANTILEANDFUSE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

static bool isAnchor(Operation *op) {
  return gc::isMatmulOp(op) ||
         isa<linalgx::GroupedMatmulOp, linalgx::AttentionOp>(op);
}

struct PlanTileAndFuse final : gc::impl::PlanTileAndFuseBase<PlanTileAndFuse> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto tileSizes = DenseI64ArrayAttr::get(ctx, {});
    int64_t nextGroupId = 0;
    auto assignSingleton = [&](Operation *op) {
      auto group = gc::FusionGroupAttr::get(ctx, nextGroupId++, tileSizes);
      op->setAttr(gc::GC_ATTR_FUSION_GROUP, group);
    };

    // Seed anchor groups before assigning the leftover operations.
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isAnchor(op)) assignSingleton(op);
    });

    // Give unassigned tensor-semantics operations singleton groups.
    getOperation().getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!op->hasTrait<OpTrait::IsTerminator>() &&
          bufferization::hasTensorSemantics(op) &&
          !op->hasAttr(gc::GC_ATTR_FUSION_GROUP))
        assignSingleton(op);
    });
  }
};

} // namespace
