//===- PlanTileAndFuse.cpp - Plan tiling and fusion groups ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_PLANTILEANDFUSE
#define GEN_PASS_DEF_PLANTILEANDFUSE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

struct PlanTileAndFuse final : gc::impl::PlanTileAndFuseBase<PlanTileAndFuse> {
  void runOnOperation() override { markAllAnalysesPreserved(); }
};

} // namespace
