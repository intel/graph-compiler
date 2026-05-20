//===--- SetGpuFastMath.cpp - Set fastmath on math.exp in GPU modules -----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_SETGPUFASTMATH
#define GEN_PASS_DEF_SETGPUFASTMATH
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct SetGpuFastMath final : gc::impl::SetGpuFastMathBase<SetGpuFastMath> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool changed = false;

    moduleOp->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod->walk([&](math::ExpOp expOp) {
        if (expOp.getFastmath() != arith::FastMathFlags::fast) {
          expOp.setFastmath(arith::FastMathFlags::fast);
          changed = true;
        }
      });
    });

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace