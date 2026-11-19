//===------ SetKernelDepends.cpp - Set GPU kernel deps ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Utils/Transform.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_SETKERNELDEPENDS
#define GEN_PASS_DEF_SETKERNELDEPENDS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct SetKernelDepends final
    : ::mlir::gc::impl::SetKernelDependsBase<SetKernelDepends> {
  void runOnOperation() override {
    // Dependency collection only follows direct users. Reject reshapes outside
    // kernels before changing metadata, until indirect dependencies are
    // handled.
    auto validation =
        getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op) ||
              getKernelLoop(op))
            return WalkResult::advance();
          op->emitOpError(
              "untiled reshape outside a GPU kernel; tile or fuse it before "
              "collecting kernel dependencies");
          return WalkResult::interrupt();
        });
    if (validation.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    getOperation().walk([&](scf::ForallOp forallOp) {
      auto kernelName = dyn_cast_if_present<StringAttr>(
          forallOp->getAttr(GC_ATTR_KERNEL_NAME));
      if (!kernelName) return WalkResult::skip();

      for (auto result : forallOp.getResults()) {
        // A result can feed multiple kernels, with multiple uses in each.
        for (auto user : result.getUsers()) {
          auto kernel = getKernelLoop(user);
          if (!kernel) continue;
          KernelAttrs attrs(kernel, getAttrValue<StringRef>(
                                        kernel->getAttr(GC_ATTR_KERNEL_NAME)));
          SmallVector<StringRef> depends =
              attrs.getDepends().value_or(SmallVector<StringRef>());
          if (std::find(depends.begin(), depends.end(),
                        kernelName.getValue()) == depends.end()) {
            depends.push_back(kernelName.getValue());
            attrs.setDepends(depends);
          }
        }
      }

      return WalkResult::skip();
    });
  }
};

} // namespace
