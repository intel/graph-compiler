//===--------- Vectorize.cpp - Vectorize structured ops ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Utils/Transform.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUKERNELOUTLINE
#define GEN_PASS_DEF_GPUKERNELOUTLINE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuKernelOutline final
    : gc::impl::GpuKernelOutlineBase<GpuKernelOutline> {

  void runOnOperation() override {
    IRRewriter rw(&getContext());
    auto op = getOperation();

    auto result = op.walk([&](func::FuncOp funcOp) {
      // Convert scf.forall ops marked with GC_ATTR_KERNEL_NAME into parallel
      // loops.
      auto result = funcOp.walk([&](scf::ForallOp forallOp) {
        auto kernelName = dyn_cast_if_present<StringAttr>(
            forallOp->getAttr(GC_ATTR_KERNEL_NAME));
        if (!kernelName)
          return WalkResult::skip();

        rw.setInsertionPoint(forallOp);
        scf::ParallelOp parallelOp;

        if (failed(scf::forallToParallelLoop(rw, forallOp, &parallelOp))) {
          signalPassFailure();
          return WalkResult::interrupt();
        }

        parallelOp.getOperation()->setAttr(GC_ATTR_KERNEL_NAME, kernelName);
        return WalkResult::skip();
      });

      if (result.wasInterrupted())
        return WalkResult::interrupt();

      OpPassManager pm(funcOp->getName().getIdentifier(),
                       OpPassManager::Nesting::Implicit);
      pm.addPass(createGpuMapParallelLoopsPass());
      pm.addPass(createConvertParallelLoopToGpuPass());
      pm.addPass(createGpuLaunchSinkIndexComputationsPass());
      if (failed(runPipeline(pm, funcOp))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });

    if (result.wasInterrupted())
      return;

    // Set the number of threads
    op.walk([&](gpu::LaunchOp launch) {
      if (auto name =
              launch.getOperation()->getDiscardableAttr(GC_ATTR_KERNEL_NAME)) {
        if (auto threadsAttr =
                KernelAttrs(op, getAttrValue<StringRef>(name)).getThreads()) {
          auto loc = launch.getLoc();
          auto threads = threadsAttr.value();
          threads.resize(3, 1);
          rw.setInsertionPoint(launch);
          launch.getBlockSizeXMutable().assign(
              arith::ConstantIndexOp::create(rw, loc, threads[0]));
          launch.getBlockSizeYMutable().assign(
              arith::ConstantIndexOp::create(rw, loc, threads[1]));
          launch.getBlockSizeZMutable().assign(
              arith::ConstantIndexOp::create(rw, loc, threads[2]));
        }
      }

      return WalkResult::skip();
    });

    OpPassManager pm(op->getName().getIdentifier(),
                     OpPassManager::Nesting::Implicit);
    pm.addPass(createGpuKernelOutliningPass());
    {
      GpuXeVMAttachTargetOptions opts;
      auto arch = DevAttrs(op).getArch();
      if (arch)
        opts.chip = arch.value().str();
      opts.optLevel = 3;
      pm.addPass(createGpuXeVMAttachTarget(std::move(opts)));
    }
    if (failed(runPipeline(pm, op))) {
      signalPassFailure();
      return;
    }

    // Set the intel_reqd_sub_group_size attribute
    op.walk([&](gpu::GPUFuncOp fn) {
      KernelAttrs attrs(op, fn.getNameAttr());
      if (auto sgSize =
              KernelAttrs(op, fn.getNameAttr()).getSgSize<int32_t>()) {
        fn.getOperation()->setAttr("intel_reqd_sub_group_size",
                                   createAttr(fn.getContext(), sgSize.value()));
      }
      return WalkResult::skip();
    });
  }
};

} // namespace
