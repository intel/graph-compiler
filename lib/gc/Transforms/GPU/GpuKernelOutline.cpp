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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
      // Convert scf.forall ops annotated with GC_ATTR_KERNEL_NAME attribute
      // into parallel loops.
      auto result = funcOp.walk([&](scf::ForallOp forallOp) {
        rw.setInsertionPoint(forallOp);
        auto kernelName = dyn_cast_if_present<StringAttr>(
            forallOp->getAttr(GC_ATTR_KERNEL_NAME));

        if (!kernelName) {
          if (failed(scf::forallToForLoop(rw, forallOp))) {
            signalPassFailure();
            return WalkResult::interrupt();
          }
          return WalkResult::skip();
        }

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
      if (auto nameAttr =
              launch.getOperation()->getDiscardableAttr(GC_ATTR_KERNEL_NAME)) {
        auto name = getAttrValue<StringRef>(nameAttr);
        launch.setModule(name);
        launch.setFunction(name);
        if (auto threadsAttr = KernelAttrs(op, name).getThreads()) {
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

    inlineSplatArgs(rw, op);

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

private:
  // Move global splat constants, that are passed to the kernel as memref args,
  // inside the kernel.
  void inlineSplatArgs(IRRewriter &rw, ModuleOp mod) {
    mod.walk([&](gpu::LaunchFuncOp launch) {
      gpu::GPUFuncOp gpuFunc;
      if (auto kmod = mod.lookupSymbol<gpu::GPUModuleOp>(
              launch.getKernelModuleName())) {
        gpuFunc = kmod.lookupSymbol<gpu::GPUFuncOp>(launch.getKernelName());
        if (!gpuFunc)
          return;
      }

      auto operands = launch.getKernelOperands();
      unsigned numOperands = operands.size();
      llvm::BitVector toErase(numOperands);

      for (unsigned i = 0; i < numOperands; ++i) {
        DenseElementsAttr attr = nullptr;
        if (auto getGlobal = operands[i].getDefiningOp<memref::GetGlobalOp>()) {
          if (auto globalOp =
                  mod.lookupSymbol<memref::GlobalOp>(getGlobal.getNameAttr());
              globalOp && globalOp.getConstant()) {
            if (auto init = globalOp.getInitialValue()) {
              attr = dyn_cast<DenseElementsAttr>(*init);
            }
          }
        }
        if (attr && attr.isSplat()) {
          toErase.set(i);
          auto arg = gpuFunc.getArgument(i);
          assert(isa<MemRefType>(arg.getType()));
          for (auto *user : llvm::make_early_inc_range(arg.getUsers())) {
            if (auto tr = dyn_cast<vector::TransferReadOp>(user)) {
              rw.setInsertionPoint(tr);
              auto cst = arith::ConstantOp::create(
                  rw, tr.getLoc(), attr.resizeSplat(tr.getVectorType()));
              rw.replaceOp(tr, cst.getResult());
            }
          }
        }
      }

      if (auto numErase = toErase.count()) {
        // Rebuild gpu.launch_func without the erased operands.
        SmallVector<Value> newOperands;
        newOperands.reserve(numOperands - numErase);
        for (unsigned i = 0; i < numOperands; ++i) {
          if (!toErase.test(i))
            newOperands.push_back(launch.getKernelOperand(i));
        }
        rw.setInsertionPoint(launch);
        gpu::LaunchFuncOp::create(
            rw, launch.getLoc(), gpuFunc,
            gpu::KernelDim3{launch.getGridSizeX(), launch.getGridSizeY(),
                            launch.getGridSizeZ()},
            gpu::KernelDim3{launch.getBlockSizeX(), launch.getBlockSizeY(),
                            launch.getBlockSizeZ()},
            launch.getDynamicSharedMemorySize(), newOperands);
        rw.eraseOp(launch);

        if (gpuFunc.eraseArguments(toErase).succeeded()) {
          // Erase the unused global constants.
          for (unsigned i = 0; i < numOperands; ++i) {
            if (toErase.test(i)) {
              auto getGlobal = operands[i].getDefiningOp<memref::GetGlobalOp>();
              if (getGlobal.use_empty()) {
                auto globalOp =
                    mod.lookupSymbol<memref::GlobalOp>(getGlobal.getNameAttr());
                rw.eraseOp(getGlobal);
                if (SymbolTable::symbolKnownUseEmpty(globalOp, mod))
                  rw.eraseOp(globalOp);
              }
            }
          }
        }
      }
    });
  }
};

} // namespace
