//===-- VerifyTargetDescription.cpp - Verity target desc --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Pass/Pass.h"
using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_VERIFYTARGETDESCRIPTION
#include "gc/Transforms/Passes.h.inc"

namespace {

static LogicalResult verifyCPUTargetDescription(RewriterBase &rewriter,
                                                Operation *op) {
  CPUTargetDescriptionAnalysis cpuTargetDesc(op);
  Location loc = op->getLoc();

  // Check if the num_threads is existed and greater than 0
  std::optional<Attribute> numThreadsAttr =
      cpuTargetDesc.getPropertyValue(CPUTargetDescriptionAnalysis::kNumThreads);
  if (numThreadsAttr) {
    if (!isa<IntegerAttr>(*numThreadsAttr) ||
        cpuTargetDesc.getNumThreads() < 1) {
      mlir::emitError(loc)
          << "num_threads must be a greater than 0 integer, but get "
          << *numThreadsAttr;
      return failure();
    }
  }

  // Check if the L1 cache size is existed and greater than 0
  std::optional<Attribute> l1CacheSizeAttr = cpuTargetDesc.getPropertyValue(
      CPUTargetDescriptionAnalysis::kL1CacheSize);
  if (l1CacheSizeAttr) {
    if (!isa<IntegerAttr>(*l1CacheSizeAttr) ||
        cpuTargetDesc.getCacheSize(1) < 1) {
      mlir::emitError(loc)
          << "L1_cache_size_in_bytes must be a greater than 0 integer, but get "
          << *l1CacheSizeAttr;
      return failure();
    }
  }

  // Check if the L2 cache size is existed and greater than 0
  std::optional<Attribute> l2CacheSizeAttr = cpuTargetDesc.getPropertyValue(
      CPUTargetDescriptionAnalysis::kL2CacheSize);
  if (l2CacheSizeAttr) {
    if (!isa<IntegerAttr>(*l2CacheSizeAttr) ||
        cpuTargetDesc.getCacheSize(2) < 1) {
      mlir::emitError(loc)
          << "L2_cache_size_in_bytes must be a greater than 0 integer, but get "
          << *l2CacheSizeAttr;
      return failure();
    }
  }

  // Check if the L3 cache size is existed and greater than 0
  std::optional<Attribute> l3CacheSizeAttr = cpuTargetDesc.getPropertyValue(
      CPUTargetDescriptionAnalysis::kL3CacheSize);
  if (l3CacheSizeAttr) {
    if (!isa<IntegerAttr>(*l3CacheSizeAttr) ||
        cpuTargetDesc.getCacheSize(3) < 1) {
      mlir::emitError(loc)
          << "L3_cache_size_in_bytes must be a greater than 0 integer, but get "
          << *l3CacheSizeAttr;
      return failure();
    }
  }

  // Check if the max_vector_width is existed and greater than 0
  std::optional<Attribute> maxVectorWidthAttr = cpuTargetDesc.getPropertyValue(
      CPUTargetDescriptionAnalysis::kMaxVectorWidth);
  if (maxVectorWidthAttr) {
    if (!isa<IntegerAttr>(*maxVectorWidthAttr) ||
        cpuTargetDesc.getMaxVectorWidth() < 1) {
      mlir::emitError(loc)
          << "max_vector_width must be a greater than 0 integer, but get "
          << *maxVectorWidthAttr;
      return failure();
    }
  }
  return success();
}

class VerifyTargetDescription
    : public impl::VerifyTargetDescriptionBase<VerifyTargetDescription> {
  using Base::Base;
  void runOnOperation() override {
    Operation *module = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    if (device == "CPU") {
      if (failed(verifyCPUTargetDescription(rewriter, module))) {
        mlir::emitError(module->getLoc())
            << "Failed to verify the target description";
        signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir