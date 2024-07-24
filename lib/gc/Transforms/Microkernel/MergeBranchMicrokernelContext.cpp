//===- MergeBranchMicrokernelContext.cpp -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Transforms/Microkernel/BrgemmRuntimeUtils.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Utils/ValueUtils.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_MERGEBRANCHMICROKERNELCONTEXT
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "merge-branch-microkernel-context"

// enum BrgemmCallType { INAPPLICABLE = -1, DISPATCH, TILECFG, TILERELEASE };

class BrgemmDispatchAnalysis {
private:
  // A map for tile_config -> tile_dispatch
  DenseMap<Operation *, Operation *> brgemmDispatches;

  Operation *traceKernelDispatch(Operation *op);
  Operation *traceDispatchInGlobalCtor(ModuleOp module,
                                       llvm::StringRef global_name);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BrgemmDispatchAnalysis)
  explicit BrgemmDispatchAnalysis(Operation *);
  void setKernelDispatch(Operation *tilecfg, Operation *dispatch) {
    LLVM_DEBUG(llvm::dbgs() << "* setKernelDispatch: " << tilecfg << "; "
                            << dispatch << "\n");
    brgemmDispatches[tilecfg] = dispatch;
  };
  Operation *getKernelDispatch(Operation *tilecfg) const {
    auto iter = brgemmDispatches.find(tilecfg);
    if (iter == brgemmDispatches.end()) {
      return nullptr;
    }
    return iter->second;
  };
};

BrgemmDispatchAnalysis::BrgemmDispatchAnalysis(Operation *root) {
  LLVM_DEBUG(llvm::dbgs() << "* construct BrgemmDispatchAnalysis: " << *root
                          << "\n");
  ModuleOp module = dyn_cast_or_null<ModuleOp>(root);
  if (!module)
    return;

  module->walk<WalkOrder::PreOrder>([this](Operation *op) {
    auto callOp = dyn_cast_or_null<func::CallOp>(op);
    if (!callOp)
      return;
    StringAttr callee = callOp.getCalleeAttr().getAttr();
    if (callee != StringAttr::get(op->getContext(), DNNL_BRGEMM_TILECFG_NAME))
      return;

    auto dispatch = traceKernelDispatch(callOp);
    assert(dispatch && "No dispatch found for tilecfg Op");
    setKernelDispatch(callOp, dispatch);
  });
}

Operation *BrgemmDispatchAnalysis::traceKernelDispatch(Operation *op) {
  ModuleOp module = op->template getParentOfType<ModuleOp>();
  auto callOp = dyn_cast_or_null<func::CallOp>(op);
  assert(callOp);
  StringAttr callee = callOp.getCalleeAttr().getAttr();
  assert(callee == StringAttr::get(op->getContext(), DNNL_BRGEMM_TILECFG_NAME));
  auto kernelProducer = callOp.getOperand(0).getDefiningOp();
  // Direct producer is supposed to be either `brgemm.dispatch` or LLVM::load
  // global Any other cases are extremely rare (mostly invalid MLIR), so
  // considered as not found
  if (auto tryCallOp = dyn_cast_or_null<func::CallOp>(kernelProducer)) {
    callee = tryCallOp.getCalleeAttr().getAttr();
    if (callee != StringAttr::get(op->getContext(), DNNL_BRGEMM_DISPATCH_NAME))
      return nullptr;
    return tryCallOp;
  } else if (auto tryLoadOp = dyn_cast_or_null<LLVM::LoadOp>(kernelProducer)) {
    auto tryAddrOfOp = dyn_cast_or_null<LLVM::AddressOfOp>(
        tryLoadOp.getOperand().getDefiningOp());
    if (!tryAddrOfOp)
      return nullptr;
    return traceDispatchInGlobalCtor(module, tryAddrOfOp.getGlobalName());
  }
  return nullptr;
}

Operation *
BrgemmDispatchAnalysis::traceDispatchInGlobalCtor(ModuleOp module,
                                                  llvm::StringRef global_name) {
  std::string gctor_name = std::string(global_name) + "_ctor";
  FlatSymbolRefAttr ctorName =
      SymbolRefAttr::get(module->getContext(), gctor_name);
  auto ctor = module.lookupSymbol<LLVM::LLVMFuncOp>(ctorName);
  if (!ctor)
    return nullptr;

  // ctor should contain only one call for kernel dispatch
  auto &body = ctor.getBody();
  for (auto &opRef : body.getOps()) {
    auto *op = &opRef;
    auto tryCallOp = dyn_cast_or_null<func::CallOp>(op);
    if (!tryCallOp)
      continue;
    auto callee = tryCallOp.getCalleeAttr().getAttr();
    if (callee == StringAttr::get(op->getContext(), DNNL_BRGEMM_DISPATCH_NAME))
      return op;
  }
  return nullptr;
}

// return a pair of <tilecfg Op, tilerelease Op> extracted from given region,
// only check direct descendants
static std::pair<Operation *, Operation *>
extractTileOpsFromRegion(Region &region) {
  std::pair<Operation *, Operation *> ret{nullptr, nullptr};

  for (auto &opRef : region.getOps()) {
    LLVM_DEBUG(llvm::dbgs() << ">>> " << opRef << "\n");
    auto *op = &opRef;
    auto tryCallOp = dyn_cast_or_null<func::CallOp>(op);
    if (!tryCallOp)
      continue;
    auto callee = tryCallOp.getCalleeAttr().getAttr();
    if (callee == StringAttr::get(op->getContext(), DNNL_BRGEMM_TILECFG_NAME))
      ret.first = op;
    else if (callee ==
             StringAttr::get(op->getContext(), DNNL_BRGEMM_TILERELEASE_NAME))
      ret.second = op;
  }

  return ret;
}

static bool dispatchHasSameContext(Operation *lhs, Operation *rhs) {
  auto lhsDispatch = dyn_cast_or_null<func::CallOp>(lhs);
  auto rhsDispatch = dyn_cast_or_null<func::CallOp>(rhs);
  if (!lhsDispatch || !rhsDispatch)
    return false;
  auto dispatchNameAttr =
      StringAttr::get(lhs->getContext(), DNNL_BRGEMM_DISPATCH_NAME);
  if (lhsDispatch.getCalleeAttr().getAttr() != dispatchNameAttr ||
      rhsDispatch.getCalleeAttr().getAttr() != dispatchNameAttr)
    return false;

  auto lhsOperands = lhsDispatch.getOperands();
  auto rhsOperands = rhsDispatch.getOperands();
  assert(lhsOperands.size() == rhsOperands.size() &&
         "Inconsistent operand size");
  for (size_t idx = 0; idx < lhsOperands.size(); idx++) {
    if (idx == 8) {
      // skip `beta` operand in index no.8
      // since per dnnl design, it does not affect BRGEMM blocking & palettes
      continue;
    }
    auto lhsCstOp =
        dyn_cast_or_null<arith::ConstantOp>(lhsOperands[idx].getDefiningOp());
    auto rhsCstOp =
        dyn_cast_or_null<arith::ConstantOp>(rhsOperands[idx].getDefiningOp());
    if (!lhsCstOp || !rhsCstOp)
      return false;
    if (lhsCstOp.getValue() != rhsCstOp.getValue())
      return false;
  }
  return true;
}

class ScfIfRewriter : public OpRewritePattern<scf::IfOp> {
private:
  BrgemmDispatchAnalysis &analysis;

public:
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  ScfIfRewriter(MLIRContext *context, BrgemmDispatchAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const final {
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    auto &ifRegion = op.getThenRegion();
    auto &elseRegion = op.getElseRegion();
    if (!ifRegion.hasOneBlock() || !elseRegion.hasOneBlock())
      return rewriter.notifyMatchFailure(op,
                                         "Cannot merge for non-full branch");
    auto ifTileOps = extractTileOpsFromRegion(ifRegion);
    auto elseTileOps = extractTileOpsFromRegion(elseRegion);
    if (!ifTileOps.first || !ifTileOps.second || !elseTileOps.first ||
        !elseTileOps.second)
      return rewriter.notifyMatchFailure(
          op, "Cannot merge for inconsistent branch");

    auto ifTileDispatch = analysis.getKernelDispatch(ifTileOps.first);
    auto elseTileDispatch = analysis.getKernelDispatch(elseTileOps.first);
    if (!ifTileDispatch || !elseTileDispatch)
      return rewriter.notifyMatchFailure(op, "Cannot find kernel dispatch");

    if (!dispatchHasSameContext(ifTileDispatch, elseTileDispatch))
      return rewriter.notifyMatchFailure(
          op, "Kernels in branch has different context");

    // Avoid breaking dominance of dispatch
    if (ifTileDispatch->getParentRegion() == &ifRegion)
      return rewriter.notifyMatchFailure(op,
                                         "Dispatch dominance prevents merging");

    // Whole branch reuses internal context of kernel in `if` region
    rewriter.eraseOp(elseTileOps.first);
    rewriter.eraseOp(elseTileOps.second);
    rewriter.moveOpBefore(ifTileOps.first, op);
    rewriter.moveOpAfter(ifTileOps.second, op);

    return success();
  }
};

class ScfIndexSwitchRewriter : public OpRewritePattern<scf::IndexSwitchOp> {
private:
  BrgemmDispatchAnalysis &analysis;

public:
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;

  ScfIndexSwitchRewriter(MLIRContext *context, BrgemmDispatchAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const final {
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    auto &defaultRegion = op.getDefaultRegion();
    auto caseRegions = op.getCaseRegions();

    auto defaultTileOps = extractTileOpsFromRegion(defaultRegion);
    if (!defaultTileOps.first || !defaultTileOps.second)
      return rewriter.notifyMatchFailure(
          op, "Cannot merge for inconsistent branch");
    SmallVector<std::pair<Operation *, Operation *>, 5> caseTilesOps;
    for (auto &caseRegion : caseRegions) {
      auto caseTileOps = extractTileOpsFromRegion(caseRegion);
      if (!caseTileOps.first || !caseTileOps.second)
        return rewriter.notifyMatchFailure(
            op, "Cannot merge for inconsistent branch");
      caseTilesOps.push_back(caseTileOps);
    }

    auto defaultTileDispatch = analysis.getKernelDispatch(defaultTileOps.first);
    if (!defaultTileDispatch)
      return rewriter.notifyMatchFailure(op, "Cannot find kernel dispatch");

    for (size_t idx = 0; idx < caseRegions.size(); idx++) {
      auto &caseRegion = caseRegions[idx];
      auto caseTileDispatch =
          analysis.getKernelDispatch(caseTilesOps[idx].first);
      if (!defaultTileDispatch)
        return rewriter.notifyMatchFailure(op, "Cannot find kernel dispatch");
      if (!dispatchHasSameContext(defaultTileDispatch, caseTileDispatch))
        return rewriter.notifyMatchFailure(
            op, "Kernels in branch has different context");
    }

    // Avoid breaking dominance of dispatch
    if (defaultTileDispatch->getParentRegion() == &defaultRegion)
      return rewriter.notifyMatchFailure(op,
                                         "Dispatch dominance prevents merging");

    // Whole branch reuses internal context of kernel in `default` region
    for (auto &ops : caseTilesOps) {
      rewriter.eraseOp(ops.first);
      rewriter.eraseOp(ops.second);
    }
    rewriter.moveOpBefore(defaultTileOps.first, op);
    rewriter.moveOpAfter(defaultTileOps.second, op);

    return success();
  }
};

class MergeBranchMicrokernelContext
    : public impl::MergeBranchMicrokernelContextBase<
          MergeBranchMicrokernelContext> {
public:
  using impl::MergeBranchMicrokernelContextBase<
      MergeBranchMicrokernelContext>::MergeBranchMicrokernelContextBase;
  void runOnOperation() final {
    auto &dispatchAnalysis = getAnalysis<BrgemmDispatchAnalysis>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ScfIfRewriter>(&getContext(), dispatchAnalysis);
    patterns.add<ScfIndexSwitchRewriter>(&getContext(), dispatchAnalysis);
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::microkernel
