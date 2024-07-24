//===- MicrokernelInvariantCodeMotion.cpp ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <sstream>
#include <utility>

#include "gc/Transforms/Microkernel/BrgemmRuntimeUtils.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Utils/ValueUtils.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_MICROKERNELINVARIANTCODEMOTION
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "microkernel-invariant-code-motion"

enum BrgemmCallType { INAPPLICABLE = -1, DISPATCH, TILECFG, TILERELEASE };

static bool isParallelLoop(Operation *op) {
  return llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::ParallelOp>(op) ||
         llvm::isa<omp::ParallelOp>(op) || llvm::isa<omp::WsloopOp>(op);
}

static bool isConcernedCF(Operation *op) {
  return llvm::isa<scf::ForOp>(op) || llvm::isa<scf::WhileOp>(op) ||
         llvm::isa<scf::IfOp>(op) || llvm::isa<scf::IndexSwitchOp>(op);
}

static BrgemmCallType getBrgemmCallType(Operation *op) {
  if (!llvm::isa<func::CallOp>(op)) {
    return BrgemmCallType::INAPPLICABLE;
  }
  auto callOp = dyn_cast<func::CallOp>(op);
  StringAttr callee = callOp.getCalleeAttr().getAttr();

  if (callee == StringAttr::get(op->getContext(), DNNL_BRGEMM_DISPATCH_NAME)) {
    return BrgemmCallType::DISPATCH;
  }
  if (callee == StringAttr::get(op->getContext(), DNNL_BRGEMM_TILECFG_NAME)) {
    return BrgemmCallType::TILECFG;
  }
  if (callee ==
      StringAttr::get(op->getContext(), DNNL_BRGEMM_TILERELEASE_NAME)) {
    return BrgemmCallType::TILERELEASE;
  }
  return BrgemmCallType::INAPPLICABLE;
}

// Tree node of structure info tree, each node represents an Op
// This tree contains only concerned Ops
struct BrgemmContextStructInfo {
  // Basic structure info retrieved by first walk
  Operation *contextRoot; // Could be parallel loop or func
  Operation *self, *parent;
  DenseSet<Operation *> child;
  SmallVector<bool, 3> containBrgemmCallType;
  // Rewrite-time info retrieved by analysing basic structure info
  union {
    Operation *maxInvariantScope; // Used by BrgemmCallOps for hoisting
    bool hasTilereleased;         // Used by other Ops as hoisting scopes to
                                  // dedup tilerelease injection
  };
  BrgemmContextStructInfo() {
    contextRoot = nullptr;
    self = nullptr;
    parent = nullptr;
    containBrgemmCallType = {false, false, false};
    maxInvariantScope = nullptr;
  }
};

typedef DenseMap<Operation *, BrgemmContextStructInfo> OpStructInfoMap;

class BrgemmTilecfgRewriter : public OpRewritePattern<func::CallOp> {
private:
  OpStructInfoMap &structInfo;

public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  BrgemmTilecfgRewriter(MLIRContext *context, OpStructInfoMap &si)
      : OpRewritePattern(context), structInfo{si} {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const final {
    ModuleOp module = op->template getParentOfType<ModuleOp>();

    StringAttr callee = op.getCalleeAttr().getAttr();
    if (!module.lookupSymbol(callee))
      return rewriter.notifyMatchFailure(op,
                                         "Invalid CallOp to unknown callee");
    if (callee !=
        StringAttr::get(rewriter.getContext(), DNNL_BRGEMM_TILECFG_NAME))
      return rewriter.notifyMatchFailure(op, "Not call to BRGEMM tilecfg");
    auto opInfoIter = structInfo.find(op);
    if (opInfoIter == structInfo.end()) {
      return rewriter.notifyMatchFailure(op, "Cannot find structInfo for Op");
    }
    auto &opStructInfo = opInfoIter->second;

    // Don't hoist if max invariant scope is itself to reduce
    // unnecessary movement
    if (opStructInfo.maxInvariantScope == op) {
      return rewriter.notifyMatchFailure(op, "No need to hoist");
    }
    rewriter.moveOpBefore(op, opStructInfo.maxInvariantScope);
    // Avoid being hoisted again
    opStructInfo.maxInvariantScope = op;
    return success();
  }
};

static void markScopeAsReleased(OpStructInfoMap &structInfo, Operation *op) {
  auto iter = structInfo.find(op);
  assert(iter != structInfo.end());
  // Don't mark BrgemmCallOps
  if (getBrgemmCallType(op) != BrgemmCallType::INAPPLICABLE)
    return;
  iter->second.hasTilereleased = true;

  for (auto ch : iter->second.child) {
    markScopeAsReleased(structInfo, ch);
  }
}

class BrgemmTilereleaseRewriter : public OpRewritePattern<func::CallOp> {
private:
  OpStructInfoMap &structInfo;

public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  BrgemmTilereleaseRewriter(MLIRContext *context, OpStructInfoMap &si)
      : OpRewritePattern(context), structInfo{si} {}
  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const final {
    ModuleOp module = op->template getParentOfType<ModuleOp>();

    StringAttr callee = op.getCalleeAttr().getAttr();
    if (!module.lookupSymbol(callee))
      return rewriter.notifyMatchFailure(op,
                                         "Invalid CallOp to unknown callee");
    if (callee !=
        StringAttr::get(rewriter.getContext(), DNNL_BRGEMM_TILERELEASE_NAME))
      return rewriter.notifyMatchFailure(op, "Not call to BRGEMM tilerelease");

    auto opInfoIter = structInfo.find(op);
    if (opInfoIter == structInfo.end()) {
      return rewriter.notifyMatchFailure(op, "Cannot find structInfo for Op");
    }
    auto &opStructInfo = opInfoIter->second;
    auto targetInfoIter = structInfo.find(opStructInfo.maxInvariantScope);
    assert(opStructInfo.maxInvariantScope);
    // Don't hoist if max invariant scope is itself to reduce
    // unnecessary movement
    if (opStructInfo.maxInvariantScope == op) {
      return rewriter.notifyMatchFailure(op, "No need to hoist");
    }
    assert(targetInfoIter != structInfo.end());
    // move last tilerelease to end of contextRoot, and remove all
    // others
    if (targetInfoIter->second.hasTilereleased) {
      rewriter.eraseOp(op);
    } else {
      // auto region = opStructInfo.maxInvariantScope->getRegion(0);
      // auto block = &region.getBlocks().front();
      // auto enditer = block->end();
      // rewriter.moveOpBefore(op, block, enditer);
      rewriter.moveOpAfter(op, opStructInfo.maxInvariantScope);
      // Mark all sub scope as released to avoid duplicate Tilerelease
      markScopeAsReleased(structInfo, opStructInfo.maxInvariantScope);
      // Avoid being hoisted again
      opStructInfo.maxInvariantScope = op;
    }
    return success();
  }
};

class MicrokernelInvariantCodeMotion
    : public impl::MicrokernelInvariantCodeMotionBase<
          MicrokernelInvariantCodeMotion> {
private:
  // This helper create structInfo tree node along the path from input Op(as
  // leaf) to contextRoot(FuncOp or any parallel Op) on demand;
  // This tree only contains concerned Ops, including BrgemmCall Ops, parallel
  // ops and related SCF Ops etc.;
  // Input Op should be a BrgemmCall Op or the op
  // depent by BrgemmTilecfg/BrgemmRelease
  BrgemmContextStructInfo getOrCreateBrgemmContext(OpStructInfoMap &structInfo,
                                                   Operation *op) {
    auto resIter = structInfo.find(op);
    if (resIter != structInfo.end()) {
      return resIter->second;
    }

    SmallVector<BrgemmContextStructInfo *, 5> createdInfo;
    Operation *contextRootForCreatedInfo = nullptr;

    auto doCreateStructInfo = [&](Operation *child, Operation *op) {
      BrgemmContextStructInfo info;
      info.self = op;
      if (child) {
        auto iter = structInfo.find(child);
        assert(iter != structInfo.end());
        iter->second.parent = op;
        info.child.insert(child);
      }
      structInfo.insert(std::make_pair(op, std::move(info)));
      auto iter = structInfo.find(op);
      createdInfo.push_back(&iter->second);
      return &iter->second;
    };
    // Create info for input Op as leaf
    auto brgemmInfo = doCreateStructInfo(nullptr, op);
    auto callType = getBrgemmCallType(op);
    if (callType != BrgemmCallType::INAPPLICABLE) {
      brgemmInfo->containBrgemmCallType[callType] = true;
    }

    auto last = op;
    auto current = op->getParentOp();
    // Traverse up the IR tree, creating structInfo for each concerned Op
    while (current) {
      bool isParaLoop = isParallelLoop(current);
      bool isCCF = isConcernedCF(current);
      if (!llvm::isa<func::FuncOp>(current) && !isParaLoop && !isCCF) {
        // Only care about selected Ops
        current = current->getParentOp();
        continue;
      }

      auto iter = structInfo.find(current);
      if (iter != structInfo.end()) {
        // StructInfo exists for current Op, then we don't need to create info
        // anymore as all ancestors have been created
        // But we still need to propagate containBrgemmCallType if we are
        // dealing with BrgemmCall Ops
        if (last) {
          auto lastIter = structInfo.find(last);
          assert(lastIter != structInfo.end());
          lastIter->second.parent = current;
          iter->second.child.insert(last);
          // Invalidate last as we don't create new info anymore
          last = nullptr;
        }
        if (callType != BrgemmCallType::INAPPLICABLE) {
          // Propagate containCallType if needed
          iter->second.containBrgemmCallType[callType] = true;
        } else
          break;
      } else {
        // StructInfo not exist, then create one for current Op and keep
        // Traversing up
        auto created = doCreateStructInfo(last, current);
        if (callType != BrgemmCallType::INAPPLICABLE) {
          created->containBrgemmCallType[callType] = true;
        }
        last = current;
      }
      if (llvm::isa<func::FuncOp>(current) || isParaLoop) {
        // Encounter `contextRoot`, then record and terminate traversing
        contextRootForCreatedInfo = current;
        break;
      }
      current = current->getParentOp();
    }

    // Assign `contextRoot` for newly created structInfo
    if (contextRootForCreatedInfo) {
      for (auto info : createdInfo)
        info->contextRoot = contextRootForCreatedInfo;
    }

    resIter = structInfo.find(op);
    assert(resIter != structInfo.end());
    return resIter->second;
  }

  // This helper expand invariant scope according to two function:
  // 1. controlFlowAllow: Whether we can hoist the BrgemmCallOp out of the scope
  // of current Op; For example, we won't move TILECFG out of an IfOp as it
  // contains underministic control flow.
  // 2. peerAllow: Whether we can hoist the BrgemmCallOp out of the scope of
  // current Op without violating other peer BrgemmCallOp in the same level; For
  // example, one scf.ForOp contains two TILECFG in the same level, then we
  // cannot hoist any of them.
  void expandInvariantScopeWithCond(
      OpStructInfoMap &structInfo, Operation *op,
      std::function<bool(Operation *)> controlFlowAllow,
      std::function<bool(Operation *, const OpStructInfoMap &, Operation *,
                         const DenseSet<Operation *> &)>
          peerAllow) {
    auto opIter = structInfo.find(op);
    assert(opIter != structInfo.end());
    auto contextRoot = opIter->second.contextRoot;
    auto current = op;
    auto currIter = opIter;
    auto parent = opIter->second.parent;
    while (parent != contextRoot) {
      auto parentIter = structInfo.find(parent);
      assert(parentIter != structInfo.end());
      // Verify whether we can expand the scope to direct parent
      bool isControlFlowAllow = controlFlowAllow(parent);
      bool isPeerAllow =
          peerAllow(op, structInfo, current, parentIter->second.child);
      if (!isControlFlowAllow || !isPeerAllow) {
        break;
      }
      current = parent;
      currIter = parentIter;
      parent = parentIter->second.parent;
    }

    opIter->second.maxInvariantScope = current;
  }

  void expandInvariantScope(OpStructInfoMap &structInfo, Operation *op) {
    BrgemmCallType brgemmCallType = getBrgemmCallType(op);
    assert(brgemmCallType == BrgemmCallType::TILECFG ||
           brgemmCallType == BrgemmCallType::TILERELEASE);

    if (brgemmCallType == BrgemmCallType::TILECFG) {
      expandInvariantScopeWithCond(
          structInfo, op,
          [](Operation *op) -> bool {
            return !llvm::isa<scf::IfOp>(op) &&
                   !llvm::isa<scf::IndexSwitchOp>(op);
          },
          [](Operation *self, const OpStructInfoMap &structInfo,
             Operation *current, const DenseSet<Operation *> &peers) -> bool {
            for (auto peer : peers) {
              if (peer == current)
                continue;
              if (peer == self->getOperand(0).getDefiningOp()) {
                // Don't break operand domination
                return false;
              }
              const auto iter = structInfo.find(peer);
              assert(iter != structInfo.end());
              const auto &containType = iter->second.containBrgemmCallType;
              if (containType[BrgemmCallType::DISPATCH] ||
                  containType[BrgemmCallType::TILECFG]) {
                return false;
              }
            }
            return true;
          });
    } else { // brgemmCallType == BrgemmCallType::TILERELEASE
      expandInvariantScopeWithCond(
          structInfo, op,
          [](Operation *op) -> bool {
            return !llvm::isa<scf::IfOp>(op) &&
                   !llvm::isa<scf::IndexSwitchOp>(op);
          },
          [](Operation *self, const OpStructInfoMap &structInfo,
             Operation *current,
             const DenseSet<Operation *> &peers) -> bool { return true; });
    }
  }

  LogicalResult collectBrgemmContextStructInfo(OpStructInfoMap &structInfo) {
    // First walk to collect basic structure
    getOperation()->walk<WalkOrder::PreOrder>(
        [this, &structInfo](Operation *op) {
          BrgemmCallType brgemmCallType = getBrgemmCallType(op);
          if (brgemmCallType == BrgemmCallType::INAPPLICABLE) {
            return;
          }

          // Construct the structInfo tree lazily upon encountering BrgemmCall
          // Op
          auto info = getOrCreateBrgemmContext(structInfo, op);
          structInfo.insert(std::make_pair(op, std::move(info)));
          if (brgemmCallType == BrgemmCallType::TILECFG) {
            // Also contruct tree node for the input of BrgemmTilecfg for
            // dependency check in `expandInvariantScope`
            auto dependOp = op->getOperand(0).getDefiningOp();
            auto dependInfo = getOrCreateBrgemmContext(structInfo, dependOp);
            structInfo.insert(std::make_pair(dependOp, std::move(dependInfo)));
          }
        });

    // Second walk to analyse hoist related info
    getOperation()->walk<WalkOrder::PreOrder>(
        [this, &structInfo](Operation *op) {
          BrgemmCallType brgemmCallType = getBrgemmCallType(op);
          if (brgemmCallType != BrgemmCallType::TILECFG &&
              brgemmCallType != BrgemmCallType::TILERELEASE) {
            return;
          }

          // find the maximal invariant scope for hoisting
          expandInvariantScope(structInfo, op);
        });

    return success();
  }

public:
  using impl::MicrokernelInvariantCodeMotionBase<
      MicrokernelInvariantCodeMotion>::MicrokernelInvariantCodeMotionBase;
  void runOnOperation() final {
    OpStructInfoMap structInfo;

    if (failed(collectBrgemmContextStructInfo(structInfo))) {
      signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<BrgemmTilecfgRewriter>(&getContext(), structInfo);
    patterns.add<BrgemmTilereleaseRewriter>(&getContext(), structInfo);
    FrozenRewritePatternSet patternSet(std::move(patterns));

    // Ignore newly created Ops
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::microkernel
