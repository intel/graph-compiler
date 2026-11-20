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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "plan-tile-and-fuse"

namespace mlir::gc {
#define GEN_PASS_DECL_PLANTILEANDFUSE
#define GEN_PASS_DEF_PLANTILEANDFUSE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

enum class FusionDirection { Producer, Consumer };

/// State maintained by the fusion driver. Checks receive a const view. The
/// driver alone adds members, updates ownership, and refreshes boundaries.
struct FusionGroup {
  FusionGroup(int64_t id, Operation *anchor) : id(id), anchor(anchor) {}

  const int64_t id;
  Operation *const anchor;
  llvm::SmallSetVector<Operation *, 8> members;
  /// External values used by members, including arguments and region captures.
  llvm::SmallSetVector<Value, 8> inputs;
  /// Member results with users outside the group (including returns).
  llvm::SmallSetVector<Value, 8> outputs;

  bool contains(Operation *op) const { return members.contains(op); }
};

/// Read-only queries over the driver's current ownership and group boundaries.
/// The driver must keep groups and the ownership map alive for this view.
class FusionAnalysis {
public:
  using OwnershipMap = llvm::DenseMap<Operation *, const FusionGroup *>;

  explicit FusionAnalysis(const OwnershipMap &ownership)
      : ownership(ownership) {}

  /// Null denotes an unassigned operation, including non-fusible operations.
  const FusionGroup *getOwner(Operation *op) const {
    return ownership.lookup(op);
  }

  /// Include region captures of unassigned operations: even a non-fusible
  /// operation can carry an alternate dependency path back to a group.
  SmallVector<Value> getDependencyInputs(Operation *op) const {
    if (const FusionGroup *group = getOwner(op))
      return SmallVector<Value>(group->inputs.begin(), group->inputs.end());
    llvm::SetVector<Value> inputs;
    inputs.insert(op->operand_begin(), op->operand_end());
    getUsedValuesDefinedAbove(op->getRegions(), inputs);
    return SmallVector<Value>(inputs.begin(), inputs.end());
  }

private:
  const OwnershipMap &ownership;
};

/// A proposed addition. References observe current driver state without copies.
struct FusionCandidate {
  Operation *op;
  const FusionGroup &group;
  FusionDirection direction;
  const FusionAnalysis &analysis;
};

/// Independent, cumulative requirements for a candidate. Checks must not
/// mutate the IR, membership, boundaries, or ownership. A direction-specific
/// condition returns success when it does not apply.
class FusionLegalityCondition {
public:
  virtual ~FusionLegalityCondition() = default;
  virtual llvm::StringRef getDebugName() const = 0;
  virtual LogicalResult check(const FusionCandidate &candidate) const = 0;
};

class FusionLegalityConditionSet {
public:
  /// Register conditions in template argument order. Constructor arguments are
  /// forwarded to each condition.
  template <typename... Ts, typename... Args> void add(Args &&...args) {
    static_assert(sizeof...(Ts) > 0, "expected at least one condition");
    static_assert((std::is_base_of_v<FusionLegalityCondition, Ts> && ...),
                  "expected a fusion legality condition");
    (conditions.push_back(std::make_unique<Ts>(std::forward<Args>(args)...)),
     ...);
  }

  /// Evaluate in registration order, stopping at the first failure. An empty
  /// set succeeds. Rejection is an ordinary result, not a pass diagnostic.
  LogicalResult checkAll(const FusionCandidate &candidate) const {
    for (const auto &condition : conditions) {
      if (failed(condition->check(candidate))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Fusion candidate "
                   << candidate.op->getName().getStringRef() << " rejected by "
                   << condition->getDebugName() << "\n");
        return failure();
      }
    }
    return success();
  }

private:
  llvm::SmallVector<std::unique_ptr<FusionLegalityCondition>> conditions;
};

static bool isAnchor(Operation *op) {
  return gc::isMatmulOp(op) ||
         isa<linalgx::GroupedMatmulOp, linalgx::AttentionOp>(op);
}

class AnchorBoundaryCondition final : public FusionLegalityCondition {
public:
  llvm::StringRef getDebugName() const override {
    return "AnchorBoundaryCondition";
  }

  LogicalResult check(const FusionCandidate &candidate) const override {
    return success(!isAnchor(candidate.op));
  }
};

class SupportedOperationAndScopeCondition final
    : public FusionLegalityCondition {
public:
  llvm::StringRef getDebugName() const override {
    return "SupportedOperationAndScopeCondition";
  }

  LogicalResult check(const FusionCandidate &candidate) const override {
    Block *block = candidate.group.anchor->getBlock();
    auto supportedInScope = [&](Operation *op) {
      return op->getBlock() == block && isSupported(op);
    };
    return success(supportedInScope(candidate.op) &&
                   llvm::all_of(candidate.group.members, supportedInScope));
  }

private:
  static bool isSupported(Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>() || op->getNumSuccessors() ||
        !bufferization::hasTensorSemantics(op))
      return false;
    auto isMemref = [](Type type) { return isa<BaseMemRefType>(type); };
    if (llvm::any_of(op->getOperandTypes(), isMemref) ||
        llvm::any_of(op->getResultTypes(), isMemref))
      return false;

    // These tensor-only anchors have known pure semantics, but do not currently
    // implement MemoryEffectOpInterface.
    if (isa<linalgx::GroupedMatmulOp, linalgx::AttentionOp>(op)) return true;

    // Linalg payloads are supported; other region operations need a richer
    // dependency model before their regions can be incorporated into a group.
    if (op->getNumRegions() && !isa<linalg::LinalgOp>(op)) return false;
    WalkResult effects = op->walk([&](Operation *nested) {
      return isMemoryEffectFree(nested) ? WalkResult::advance()
                                        : WalkResult::interrupt();
    });
    if (effects.wasInterrupted()) return false;

    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(op->getRegions(), captures);
    return llvm::all_of(captures, [&](Value value) {
      return llvm::is_contained(op->getOperands(), value);
    });
  }
};

class OwnershipCondition final : public FusionLegalityCondition {
public:
  llvm::StringRef getDebugName() const override { return "OwnershipCondition"; }

  LogicalResult check(const FusionCandidate &candidate) const override {
    return success(!candidate.analysis.getOwner(candidate.op) &&
                   !candidate.group.contains(candidate.op));
  }
};

class BoundaryAdjacencyCondition final : public FusionLegalityCondition {
public:
  llvm::StringRef getDebugName() const override {
    return "BoundaryAdjacencyCondition";
  }

  LogicalResult check(const FusionCandidate &candidate) const override {
    if (candidate.direction == FusionDirection::Producer)
      return success(
          llvm::any_of(candidate.op->getResults(), [&](Value result) {
            return candidate.group.inputs.contains(result);
          }));
    return success(
        llvm::any_of(candidate.op->getOperands(), [&](Value operand) {
          return candidate.group.outputs.contains(operand);
        }));
  }
};

class ProducerSingleUserCondition final : public FusionLegalityCondition {
public:
  llvm::StringRef getDebugName() const override {
    return "ProducerSingleUserCondition";
  }

  LogicalResult check(const FusionCandidate &candidate) const override {
    if (candidate.direction != FusionDirection::Producer) return success();
    llvm::SmallPtrSet<Operation *, 4> users;
    for (Value result : candidate.op->getResults())
      for (Operation *user : result.getUsers()) {
        users.insert(user);
        if (users.size() > 1) return failure();
      }
    return success(users.size() == 1 &&
                   candidate.group.contains(*users.begin()));
  }
};

class ConsumerAcyclicityCondition final : public FusionLegalityCondition {
public:
  llvm::StringRef getDebugName() const override {
    return "ConsumerAcyclicityCondition";
  }

  LogicalResult check(const FusionCandidate &candidate) const override {
    if (candidate.direction != FusionDirection::Consumer) return success();
    SmallVector<Operation *> worklist;
    for (Value operand : candidate.op->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        if (!candidate.group.contains(producer)) worklist.push_back(producer);

    llvm::SmallPtrSet<Operation *, 16> visited;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      const FusionGroup *owner = candidate.analysis.getOwner(op);
      if (owner == &candidate.group) return failure();

      // All members of another group denote one dependency node.
      if (!visited.insert(owner ? owner->anchor : op).second) continue;
      for (Value input : candidate.analysis.getDependencyInputs(op))
        if (Operation *producer = input.getDefiningOp())
          worklist.push_back(producer);
    }
    return success();
  }
};

static void recomputeBoundaries(FusionGroup &group) {
  group.inputs.clear();
  group.outputs.clear();
  for (Operation *member : group.members) {
    // Seeded anchors may have captures that legality checks would reject during
    // expansion. Their dependencies must still be visible to other groups.
    llvm::SetVector<Value> inputs;
    inputs.insert(member->operand_begin(), member->operand_end());
    getUsedValuesDefinedAbove(member->getRegions(), inputs);
    for (Value input : inputs)
      if (!group.contains(input.getDefiningOp())) group.inputs.insert(input);

    for (Value result : member->getResults())
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !group.contains(user); }))
        group.outputs.insert(result);
  }
}

/// Snapshot distinct boundary neighbors in function preorder, independent of
/// value use-list order or the order in which members were added.
static SmallVector<Operation *>
getCandidates(const FusionGroup &group, FusionDirection direction,
              const llvm::DenseMap<Operation *, unsigned> &operationOrder) {
  llvm::SmallSetVector<Operation *, 8> neighbors;
  if (direction == FusionDirection::Producer) {
    for (Value input : group.inputs)
      if (Operation *producer = input.getDefiningOp())
        neighbors.insert(producer);
  } else {
    for (Value output : group.outputs)
      for (Operation *consumer : output.getUsers()) neighbors.insert(consumer);
  }

  SmallVector<Operation *> candidates(neighbors.begin(), neighbors.end());
  llvm::sort(candidates, [&](Operation *lhs, Operation *rhs) {
    return operationOrder.lookup(lhs) < operationOrder.lookup(rhs);
  });
  return candidates;
}

struct PlanTileAndFuse final : gc::impl::PlanTileAndFuseBase<PlanTileAndFuse> {
  void runOnOperation() override {
    // Validate the whole function before changing any annotations.
    WalkResult result =
        getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!op->hasAttr(gc::GC_ATTR_FUSION_GROUP))
            return WalkResult::advance();
          op->emitError("plan-tile-and-fuse requires input without "
                        "gc.fusion_group attributes");
          return WalkResult::interrupt();
        });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    SmallVector<std::unique_ptr<FusionGroup>> groups;
    FusionAnalysis::OwnershipMap ownership;
    const FusionAnalysis analysis(ownership);
    llvm::DenseMap<Operation *, unsigned> operationOrder;
    int64_t nextGroupId = 0;

    // Stable heap allocations keep ownership references valid as groups grow.
    // Leave all other operations unassigned until expansion has converged.
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      operationOrder.try_emplace(op, operationOrder.size());
      if (!isAnchor(op)) return;
      auto group = std::make_unique<FusionGroup>(nextGroupId++, op);
      group->members.insert(op);
      ownership.try_emplace(op, group.get());
      recomputeBoundaries(*group);
      groups.push_back(std::move(group));
    });

    FusionLegalityConditionSet conditions;
    conditions.add<AnchorBoundaryCondition, SupportedOperationAndScopeCondition,
                   OwnershipCondition, BoundaryAdjacencyCondition,
                   ProducerSingleUserCondition, ConsumerAcyclicityCondition>();
    auto tryAdd = [&](Operation *op, FusionGroup &group,
                      FusionDirection direction) {
      if (failed(conditions.checkAll({op, group, direction, analysis})))
        return false;

      // Preserve existing assignments even if no legality checks are installed.
      // Only adding a previously unowned operation counts as progress.
      if (analysis.getOwner(op) || group.contains(op)) return false;
      group.members.insert(op);
      ownership.try_emplace(op, &group);
      recomputeBoundaries(group);
      return true;
    };

    bool changed;
    do {
      changed = false;
      for (const auto &group : groups) {
        for (Operation *producer :
             getCandidates(*group, FusionDirection::Producer, operationOrder))
          changed |= tryAdd(producer, *group, FusionDirection::Producer);

        // Consumer discovery observes all additions from the producer phase.
        for (Operation *consumer :
             getCandidates(*group, FusionDirection::Consumer, operationOrder))
          changed |= tryAdd(consumer, *group, FusionDirection::Consumer);
      }
    } while (changed);

    // Materialize final memberships only after the fixed point.
    MLIRContext *ctx = &getContext();
    auto tileSizes = DenseI64ArrayAttr::get(ctx, {});
    for (const auto &group : groups) {
      auto attr = gc::FusionGroupAttr::get(ctx, group->id, tileSizes);
      for (Operation *member : group->members)
        member->setAttr(gc::GC_ATTR_FUSION_GROUP, attr);
    }

    // Give unassigned tensor-semantics operations singleton groups.
    getOperation().getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!op->hasTrait<OpTrait::IsTerminator>() &&
          bufferization::hasTensorSemantics(op) && !analysis.getOwner(op)) {
        auto attr = gc::FusionGroupAttr::get(ctx, nextGroupId++, tileSizes);
        op->setAttr(gc::GC_ATTR_FUSION_GROUP, attr);
      }
    });
  }
};

} // namespace
