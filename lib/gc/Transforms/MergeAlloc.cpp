//===- MergeAlloc.cpp - Calling convention conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "gc/Transforms/StaticMemoryPlanning.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace mlir {
namespace gc {
#define GEN_PASS_DEF_MERGEALLOC
#include "gc-dialects/Passes.h.inc"

/// Return `true` if the given MemRef type has a static identity layout (i.e.,
/// no layout).
static bool hasStaticIdentityLayout(MemRefType type) {
  return type.hasStaticShape() && type.getLayout().isIdentity();
}

namespace {
static constexpr int64_t NO_ACCESS = -1;
static constexpr int64_t COMPLEX_ACCESS = -2;
struct Tick {
  int64_t firstAccess = NO_ACCESS;
  int64_t lastAccess = NO_ACCESS;

  void access(int64_t tick) {
    if (tick == COMPLEX_ACCESS) {
      firstAccess = COMPLEX_ACCESS;
      lastAccess = COMPLEX_ACCESS;
    }
    if (firstAccess == COMPLEX_ACCESS) {
      return;
    }
    if (firstAccess == NO_ACCESS) {
      firstAccess = tick;
    } else {
      firstAccess = std::min(firstAccess, tick);
    }
    lastAccess = std::max(lastAccess, tick);
  }
};

TypedValue<MemRefType> getMemrefBase(TypedValue<MemRefType> v) {
  auto op = v.getDefiningOp();
  if (auto viewop = dyn_cast_if_present<ViewLikeOpInterface>(op)) {
    return getMemrefBase(cast<TypedValue<MemRefType>>(viewop.getViewSource()));
  }
  return v;
}

bool isMergeableAlloc(Operation *op, int64_t tick) {
  if (tick == COMPLEX_ACCESS) {
    return false;
  }
  if (!hasStaticIdentityLayout(
          op->getResultTypes().front().cast<MemRefType>())) {
    return false;
  }
  // currently only support alignment: none, 1, 2, 4, 8, 16, 32, 64
  auto alignment = cast<memref::AllocOp>(op).getAlignment();
  if (!alignment) {
    return true; // ok if no alignment
  }
  return alignment > 0 && (64 % alignment.value() == 0);
}

// find the closest surrounding parent operation with AutomaticAllocationScope
// trait, and is not scf.for
Operation *getAllocScope(Operation *op) {
  auto parent = op;
  for (;;) {
    parent = parent->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (!parent) {
      return nullptr;
    }
    if (!isa<scf::ForOp>(parent)) {
      return parent;
    }
  }
}

FailureOr<size_t> getAllocSize(Operation *op) {
  auto refType = op->getResultTypes().front().cast<MemRefType>();
  int64_t size = refType.getElementTypeBitWidth() / 8;
  // treat bool (i1) as 1 byte. It may not be true for all targets, but we at
  // least have a large enough size for i1
  size = (size != 0) ? size : 1;
  for (auto v : refType.getShape()) {
    size *= v;
  }
  if (size > 0) {
    return static_cast<size_t>(size);
  }
  return op->emitError("Expecting static shaped allocation");
}

// A complex scope object is addition info for a RegionBranchOpInterface or
// LoopLikeOpInterface. It contains the scope itself, and the referenced alloc
// ops inside this scope. We use this object to track which buffers this scope
// accesses. These buffers must have overlapped lifetime
struct ComplexScope {
  Operation *scope;
  int64_t startTick;
  llvm::SmallPtrSet<Operation *, 8> operations;
  ComplexScope(Operation *scope, int64_t startTick)
      : scope{scope}, startTick{startTick} {}
  // called when walk() runs outside of the scope
  void onPop(int64_t endTick, llvm::DenseMap<Operation *, Tick> &allocTicks) {
    for (auto op : operations) {
      // if the allocation is not in the scope, conservatively set the ticks
      if (!scope->isProperAncestor(op)) {
        // let all referenced buffers have overlapped lifetime
        auto &tick = allocTicks[op];
        tick.access(startTick);
        tick.access(endTick);
      }
    }
  }
};

struct TickCollecter {
  int64_t curTick = 0;
  llvm::DenseMap<Operation *, Tick> allocTicks;
  llvm::SmallVector<ComplexScope> complexScopeStack;

  void popScopeIfNecessary(Operation *op) {
    // first check if we have walked outside of the previous ComplexScope
    while (!complexScopeStack.empty()) {
      auto &scope = complexScopeStack.back();
      if (!op || !scope.scope->isProperAncestor(op)) {
        scope.onPop(curTick, allocTicks);
        complexScopeStack.pop_back();
      } else {
        break;
      }
    }
  }

  void forwardTick() { curTick++; }

  void accessValue(Value v, bool complex) {
    if (v.getType().isa<MemRefType>()) {
      auto base = getMemrefBase(llvm::cast<TypedValue<MemRefType>>(v));
      auto defop = base.getDefiningOp();
      if (isa_and_present<memref::AllocOp>(defop)) {
        allocTicks[defop].access(complex ? COMPLEX_ACCESS : curTick);
        if (!complexScopeStack.empty()) {
          complexScopeStack.back().operations.insert(defop);
        }
      }
    }
  }

  void onMemrefViews(ViewLikeOpInterface op) {
    auto viewSrc = op.getViewSource();
    // don't need to access the first operand, which is "source".
    // The "source" operand is not really read or written at this point
    for (auto val : op.getOperation()->getOperands()) {
      if (val != viewSrc)
        accessValue(val, false);
    }
  }

  void onReturnOp(Operation *op) {
    for (auto val : op->getOperands()) {
      accessValue(val, true);
    }
  }

  void onGeneralOp(Operation *op) {
    for (auto val : op->getOperands()) {
      accessValue(val, false);
    }
  }

  void pushComplexScope(Operation *op) {
    complexScopeStack.emplace_back(op, curTick);
  }

  FailureOr<memoryplan::ScopeTraceData> getTrace() {
    struct TraceWithTick {
      int64_t tick;
      memoryplan::MemoryTrace trace;
      TraceWithTick(int64_t tick, uintptr_t bufferId, size_t size)
          : tick{tick}, trace{bufferId, size} {}
    };
    llvm::DenseMap<Operation *, llvm::SmallVector<TraceWithTick, 8>> raw;
    for (auto &[op, tick] : allocTicks) {
      if (!isMergeableAlloc(op, tick.firstAccess)) {
        continue;
      }
      auto scope = getAllocScope(op);
      if (!scope) {
        return op->emitError(
            "This op should be surrounded by an AutomaticAllocationScope");
      }
      auto allocSize = getAllocSize(op);
      if (failed(allocSize)) {
        return allocSize;
      }
      // tick.firstAccess * 2 and tick.lastAccess * 2 + 1 to make sure "dealloc"
      // overlaps "alloc"
      raw[scope].emplace_back(tick.firstAccess * 2,
                              reinterpret_cast<uintptr_t>(op), *allocSize);
      raw[scope].emplace_back(tick.lastAccess * 2 + 1,
                              reinterpret_cast<uintptr_t>(op), 0);
    }
    memoryplan::ScopeTraceData ret;
    for (auto &[scope, trace] : raw) {
      std::stable_sort(trace.begin(), trace.end(),
                       [](const TraceWithTick &a, const TraceWithTick &b) {
                         return a.tick < b.tick;
                       });
      auto &retTrace = ret[scope];
      retTrace.reserve(trace.size());
      for (auto &tr : trace) {
        retTrace.emplace_back(tr.trace);
      }
    }
    return ret;
  }
};

} // namespace

FailureOr<mlir::gc::memoryplan::ScopeTraceData>
collectMemoryTrace(Operation *root, bool markOnly) {
  TickCollecter collecter;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    collecter.popScopeIfNecessary(op);
    collecter.forwardTick();
    if (auto viewop = dyn_cast<ViewLikeOpInterface>(op)) {
      collecter.onMemrefViews(viewop);
    } else if (op->hasTrait<OpTrait::ReturnLike>()) {
      collecter.onReturnOp(op);
    } else {
      collecter.onGeneralOp(op);
    }
    // finally, if op is complex scope, push one ComplexScope
    if (isa<RegionBranchOpInterface>(op) || isa<LoopLikeOpInterface>(op)) {
      collecter.pushComplexScope(op);
    }
  });
  collecter.popScopeIfNecessary(nullptr);
  if (markOnly) {
    for (auto &[alloc, tick] : collecter.allocTicks) {
      auto allocscope = getAllocScope(alloc);
      alloc->setAttr(
          "__mergealloc_lifetime",
          DenseI64ArrayAttr::get(root->getContext(),
                                 {reinterpret_cast<int64_t>(allocscope),
                                  tick.firstAccess, tick.lastAccess}));
      allocscope->setAttr(
          "__mergealloc_scope",
          IntegerAttr::get(mlir::IntegerType::get(root->getContext(), 64),
                           reinterpret_cast<int64_t>(allocscope)));
    }
    return mlir::gc::memoryplan::ScopeTraceData();
  }
  return collecter.getTrace();
}

} // namespace gc
} // namespace mlir

namespace {
struct MergeAllocPass : mlir::gc::impl::MergeAllocBase<MergeAllocPass> {
  using parent = mlir::gc::impl::MergeAllocBase<MergeAllocPass>;
  void runOnOperation() override {
    auto op = getOperation();
    auto tracesOrFail = mlir::gc::collectMemoryTrace(op, this->optionCheck);
    if (failed(tracesOrFail)) {
      signalPassFailure();
      return;
    }
    if (this->optionCheck) {
      return;
    }
    std::unordered_map<uintptr_t, std::vector<uintptr_t>> dummy;
    for (auto &[scope, traces] : *tracesOrFail) {
      std::unordered_map<uintptr_t, std::size_t> outSchedule;
      if (traces.empty())
        continue;
      auto total = mlir::gc::memoryplan::schedule_memory_allocations(
          traces, 64, !this->optionNoLocality,
          gc::memoryplan::inplace_info_map(), outSchedule, dummy);
      auto &block = scope->getRegion(0).getBlocks().front();
      OpBuilder builder{&block.front()};
      auto alignment =
          builder.getIntegerAttr(IntegerType::get(op.getContext(), 64), 64);
      auto alloc = builder.create<memref::AllocOp>(
          scope->getLoc(),
          MemRefType::get({static_cast<int64_t>(total)}, builder.getI8Type()),
          alignment);
      for (auto &[key, offset] : outSchedule) {
        auto origBuf = reinterpret_cast<Operation *>(key);
        builder.setInsertionPoint(origBuf);
        auto byteShift = builder.create<arith::ConstantIndexOp>(
            origBuf->getLoc(), static_cast<int64_t>(offset));
        auto view = builder.create<memref::ViewOp>(
            origBuf->getLoc(), origBuf->getResultTypes().front(), alloc,
            byteShift, ValueRange{});
        origBuf->replaceAllUsesWith(view->getResults());
        origBuf->remove();
      }
    }
  }

public:
  MergeAllocPass(const mlir::gc::MergeAllocOptions &o) : parent{o} {}
  MergeAllocPass() : parent{mlir::gc::MergeAllocOptions{}} {}
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::gc::createMergeAllocPass() {
  return std::make_unique<MergeAllocPass>();
}
