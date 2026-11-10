//===-- Transform.h - Transformation untils ----------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORM_H
#define GC_TRANSFORM_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/XeGPU/uArch/uArchCommon.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "gc/IR/Attributes.h"

namespace mlir::gc {

// This class is a placeholder for the rewriter-related boilerplate code.
struct OpRewriter final : IRRewriter {
  Location loc;

  explicit OpRewriter(func::FuncOp &func)
      : IRRewriter(func.getContext()), loc(func.getLoc()) {}

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    return OpTy::create(*this, loc, std::forward<Args>(args)...);
  }

  arith::ConstantIndexOp createConstant(int64_t v) {
    return create<arith::ConstantIndexOp>(v);
  }

  arith::ConstantFloatOp createConstant(double v) {
    return create<arith::ConstantFloatOp>(getF64Type(), APFloat(v));
  }
};

struct TruePredicate {
  template <typename T> constexpr bool operator()(T &&) const { return true; }
};
template <typename Predicate = TruePredicate> struct ReverseIterator {

  static auto makeIterable(Block &block) {
    auto reversed = llvm::reverse(ForwardIterator::makeIterable(block));
    if constexpr (std::is_same_v<Predicate, TruePredicate>) return reversed;
    else return llvm::make_filter_range(reversed, Predicate{});
  }

  template <typename T> static auto makeIterable(T &range) {
    return llvm::reverse(ForwardIterator::makeIterable(range));
  }
};

template <typename T = Operation *, typename Predicate = TruePredicate>
T findLast(Operation *root, std::function<bool(T)> predicate) {
  T last = nullptr;
  root->walk<WalkOrder::PreOrder, ReverseIterator<Predicate>>([&](T op) {
    if (predicate(op)) {
      last = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return last;
}
// ---------------------------------------------------------- //

// Check recursively if the specified operation has an operand that
// depends on a result of a previous operation, matching the predicate.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOperandDependsOnOp(std::function<bool(Operation *)> predicate,
                          Operation *operation, unsigned depth = 0) {
  for (auto operand : operation->getOperands()) {
    if (auto op = operand.getDefiningOp();
        op &&
        (predicate(op) || (depth < MaxDepth &&
                           isOperandDependsOnOp(predicate, op, depth + 1)))) {
      return true;
    }
  }
  return false;
}

// Check recursively if there are any operation, matching the predicate, that
// depends on the result of the specified operation.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOpDependsOnResult(std::function<bool(Operation *)> predicate,
                         Operation *operation, unsigned depth = 0) {
  for (auto res : operation->getResults()) {
    for (auto u : res.getUsers()) {
      if (predicate(u) ||
          (depth < MaxDepth && isOpDependsOnResult(predicate, u, depth + 1))) {
        return true;
      }
    }
  }
  return false;
}

inline bool isMatmulOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && linalg::isaContractionOpInterface(linalgOp);
  // TODO: Check matmul like generics
}

inline bool containsMatmulOp(Operation *op) {
  if (isMatmulOp(op)) return true;
  bool found = false;
  op->walk([&](Operation *nested) {
    if (nested != op && isMatmulOp(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// If a slice inside the loop is created from an external empty tensor and
// the tensor is not passed to the loop's shared_outs, but referenced
// directly, replace the slice with an empty tensor of the same size.
inline void replaceEmptySlices(OpRewriter &rw, LoopLikeOpInterface loop) {
  loop.walk([&](tensor::ExtractSliceOp slice) {
    if (auto empty = slice.getSource().getDefiningOp<tensor::EmptyOp>();
        empty && empty->getParentOfType<LoopLikeOpInterface>() != loop) {
      auto type = slice.getType();
      rw.setInsertionPointAfter(slice);
      SmallVector<Value> dynDims;
      for (int64_t i = 0, r = type.getRank(); i < r; ++i) {
        if (type.isDynamicDim(i)) {
          dynDims.push_back(rw.create<tensor::DimOp>(slice, i));
        }
      }
      rw.replaceOp(slice, rw.create<tensor::EmptyOp>(
                              type.getShape(), type.getElementType(), dynDims));
    }
  });
}

inline scf::ForallOp getKernelLoop(Operation *op) {
  for (; op; op = op->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForallOp>(op);
        loop && loop->hasAttr(GC_ATTR_KERNEL_NAME))
      return loop;
  }
  return nullptr;
}

inline void canonicalizeLoop(LoopLikeOpInterface &loop) {
  auto parent = loop->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  assert(parent);
  auto ctx = parent->getContext();
  RewritePatternSet patterns(ctx);
  if (isa<scf::ForallOp>(loop.getOperation())) {
    scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
  } else if (isa<scf::ForOp>(loop.getOperation())) {
    scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  }

  static size_t stamp = 0;
  auto st = ++stamp;
  // The loop's operation can be replaced by the patterns. Using a stamp
  // to find it again.
  loop->setDiscardableAttr(GC_ATTR_LOOP_STAMP, createAttr(ctx, st));
  if (failed(applyPatternsGreedily(parent, std::move(patterns)))) {
    loop->emitWarning() << "Loop canonicalization failed";
  }
  parent->walk([&](LoopLikeOpInterface o) {
    if (getDiscardableAttr<size_t>(o, GC_ATTR_LOOP_STAMP, 0) == st) {
      o->removeDiscardableAttr(GC_ATTR_LOOP_STAMP);
      loop = o;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

// `result` is an OpResult produced by some srcOp.
// `consumer` is a consumer of this result.
// Check if the `consumer` has any other input, that depend on `srcOp` through a
// different path.
inline bool hasDiamondDep(OpResult result, Operation *consumer) {
  auto srcOp = result.getDefiningOp();
  SmallVector<Operation *> stack = llvm::filter_to_vector(
      llvm::map_range(consumer->getOperands(),
                      [&](Value v) { return v.getDefiningOp(); }),
      [&](Operation *o) { return o && o != srcOp; });
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    if (!visited.insert(op).second) continue;
    if (op == srcOp) return true;
    for (auto operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) stack.push_back(defOp);
    }
  }
  return false;
}

// Check if all subgraphs from the `result` have a common intersection.
template <typename Predicate>
bool allUsersIntersect(OpResult result, Predicate predicate) {
  SmallVector<Value::user_range> stack = {result.getUsers()};
  if (stack.size() <= 1) return true;
  Operation *intersection = nullptr;
  llvm::SmallSet<Operation *, 32> visited;
  while (!stack.empty()) {
    auto range = stack.pop_back_val();
    llvm::SmallSet<Operation *, 8> unique;
    for (auto op : range) unique.insert(op);
    for (auto op : unique) {
      if (!visited.insert(op).second) {
        if (intersection == nullptr) intersection = op;
        else if (intersection != op) return false;
        continue;
      }
      if (op->hasTrait<OpTrait::ReturnLike>()) continue;
      if (!predicate(op)) return false;
      for (auto result : op->getResults()) stack.push_back(result.getUsers());
    }
  }
  return true;
}

// Find ParallelInsertSliceOp corresponding to the specified loop result.
inline tensor::ParallelInsertSliceOp
findParallelInsertSlice(scf::ForallOp &forall, OpResult loopRes) {
  BlockArgument outArg = forall.getTiedBlockArgument(loopRes);
  tensor::ParallelInsertSliceOp ins;
  for (auto &op : forall.getTerminator().getYieldingOps())
    if (auto i = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
        i && i.getDest() == outArg) {
      ins = i;
      break;
    }
  return ins;
}

} // namespace mlir::gc
#endif // GC_TRANSFORM_H
