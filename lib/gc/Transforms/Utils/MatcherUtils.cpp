//===- MatcherUtils.cpp - Matcher utils -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Utils/StructuredOpMatcher.h"
#include "gc/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace structured_match {
namespace utils {

// Return true if all the operand have the same type, i.e., no implicit
// conversion in the linalgOp.
static LogicalResult hasEqualOperandTypes(Operation *operation) {
  if (!isa<linalg::LinalgOp>(operation))
    return failure();
  auto linalgOp = cast<linalg::LinalgOp>(operation);
  OpOperand &outputOperand = linalgOp.getDpsInitsMutable()[0];
  auto elemType = getElementTypeOrSelf(outputOperand.get().getType());

  if (!llvm::all_of(linalgOp.getDpsInitsMutable(), [&](OpOperand &operand) {
        auto currentOperandType = getElementTypeOrSelf(operand.get().getType());
        return currentOperandType == elemType;
      })) {
    return failure();
  }

  if (!llvm::all_of(linalgOp.getDpsInputOperands(), [&](OpOperand *operand) {
        auto currentOperandType =
            getElementTypeOrSelf(operand->get().getType());
        return currentOperandType == elemType;
      })) {
    return failure();
  }
  return success();
}

static bool isTppOp(linalg::LinalgOp linalgOp) {
  // clang-format off
  auto tppMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .output(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasStaticShape())
      .operation(NumRegions(EqualsTo(1)))
      .output(MatchAll(), HasStaticStrides())
      .output(MatchAll(), HasElementType<FloatType>())
      .input(MatchAll(), HasStaticStrides())
      .operation(VerifyOpProperty(hasEqualOperandTypes));
  // clang-format on
  return tppMatcher.match(linalgOp);
}

static bool isTppBinaryOp(linalg::LinalgOp linalgOp) {
  // clang-format off
  auto binaryMatcher =
      StructuredOpMatcher::make<linalg::LinalgOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(1), EqualsTo(2))))
          .output(MatchAll(), HasRank({2}))
          // TODO: (lorenzo) When we introduce broadcast op we
          // will restrict the input to 2d tiles.
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)))
          .output(MatchAll(), HasMap(Identity()))
          .input(MatchAll(), HasMap(BroadcastableProjectedPermutation()));
  // clang-format on
  return isTppOp(linalgOp) && binaryMatcher.match(linalgOp);
}

static bool isTppUnaryOp(linalg::LinalgOp linalgOp) {
  // clang-format off
  auto unaryMatcher =
      StructuredOpMatcher::make<linalg::LinalgOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(0), EqualsTo(1))))
          // TODO: (lorenzo) When we introduce reduce operations
          // we will relax this constraint, and allow SCALAR, 1d
          // and 2d.
          .output(MatchAll(), HasRank({2}))
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)));
  // clang-format on
  return isTppOp(linalgOp) && unaryMatcher.match(linalgOp);
}

template <typename OpTy>
static bool isTwoDEltWiseOpOfTypeTy(linalg::LinalgOp linalgOp,
                                    SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto matcher =
    StructuredOpMatcher::make<linalg::LinalgOp>().region(
      MatchOne(0), WithSingleOp<OpTy>(operands));
  // clang-format on
  return isTppBinaryOp(linalgOp) && matcher.match(linalgOp);
}

bool isTwoDAddOp(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  return isTwoDEltWiseOpOfTypeTy<arith::AddFOp>(linalgOp, operands);
}

static bool hasReluBody(Operation *op, SmallVectorImpl<Value> *captured) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;
  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;
  Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());

  // If lhs is a zero get rhs as input for the relu if it is a block argument,
  // return false otherwise.
  auto getOperand = [&](Value lhs, Value rhs) -> bool {
    if (mlir::utils::isZeroTensor(lhs)) {
      auto blockArg = dyn_cast<BlockArgument>(rhs);
      if (!blockArg || blockArg.getParentBlock() != linalgOp.getBlock())
        return false;
      OpOperand *operand =
          linalgOp.getMatchingOpOperand(cast<BlockArgument>(rhs));
      if (captured) {
        captured->push_back(operand->get());
        captured->push_back(linalgOp.getDpsInitOperand(0)->get());
      }
      return true;
    }
    return false;
  };

  // Multiple patterns map to Relu.
  if (auto maxfOp = dyn_cast<arith::MaximumFOp>(innerOp)) {
    // Pattern 1 - arith.maximumf(tensor, const 0)
    if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
      return false;
    Value maxfLhs = maxfOp.getLhs();
    Value maxfRhs = maxfOp.getRhs();

    return (getOperand(maxfLhs, maxfRhs) || getOperand(maxfRhs, maxfLhs));
  }
  if (auto cmpfOp = dyn_cast<arith::CmpFOp>(innerOp)) {
    // Pattern 2 - arith.cmpf, arith.select
    //
    // x = arith.cmpf ugt, value, 0
    // y = arith.select x, value, 0
    //
    // NOTE: ugt - unsigned greater than, one of the predicates
    if (linalgOp.getBlock()->getOperations().size() != 3)
      return false;

    auto opIterator = linalgOp.getBlock()->getOperations().begin();
    Operation *cmpOp = &*opIterator;
    opIterator++;

    if (!isa<arith::SelectOp>(&*opIterator))
      return false;

    Operation *selectOp = &*opIterator;
    opIterator++;
    if (yieldOp != &*opIterator)
      return false;

    if (yieldOp->getOperand(0).getDefiningOp() != selectOp)
      return false;

    if (selectOp->getOperand(0).getDefiningOp() != cmpOp)
      return false;

    auto cmpPredicate = cmpfOp.getPredicate();
    Value cmpLhs = cmpfOp.getLhs();
    Value cmpRhs = cmpfOp.getRhs();

    auto selOp = cast<arith::SelectOp>(selectOp);
    auto trueVal = selOp.getTrueValue();
    auto falseVal = selOp.getFalseValue();

    if (cmpPredicate == arith::CmpFPredicate::UGT ||
        cmpPredicate == arith::CmpFPredicate::UGE) {
      if (cmpLhs == trueVal && mlir::utils::isZeroTensor(cmpRhs) &&
          mlir::utils::isZeroTensor(falseVal)) {
        // case: %in > 0 ? %in : 0
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
      if (mlir::utils::isZeroTensor(cmpLhs) &&
          mlir::utils::isZeroTensor(trueVal) && cmpRhs == falseVal) {
        // case: 0 > %in ? 0 : %in
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
    } else if (cmpPredicate == arith::CmpFPredicate::ULT ||
               cmpPredicate == arith::CmpFPredicate::ULE) {
      if (cmpLhs == falseVal && mlir::utils::isZeroTensor(cmpRhs) &&
          mlir::utils::isZeroTensor(trueVal)) {
        // case: %in < 0 ? 0 : %in
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
      if (mlir::utils::isZeroTensor(cmpLhs) &&
          mlir::utils::isZeroTensor(falseVal) && cmpRhs == trueVal) {
        // case: 0 < %in ? %in : 0
        return (getOperand(cmpLhs, cmpRhs) || getOperand(cmpRhs, cmpLhs));
      }
    }
  }
  return false;
}

namespace {
// Helper matcher functor for relu detection.
struct WithReluBody {
  WithReluBody() = delete;
  WithReluBody(SmallVectorImpl<Value> *captures) : captures(captures){};

  bool operator()(Region *region, Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return false;

    return hasReluBody(linalgOp, captures);
  }

private:
  SmallVectorImpl<Value> *captures;
};
} // namespace

// Return true if the linalg.generic can be mapped to a tpp.relu.
bool isTwoDReluOp(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto reluMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .output(MatchAll(), HasMap(Identity()))
    .input(MatchAll(), HasMap(BroadcastableProjectedPermutation()))
    .region(MatchOne(0), WithReluBody(operands));
  // clang-format on
  return isTppUnaryOp(linalgOp) && reluMatcher.match(linalgOp);
}

} // namespace utils
} // namespace structured_match
} // namespace mlir
