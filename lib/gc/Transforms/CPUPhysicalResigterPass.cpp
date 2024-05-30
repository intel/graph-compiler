//===- CPUPhysicalResigterPass.cpp.cpp - OneDNNGraph To Linalg
// Lowering -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/Casting.h"
#include <deque>
#include <iostream>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CPUPHYSICALREGISTERPASS
#include "gc/Transforms/Passes.h.inc"
namespace {
#define DEBUG_TYPE "lower-to-physical-register-pass"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

struct HardwareInfo {
  bool favx512f = true;
  bool favx2 = true;
} HW;

bool isSpecialOp(Operation *op) {
  return llvm::isa<vector::TransposeOp>(op) ||
         llvm::isa<vector::BroadcastOp>(op) ||
         llvm::isa<vector::ReductionOp>(op) ||
         llvm::isa<vector::ShapeCastOp>(op) ||
         llvm::isa<vector::MultiDimReductionOp>(op);
}

bool is_innermost_operation(Operation *op) {
  bool inner_most = true;
  op->walk([&inner_most](Operation *p) {
    if (llvm::isa<scf::ForOp>(p)) {
      inner_most = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return inner_most;
}

int generateValidSteps(int steps, VectorType type) {
  return type.getShape().back() >= steps ? steps : 1;
}

// Get the maximum number of current data types that a register can hold
[[nodiscard]] int getDataTypeMAXSIMDLength(VectorType type) {
  auto typebits = type.getElementTypeBitWidth();
  const int favx512bits = 512;
  const int favx2bits = 256;
  if (HW.favx512f) {
    return generateValidSteps(favx512bits / typebits, type);
  } else if (HW.favx2) {
    return generateValidSteps(favx2bits / typebits, type);
  } else {
    // invalid
    LDBG("Please check the hardware information.");
    assert(false && "Invalid hardware.");
    return -1;
  }
}

mlir::FailureOr<VectorType> getOperationVectorType(Operation *op) {
  return TypeSwitch<Operation *, mlir::FailureOr<VectorType>>(op)
      .Case<vector::TransferWriteOp>(
          [&](vector::TransferWriteOp transferWriteOp)
              -> mlir::FailureOr<VectorType> {
            return transferWriteOp.getVectorType();
          })
      .Case<vector::TransferReadOp>([&](vector::TransferReadOp transferReadOp)
                                        -> mlir::FailureOr<VectorType> {
        return transferReadOp.getVectorType();
      })
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constantOp) { return failure(); })
      .Default([&](Operation *op) -> mlir::FailureOr<VectorType> {
        if (!op->getResults().empty()) {
          auto t = op->getResultTypes().front().dyn_cast<VectorType>();
          if (t) {
            return t;
          }
        }
        return failure();
      });
}

// Filter out the operations that can be vectorized. We are only interested in
// operations that do not contain any for loops(innermost IR).
[[nodiscard]] bool filterOperation(Operation *op) {
  if (!is_innermost_operation(op)) {
    LDBG("Operation is not innermost" << *op << "\n");
    return false;
  }

  // We are only interested about the operation in vector dialect
  if (failed(getOperationVectorType(op))) {
    LDBG("Operation is not in vector dialect" << *op << "\n");
    return false;
  }
  return true;
}

// Since we rewrote transfer_read and transfer_write, the `permutationmap` must
// be changed.
void setOpVectorizationPermutationMap(Operation *op, IRRewriter &rewriter,
                                      RankedTensorType tensorType) {
  SmallVector<AffineExpr, 1> affineExprs;
  affineExprs.push_back(rewriter.getAffineDimExpr(tensorType.getRank() - 1));
  auto destAffineMap = AffineMap::get(tensorType.getRank(), 0, affineExprs,
                                      rewriter.getContext());
  SmallVector<bool> inBounds(1, true);
  if (mlir::isa<vector::TransferWriteOp>(op)) {
    auto transferWriteOp = mlir::dyn_cast<vector::TransferWriteOp>(op);
    transferWriteOp.setPermutationMap(destAffineMap);
    transferWriteOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  } else if (mlir::isa<vector::TransferReadOp>(op)) {
    auto transferReadOp = mlir::dyn_cast<vector::TransferReadOp>(op);
    transferReadOp.setPermutationMap(destAffineMap);
    transferReadOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  }
}

// scf.for yield helper function
void maybeYieldValue(OpBuilder &b, Location loc, ValueRange value) {
  bool hasRetVal = !value.empty();
  if (hasRetVal) {
    b.create<scf::YieldOp>(loc, value);
  } else {
    b.create<scf::YieldOp>(loc);
  }
}

//
void checkAndSetOperand(
    Operation *op, const ValueRange &iterArgs,
    const llvm::DenseMap<Value, int> &operandIdxMap,
    const llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
    if (operandIdxMap.contains(opd)) {
      op->setOperand(idx, iterArgs[operandIdxMap.at(opd)]);
    }
  }
  int offset = isa<vector::TransferWriteOp>(op) ? 2 : 1;
  if (llvm::dyn_cast<vector::TransferWriteOp>(op) ||
      llvm::dyn_cast<vector::TransferReadOp>(op)) {
    assert(opPermuationMap.contains(op));
    op->dump();
    std::cout << inductionVars.size() << std::endl;

    auto permutationMap = opPermuationMap.at(op);

    auto dimExpr = permutationMap.getResults();
    for (auto [idx, x] : llvm::enumerate(dimExpr)) {
      if (mlir::dyn_cast<AffineDimExpr>(x)) {
        auto dim = mlir::dyn_cast<AffineDimExpr>(x).getPosition();
        op->setOperand(dim + offset, inductionVars[dim]);
      }
    }
  }
}

// TODO: need to rewrite reduce operation as a performance forms like
// graph-compiler v1
scf::ForOp constructReductionNestedForOp(
    OpBuilder &b, const Location &loc, const ValueRange &iterArgs,
    const VectorType &type, const llvm::ArrayRef<int64_t> &dims, size_t idx,
    std::queue<Operation *> &queue, const llvm::SetVector<Value> &resultSet,
    llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Value, int> &operandIdxMap,
    const llvm::SmallVector<int64_t, 5> &rdDims,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  const int loop_step = getDataTypeMAXSIMDLength(type);

  // loop initialization variable
  auto zero =
      b.create<arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(),
                                  b.getIntegerAttr(b.getIndexType(), 0));
  auto forSteps = b.create<arith::ConstantOp>(
      b.getUnknownLoc(), b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(),
                       idx == dims.size() - 1 ? loop_step : 1));
  auto numIter = b.create<arith::ConstantOp>(
      b.getUnknownLoc(), b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(), dims[idx]));

  // Create a loop and move vectorized operation into loops.
  auto forOp = b.create<scf::ForOp>(
      b.getUnknownLoc(), zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (idx == dims.size() - 1) {
          Operation *lastOperation = queue.front();
          while (!queue.empty()) {
            auto x = queue.front();
            queue.pop();
            if (lastOperation == x) {
              x->moveBefore(b.getBlock(), b.getBlock()->begin());
            } else {
              x->moveAfter(lastOperation);
              lastOperation = x;
            }
            // check operation type to set correct operand
            checkAndSetOperand(x, loopState, operandIdxMap, inductionVars,
                               opPermuationMap);
          }
          maybeYieldValue(b, loc, resultSet.getArrayRef());
        } else {

          // outter loop
          auto nxtFor = constructReductionNestedForOp(
              b, loc, loopState, type, dims, idx + 1, queue, resultSet,
              inductionVars, operandIdxMap, rdDims, opPermuationMap);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
  return forOp;
}

scf::ForOp constructNestedForOp(
    OpBuilder &b, const Location &loc, const ValueRange &iterArgs,
    const VectorType &type, const llvm::ArrayRef<int64_t> &dims, size_t idx,
    std::queue<Operation *> &queue, const llvm::SetVector<Value> &resultSet,
    llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Value, int> &operandIdxMap,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  const int loop_step = getDataTypeMAXSIMDLength(type);

  // loop initialization variable
  auto zero =
      b.create<arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(),
                                  b.getIntegerAttr(b.getIndexType(), 0));
  auto forSteps = b.create<arith::ConstantOp>(
      b.getUnknownLoc(), b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(),
                       idx == dims.size() - 1 ? loop_step : 1));
  auto numIter = b.create<arith::ConstantOp>(
      b.getUnknownLoc(), b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(), dims[idx]));

  // Create a loop and move vectorized operation into loops.
  auto forOp = b.create<scf::ForOp>(
      b.getUnknownLoc(), zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (idx == dims.size() - 1) {
          Operation *lastOperation = queue.front();
          while (!queue.empty()) {
            auto x = queue.front();
            queue.pop();
            if (lastOperation == x) {
              x->moveBefore(b.getBlock(), b.getBlock()->begin());
            } else {
              x->moveAfter(lastOperation);
              lastOperation = x;
            }
            // check operation type to set correct operand
            checkAndSetOperand(x, loopState, operandIdxMap, inductionVars,
                               opPermuationMap);
          }
          maybeYieldValue(b, loc, resultSet.getArrayRef());
        } else {

          // outter loop
          auto nxtFor = constructNestedForOp(
              b, loc, loopState, type, dims, idx + 1, queue, resultSet,
              inductionVars, operandIdxMap, opPermuationMap);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
  return forOp;
}

bool isCompatibleVectorType(Operation *op1, Operation *op2) {
  auto type1 = getOperationVectorType(op1);
  auto type2 = getOperationVectorType(op2);
  if (failed(type1) || failed(type2)) {
    return false;
  }
  auto sp1 = type1.value();
  auto sp2 = type2.value();
  auto min_rank = std::min(sp1.getRank(), sp2.getRank()) - 1;
  for (auto i = min_rank; i >= 0; i--) {
    if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
      return false;
    }
  }

  return true;
}

bool isNeedNewGroup(llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
                    Operation *op) {
  // 1. check previous operation
  if (!opGroups.back().empty()) {
    auto prevOp = opGroups.back().back();
    // not in the same operation
    if (prevOp->getParentOp() != op->getParentOp()) {
      return true;
    }
    // previous operation is a special operation
    if (isSpecialOp(prevOp)) {
      return true;
    }
    // previous operation vector type is not compatible with current operation
    if (!isCompatibleVectorType(prevOp, op)) {
      return true;
    }
  }

  // 2. check current operation
  if (isSpecialOp(op)) {
    return true;
  }
  return false;
}

void addOperationToGroup(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap, Operation *op,
    llvm::SmallVector<VectorType, 8> &groupsShapes) {
  //
  if (isNeedNewGroup(opGroups, op)) {
    opGroups.emplace_back(std::queue<Operation *>());
  }
  if (opGroups.size() != groupsShapes.size()) {
    groupsShapes.emplace_back(getOperationVectorType(op).value());
  }
  opGroups.back().push(op);
  opGroupIndexMap[op] = opGroups.size() - 1;
}

// We classify the operations we are interested in after filtering. Operations
// of in the same group have no data dependencies. Those operations can generate
// a same outter for loop.
void classifyOperations(func::FuncOp func,
                        llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
                        llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
                        llvm::SmallVector<VectorType, 8> &groupsShapes) {
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *>(op).Default([&](Operation *op) {
      if (filterOperation(op)) {
        addOperationToGroup(opGroups, opGroupIndexMap, op, groupsShapes);
      }
    });
  });
}

Value setOutGroupOperationOperandResult(Operation *op,
                                        const VectorType &newOperandType) {
  auto ret = TypeSwitch<Operation *, Value>(op)
                 .Case<arith::ConstantOp>([&](arith::ConstantOp constantOp) {
                   IRRewriter rewriter(op);
                   rewriter.setInsertionPointAfter(op);
                   Type resultElementType = newOperandType.getElementType();

                   Attribute initValueAttr;
                   if (isa<FloatType>(resultElementType))
                     initValueAttr = FloatAttr::get(resultElementType, 0.0);
                   else
                     initValueAttr = IntegerAttr::get(resultElementType, 0);
                   auto cntOp = rewriter.create<arith::ConstantOp>(
                       rewriter.getUnknownLoc(),
                       DenseElementsAttr::get(newOperandType, {initValueAttr}));
                   return cntOp->getResults()[0];
                 })
                 .Default([&](Operation *op) { return Value(); });
  return ret;
}

void setOperationOperandResult(
    Operation *op, const VectorType &newOperandType,
    const llvm::DenseMap<Operation *, size_t> &opMap) {
  for (auto [idx, x] : llvm::enumerate(op->getOperands())) {
    if (x.getType().dyn_cast<VectorType>()) {
      if (!opMap.contains(x.getDefiningOp())) {
        auto result = setOutGroupOperationOperandResult(x.getDefiningOp(),
                                                        newOperandType);
        op->setOperand(idx, result);
      } else {
        x.setType(newOperandType);
      }
    }
  }
  for (auto x : op->getResults()) {
    if (x.getType().dyn_cast<VectorType>()) {
      x.setType(newOperandType);
    }
  }
};

/// Rewrite the operations in the group to vectorized form.
void rewriteOperationAsVectorize(
    const std::queue<Operation *> &groupOps,
    llvm::DenseMap<Operation *, size_t> &opMap, IRRewriter &rewriter,
    llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  std::queue<Operation *> transformQueue(groupOps);

  auto getVectorzedType = [](Operation *op) -> VectorType {
    // Check that the operation type can be broken
    // down into a loop.
    auto baseType = getOperationVectorType(op);
    if (failed(baseType)) {
      LDBG("Failed to get vector type for operation: " << *op << "\n");
      assert(false && "Failed to get vector type for operation");
      return VectorType();
    }
    auto vectorizedType = baseType.value();
    const int loop_step = getDataTypeMAXSIMDLength(vectorizedType);
    return VectorType::get({loop_step}, vectorizedType.getElementType());
  };

  while (!transformQueue.empty()) {
    auto op = transformQueue.front();
    transformQueue.pop();
    auto lowerResult =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<vector::TransferWriteOp>(
                [&](vector::TransferWriteOp transferWriteOp) {
                  auto newOperandType = getVectorzedType(transferWriteOp);
                  if (!isSpecialOp(
                          transferWriteOp->getOperand(0).getDefiningOp())) {
                    opPermuationMap.insert(
                        {transferWriteOp, transferWriteOp.getPermutationMap()});
                    transferWriteOp->getOperand(0).setType(newOperandType);
                    setOpVectorizationPermutationMap(
                        transferWriteOp, rewriter,
                        transferWriteOp->getResult(0)
                            .getType()
                            .dyn_cast<RankedTensorType>());
                  }

                  return success();
                })
            .Case<vector::TransferReadOp>(
                [&](vector::TransferReadOp transferReadOp) {
                  auto newOperandType = getVectorzedType(transferReadOp);
                  auto users = transferReadOp->getUsers();
                  bool isUserSpecial = false;
                  for (auto *opUse : users) {
                    if (isSpecialOp(opUse)) {
                      isUserSpecial = true;
                      break;
                    }
                  }
                  if (!isUserSpecial) {
                    opPermuationMap.insert(
                        {transferReadOp, transferReadOp.getPermutationMap()});
                    transferReadOp->getResult(0).setType(newOperandType);
                    setOpVectorizationPermutationMap(
                        transferReadOp, rewriter,
                        transferReadOp.getSource()
                            .getType()
                            .dyn_cast<RankedTensorType>());
                  }

                  return success();
                })
            .Case<vector::MultiDimReductionOp>(
                [&](vector::MultiDimReductionOp multiReductionOp) {
                  return success();
                })
            .Default([&](Operation *op) {
              if (isSpecialOp(op)) {
                return success();
              }
              setOperationOperandResult(op, getVectorzedType(op), opMap);
              return success();
            });
    if (failed(lowerResult)) {
      LDBG("Failed to rewrite operation: " << *op << "\n");
      assert(false && "Failed to rewrite operation");
    }
  }
}

// analysis operation' operands are coming from which operation's result
void analysisOperaionOperandSource(
    size_t idx, std::queue<Operation *> &grp,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOperandNeedSet) {
  auto tmpOpQueue(grp);
  llvm::SetVector<Value> opOperands;
  while (!tmpOpQueue.empty()) {
    auto t = tmpOpQueue.front();
    for (auto x : t->getOperands()) {
      // not in the same group
      if (opGroupIndexMap.contains(x.getDefiningOp()) &&
          opGroupIndexMap[x.getDefiningOp()] != idx) {
        groupOperandNeedSet[idx].insert(x);
      } else {
        groupOperandNeedSet[idx].insert(x);
      }
    }
    tmpOpQueue.pop();
  }
}

Operation *createTensorEmptyBefore(Operation *op) {
  auto rtType = op->getResultTypes()[0].dyn_cast<ShapedType>();
  IRRewriter reWriter(op);

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i))
      dynDims.push_back(reWriter.create<tensor::DimOp>(reWriter.getUnknownLoc(),
                                                       op->getResult(0), i));
  }
  return reWriter.create<tensor::EmptyOp>(op->getLoc(), rtType.getShape(),
                                          rtType.getElementType(), dynDims);
}

Operation *
createTransferReadOpBefore(Operation *op, const Value &operand,
                           vector::TransferReadOp *srcReadOp = nullptr) {
  auto operandType = operand.getType().dyn_cast<ShapedType>();

  IRRewriter rewriter(op);
  auto zero =
      rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
  auto padValue = rewriter.create<arith::ConstantOp>(
      rewriter.getUnknownLoc(),
      rewriter.getZeroAttr(operandType.getElementType()));

  if (srcReadOp) {
    auto resultType = srcReadOp->getType().dyn_cast<ShapedType>();
    SmallVector<bool> inBoundsVal(resultType.getRank(), true);
    auto srcReadOpAffineMap = srcReadOp->getPermutationMap();
    // result of read operation should be same as operand
    auto t = rewriter.create<vector::TransferReadOp>(
        rewriter.getUnknownLoc(),
        /*vectorType=*/
        VectorType::get(resultType.getShape(), resultType.getElementType()),
        /*source=*/operand,
        /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
        /**affinemap*/ srcReadOpAffineMap,
        /*inBounds=*/inBoundsVal);

    return t;
  } else {
    SmallVector<bool> inBoundsVal(operandType.getRank(), true);
    auto t = rewriter.create<vector::TransferReadOp>(
        rewriter.getUnknownLoc(),
        /*vectorType=*/
        VectorType::get(operandType.getShape(), operandType.getElementType()),
        /*source=*/operand,
        /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
        /**affinemap*/ padValue,
        /*inBounds=*/inBoundsVal);
    return t;
  }
}

Operation *createTransferWriteOpAfter(Operation *op, const Value &dest) {
  auto rtType = op->getResultTypes()[0].dyn_cast<ShapedType>();
  auto rank = rtType.getRank();
  auto dstType = dest.getType().dyn_cast<ShapedType>();
  IRRewriter reWriter(op);

  auto zero =
      reWriter.create<arith::ConstantIndexOp>(reWriter.getUnknownLoc(), 0);

  reWriter.setInsertionPointAfter(op);
  SmallVector<bool> inBoundsVal(rank, true);

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i)) {
      dynDims.push_back(reWriter.create<tensor::DimOp>(reWriter.getUnknownLoc(),
                                                       op->getResult(0), i));
    }
  }
  return reWriter.create<vector::TransferWriteOp>(
      reWriter.getUnknownLoc(),
      /*vector=*/op->getResult(0),
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(dstType.getRank(), zero),
      /*inBounds=*/inBoundsVal);
}

// canonicalizing operation as tensor empty and transfer write the operation
// result into the empty tensor
[[nodiscard]] std::pair<Value, Value>
canonicalizeSourceOperation(Operation *op) {
  auto emtpyOp = createTensorEmptyBefore(op);
  auto writeOp = createTransferWriteOpAfter(op, emtpyOp->getResults()[0]);
  return std::make_pair(emtpyOp->getResults()[0], writeOp->getResults()[0]);
}

[[nodiscard]] Value
canonicalizeCurrentOperation(Operation *op, const Value &transferReadOperand,
                             size_t operandIdx,
                             vector::TransferReadOp *srcReadOp = nullptr) {
  // transfer_read operation
  auto readOp = createTransferReadOpBefore(op, transferReadOperand, srcReadOp);
  op->setOperand(operandIdx, readOp->getResults()[0]);
  return readOp->getResults()[0];
}

mlir::FailureOr<Value> getOperationDestnationOperand(Operation *op) {
  return llvm::TypeSwitch<Operation *, mlir::FailureOr<Value>>(op)
      .Case<vector::TransferWriteOp>(
          [&](vector::TransferWriteOp transferWriteOp) {
            LDBG(" DPS operation : " << *op << "\n");
            return transferWriteOp->getOperand(1);
          })
      .Case<vector::TransferReadOp>([&](vector::TransferReadOp transferReadOp) {
        LDBG(" DPS operation : " << *op << "\n");
        return transferReadOp->getOperand(0);
      })
      .Default([&](Operation *op) {
        LDBG("Try to get not DPS operation inits: " << *op << "\n");
        return failure();
      });
}

// analysis operations of current group need which operation's result value
void analysisGroupOperationOperands(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOperandNeedSet) {

  for (auto [idx, grp] : enumerate(opGroups)) {
    analysisOperaionOperandSource(idx, grp, opGroupIndexMap,
                                  groupOperandNeedSet);
  }
}

// TODO: need to rewrite reduce
// llvm::SmallVector<int64_t, 5> &
// getReductionDims(vector::MultiDimReductionOp &reductionOp,
//                  llvm::SmallVector<int64_t, 5> &rdDims) {
//   auto rdDimsAttr = reductionOp.getReductionDims().getValue();
//   for (auto x : rdDimsAttr) {
//     rdDims.emplace_back(x.cast<IntegerAttr>().getInt());
//   }
//   return rdDims;
// }

void updateOpOperandResultInGroups(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap, size_t opGid,
    Operation *op, Value &init, const Value &result = Value()) {
  auto tmpOpQueue(opGroups[opGid]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    if (curOp == op) {
      if (!failed(getOperationVectorType(init.getDefiningOp()))) {
        newOpQueue.push(init.getDefiningOp());
        opGroupIndexMap[init.getDefiningOp()] = opGid;
      }

      newOpQueue.push(op);

      if (result && !failed(getOperationVectorType(result.getDefiningOp()))) {
        newOpQueue.push(result.getDefiningOp());
        opGroupIndexMap[result.getDefiningOp()] = opGid;
      }
    } else {
      newOpQueue.push(curOp);
    }
  }
  opGroups[opGid] = newOpQueue;
}

// analysis operation result of current group whether needed by other
// operation which out of current group
void analysisGroupOperationResults(
    func::FuncOp &func, llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    IRMapping &mapOpResultToYield,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupResultYeildSet,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpDestination) {
  llvm::DenseMap<Operation *, std::pair<Value, Value>> srcOpCanoniclizedMap;
  IRRewriter rewriter(func);
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
      auto sourceOp = opd.getDefiningOp();
      if (opGroupIndexMap.contains(sourceOp)) {
        auto sourceOpGid = opGroupIndexMap[sourceOp];
        //
        bool notInSameGroup =
            opGroupIndexMap.contains(op) && sourceOpGid != opGroupIndexMap[op];
        bool outOfGroup = !opGroupIndexMap.contains(op);
        if (notInSameGroup or outOfGroup) {
          // update init iterargs
          auto dstRet = getOperationDestnationOperand(sourceOp);
          if (failed(dstRet)) {
            if (!srcOpCanoniclizedMap.contains(sourceOp)) {
              auto [init, result] = canonicalizeSourceOperation(sourceOp);
              srcOpCanoniclizedMap.insert({sourceOp, {init, result}});
              updateOpOperandResultInGroups(opGroups, opGroupIndexMap,
                                            sourceOpGid, sourceOp, init,
                                            result);
              groupOpDestination[sourceOpGid].insert(init);
              groupResultYeildSet[sourceOpGid].insert(result);
              mapOpResultToYield.map(result, result);
            }

            auto opInit = canonicalizeCurrentOperation(
                op, srcOpCanoniclizedMap[sourceOp].second, idx);
            updateOpOperandResultInGroups(opGroups, opGroupIndexMap,
                                          opGroupIndexMap[op], op, opInit);

          } else {
            if (mlir::isa<vector::TransferReadOp>(sourceOp)) {
              auto transferReadOp =
                  mlir::dyn_cast<vector::TransferReadOp>(sourceOp);
              auto opInit = canonicalizeCurrentOperation(op, dstRet.value(),
                                                         idx, &transferReadOp);
              updateOpOperandResultInGroups(opGroups, opGroupIndexMap,
                                            opGroupIndexMap[op], op, opInit);

            } else {
              groupOpDestination[sourceOpGid].insert(dstRet.value());
              groupResultYeildSet[sourceOpGid].insert(opd);

              // just map to it self,  placeholder
              mapOpResultToYield.map(opd, opd);
            }
          }
        }
      }
    }
  });
  // If the group operations do not have result need to be returned, these are
  // useless code.
  for (auto [idx, grp] : enumerate(opGroups)) {
    if (groupResultYeildSet[idx].empty()) {
      std::queue<Operation *>().swap(grp);
    }
  }
  LDBG("Complete analysis group operation results\n");
}

void analysisGroupOperaionOperandsResults(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOperandNeedSet,
    func::FuncOp &func,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupResultYeildSet,
    IRMapping &mapOpResultToYield,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpDestination) {
  // Operands
  analysisGroupOperationOperands(opGroups, opGroupIndexMap,
                                 groupOperandNeedSet);
  // Results
  analysisGroupOperationResults(func, opGroups, mapOpResultToYield,
                                opGroupIndexMap, groupResultYeildSet,
                                groupOpDestination);
}

mlir::FailureOr<scf::ForOp> generateVectorizedForLoop(
    IRRewriter &rewriter, const llvm::SetVector<Value> &resultSet,
    const llvm::SetVector<Value> &dstOperandSet, const VectorType &vectorType,
    std::queue<Operation *> &queue,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  assert(!resultSet.empty() && "Expected non-empty value");
  // prepare for loop iterargs
  llvm::SmallVector<Value, 4> operands;
  llvm::DenseMap<Value, int> operandIdxMap;
  for (auto [idx, x] : llvm::enumerate(dstOperandSet)) {
    operands.emplace_back(x);
    operandIdxMap[x] = operands.size() - 1;
  }
  ValueRange iterArgs(operands);
  auto shapes = vectorType.getShape();
  llvm::SmallVector<Value, 5> inductionVars;
  // TODO: special operation process
  bool isOpSpecial = false;
  std::queue<Operation *> tmpQ(queue);
  // temporary for special operation generation
  while (!tmpQ.empty()) {
    if (isSpecialOp(tmpQ.front())) {
      isOpSpecial = true;
      break;
    }
    tmpQ.pop();
  }
  if (isOpSpecial) {
    return failure();
  }
  // generate for loop
  auto forOp = constructNestedForOp(
      rewriter, rewriter.getUnknownLoc(), iterArgs, vectorType, shapes, 0,
      queue, resultSet, inductionVars, operandIdxMap, opPermuationMap);
  return forOp;
}

void updateLoopResultUses(
    const size_t groupIdx,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupResultYeildSet,
    scf::ForOp *forOp) {
  if (groupResultYeildSet[groupIdx].empty()) {
    return;
  }
  IRRewriter rewriter(*forOp);
  OpBuilder::InsertionGuard g(rewriter);
  // Only different group operation operand need to be replaced due to same
  // group operation should directly use original operand.
  auto producerOp = groupResultYeildSet[groupIdx].front().getDefiningOp();
  auto needToReplaced = [&](OpOperand &operand) {
    return producerOp->getBlock() != operand.getOwner()->getBlock();
  };
  // update loop result uses
  for (auto [retIdx, rt] : llvm::enumerate(groupResultYeildSet[groupIdx])) {
    producerOp = rt.getDefiningOp();
    rewriter.replaceUsesWithIf(rt, forOp->getResult(retIdx), needToReplaced);
  }
}

void generateGroupOpVectorizedIR(
    std::queue<Operation *> &grp, const size_t idx,
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
    llvm::SmallVector<VectorType, 8> &groupsShapes,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupResultYeildSet,
    llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOpDestination,
    IRMapping &mapOpResultToYield, func::FuncOp &func,
    llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  if (grp.empty()) {
    LDBG("Current operation Group is empty.");
    return;
  }
  IRRewriter rewriter(grp.back());
  rewriter.setInsertionPointAfter(grp.back());
  // 1. Rewrite operation as vectorized form
  rewriteOperationAsVectorize(opGroups[idx], opGroupIndexMap, rewriter,
                              opPermuationMap);
  // 2. Generate loop
  auto forOp = generateVectorizedForLoop(
      rewriter, groupResultYeildSet[idx], groupOpDestination[idx],
      groupsShapes[idx], opGroups[idx], opPermuationMap);
  // special operation do not need to change anything
  if (failed(forOp)) {
    return;
  }
  // 3 Update loop result uses
  updateLoopResultUses(idx, groupResultYeildSet, &forOp.value());
  moveLoopInvariantCode(forOp.value());
}

/// Pass that lower to physical vector.
struct CPUPhysicalRegisterPass
    : public impl::CPUPhysicalRegisterPassBase<CPUPhysicalRegisterPass> {

  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    // 1. Classify operaions:
    // classify the operations into :
    //    a. reorder, transpose. Reorder(or transpose) dim may bring data
    //    dependency.
    //    b. elemenwise. Those operations can be fused into a common for loop.
    //    c. broadcast. Need to analysis broadcast dim and the data
    //    dependency.
    //    d. reduction. Need to analysis broadcast dim and the
    //    data dependency.
    // Same group operations have no data dependencies. They can be fused into a
    // common for loop body.

    // Using queue to store the operation order. In order to ensure that
    // subsequent moves to the operation will not cause semantic changes.
    llvm::SmallVector<std::queue<Operation *>, 8> opGroups;
    llvm::SmallVector<VectorType, 8> groupsShapes;
    // dummy
    opGroups.emplace_back(std::queue<Operation *>());

    // query current operation in which group, return group index
    llvm::DenseMap<Operation *, size_t> opGroupIndexMap;
    classifyOperations(func, opGroups, opGroupIndexMap, groupsShapes);

    // 2. Analysis the operation's operands and results
    // We need to analyze which operation results are needed by other
    // operations, and we need to pass these results correctly. Mapping the
    // operation result value to forloop yeild result value. We can replace the
    // operation operand as: map(operand, forloop yield result) -> operand =
    // loop yield result We put all the operation result into this map.

    // 2.a. Find what results should be generated by current group for
    // using as operands to other operations?

    // Traverse all operations. If the operand of operations in other groups or
    // outside the group is the result of the current group operation, then the
    // current operation needs to generate a result. We use `setvector` to save
    // the results that need to be generated by the current group.

    //  2.b. What operands are needed to find in the current group, and where
    //  can they be obtained ?

    //  Thanks to 2.a, we get the result generated by the operations of
    //  each group, and this result will use `for loop yield` to generate a
    //  new result. Since the scope of the parent block of mlir is covered
    //  the current operation, the current operation does not need to pass these
    //  `for loop results` to the `iter args` of the required `for loop`. It
    //  only needs to replace the operand of the current operation with the
    //  corresponding `for loop yield result`.

    // However, for some operations that are not DPS, we need to canonicalize
    // them. Canonicalization means that the operand of this operation is a
    // vector but we can't get this vector due to it locates in another block
    // which has a different scope. Therefore, it is necessary to write the
    // vector results into a temporary tensor to save it. Then the vector needs
    // to be read from the tensor before the current operation operate on it.
    // Therefore,  `empty tensor`, `transfer_write` and `transfer_read` need to
    // be inserted at target place.

    llvm::SmallVector<llvm::SetVector<Value>, 8> groupOperandNeedSet(
        opGroups.size(), llvm::SetVector<Value>()),
        groupResultYeildSet(opGroups.size(), llvm::SetVector<Value>()),
        groupOpDestination(opGroups.size(), llvm::SetVector<Value>());
    // Query groupResultYeildSet to map operaion result value to scf.yield
    // result value.
    IRMapping mapOpResultToYield;
    analysisGroupOperaionOperandsResults(
        opGroups, opGroupIndexMap, groupOperandNeedSet, func,
        groupResultYeildSet, mapOpResultToYield, groupOpDestination);

    OpBuilder builder(ctx);
    // store read and write operations permutation maps in order to convenient
    // to replace loop induction var
    llvm::DenseMap<Operation *, AffineMap> opPermuationMap;

    // 3.Generate vectorized IR for each operation group
    for (auto [idx, grp] : llvm::enumerate(opGroups)) {

      generateGroupOpVectorizedIR(grp, idx, opGroups, opGroupIndexMap,
                                  groupsShapes, groupResultYeildSet,
                                  groupOpDestination, mapOpResultToYield, func,
                                  opPermuationMap);
    }

    // 4. Some IR cleanup work
    DominanceInfo domInfo;
    auto reWriter = IRRewriter(func);
    eliminateCommonSubExpressions(reWriter, domInfo, func);
  }
};
} // namespace

} // namespace gc
} // namespace mlir