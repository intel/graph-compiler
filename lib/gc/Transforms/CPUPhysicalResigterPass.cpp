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
         llvm::isa<vector::MultiDimReductionOp>(op) ||
         llvm::isa<func::CallOp>(op);
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
            auto retType = mlir::dyn_cast<VectorType>(
                transferWriteOp->getOperand(0).getType());
            if (retType) {
              return retType;
            }
            LDBG("TransferWrite Operation has wrong vector to write.");
            return failure();
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

VectorType getVectorzedType(Operation *op) {
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
}

// Since we rewrite transfer_read and transfer_write, the `permutationmap` must
// be changed.
void setOpVectorizationPermutationMap(Operation *op, OpBuilder &rewriter,
                                      const RankedTensorType &tensorType,
                                      const AffineMap &permutationMap) {

  auto dimExpr = permutationMap.getResults();
  auto lastDim = mlir::dyn_cast<AffineDimExpr>(dimExpr.back());
  assert(mlir::isa<AffineDimExpr>(lastDim));

  SmallVector<AffineExpr, 1> affineExprs;
  affineExprs.push_back(lastDim);
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

Type getScalarType(Operation *op) {
  // Check that the operation type can be broken
  // down into a loop.
  auto baseType = getOperationVectorType(op);
  if (failed(baseType)) {
    LDBG("Failed to get vector type for operation: " << *op << "\n");
    assert(false && "Failed to get vector type for operation");
    return VectorType();
  }
  auto vectorizedType = baseType.value();
  return VectorType::get({1}, vectorizedType.getElementType());
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
        op->getLoc(),
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
        op->getLoc(),
        /*vectorType=*/
        VectorType::get(operandType.getShape(), operandType.getElementType()),
        /*source=*/operand,
        /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
        /**affinemap*/ padValue,
        /*inBounds=*/inBoundsVal);
    return t;
  }
}

// __________________________________
// Speical operations canonicalization
// __________________________________

//===----------------------------------------------------------------------===//
// MultiReduce Operation
//===----------------------------------------------------------------------===//

enum class MultiReduceOpAxisKind { Reduction, Parallel };
void updateReduceReadWriteOperationOperand(
    const llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::SmallVector<int64_t, 4> &parallelAxis, Operation *op,
    MultiReduceOpAxisKind rdKind = MultiReduceOpAxisKind::Parallel) {
  int indiceOffset = mlir::isa<vector::TransferReadOp>(op) ? 1 : 2;
  for (auto [idx, inductionVar] : llvm::enumerate(inductionVars)) {
    if (rdKind == MultiReduceOpAxisKind::Parallel &&
        idx >= parallelAxis.size()) {
      break;
    }
    op->setOperand(idx + indiceOffset, inductionVar);
  }
}

vector::TransferReadOp makeNewTransferReadOp(
    Value &source, OpBuilder &b, IRMapping &readMap,
    const llvm::SmallVector<int64_t, 4> &parallelAxis,
    llvm::SmallVector<Value, 5> &inductionVars, bool lastDimReduction,
    MultiReduceOpAxisKind rdKind = MultiReduceOpAxisKind::Parallel) {
  IRRewriter rewriter(b);
  auto readOp = mlir::dyn_cast<vector::TransferReadOp>(source.getDefiningOp());
  assert(readOp && " Not transfer_read operation. Current multireduction "
                   "operation may have wrong analysis IR.");

  auto clonedOp = b.clone(*readOp, readMap);
  auto newReadOp = mlir::dyn_cast<vector::TransferReadOp>(clonedOp);
  updateReduceReadWriteOperationOperand(inductionVars, parallelAxis, newReadOp,
                                        rdKind);

  // modify the type of the new read operation
  auto newOperandType =
      (lastDimReduction && rdKind == MultiReduceOpAxisKind::Reduction)
          ? getVectorzedType(newReadOp)
          : getScalarType(newReadOp);
  newReadOp->getResult(0).setType(newOperandType);
  setOpVectorizationPermutationMap(
      newReadOp, b,
      newReadOp.getSource().getType().dyn_cast<RankedTensorType>(),
      newReadOp.getPermutationMap());

  rewriter.replaceOp(readOp, newReadOp);
  return newReadOp;
}

vector::TransferWriteOp
makeNewTransferWriteOp(Value source, IRMapping &writeMap, OpBuilder &b,
                       const llvm::SmallVector<int64_t, 4> &parallelAxis,
                       llvm::SmallVector<Value, 5> &inductionVars) {
  IRRewriter bodyRewriter(b);
  auto writeOp = source.getDefiningOp();
  auto newWriteOp =
      mlir::dyn_cast<vector::TransferWriteOp>(b.clone(*writeOp, writeMap));
  updateReduceReadWriteOperationOperand(inductionVars, parallelAxis, newWriteOp,
                                        MultiReduceOpAxisKind::Parallel);
  setOpVectorizationPermutationMap(
      newWriteOp, b,
      newWriteOp->getResult(0).getType().dyn_cast<RankedTensorType>(),
      newWriteOp.getPermutationMap());
  bodyRewriter.replaceOp(writeOp, newWriteOp);
  return newWriteOp;
}

scf::ForOp reductionAxisGenerateForLoop(
    OpBuilder &opBuilder, vector::MultiDimReductionOp &multiReductionOp,
    const llvm::SmallVector<int64_t, 4> &reductionAxis,
    const size_t reductionIdx, const VectorType &vectorType,
    llvm::SmallVector<Value, 5> &inductionVars, const ValueRange &iterArgs,
    bool lastDimReduction, Location &loc, const int loopStep) {

  auto zero = opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(), 0));
  auto forSteps = opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(
          opBuilder.getIndexType(),
          (reductionIdx == reductionAxis.size() - 1 && lastDimReduction)
              ? loopStep
              : 1));
  auto numIter = opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(),
                               vectorType.getShape()[reductionIdx]));
  auto forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        if (reductionIdx == reductionAxis.size() - 1) {
          IRRewriter rewriter(b);
          IRMapping readMap;
          Value reductionTarget = multiReductionOp->getOperand(0);
          llvm::SmallVector<int64_t, 4> parallelAxis;
          auto newReadOp = makeNewTransferReadOp(
              reductionTarget, b, readMap, parallelAxis, inductionVars,
              lastDimReduction, MultiReduceOpAxisKind::Reduction);

          // reduction or elementwise reduce
          // if (lastDimReduction) {
          //   Operation *reductionOp = rewriter.create<vector::ReductionOp>(
          //       loc, multiReductionOp.getKind(), newReadOp->getResult(0),
          //       loopState.back());
          //   maybeYieldValue(b, loc, reductionOp->getResults());
          // } else {
          auto reductionResult =
              makeArithReduction(b, loc, multiReductionOp.getKind(),
                                 newReadOp->getResult(0), loopState.back());
          maybeYieldValue(b, loc, reductionResult);
          // }
        } else {
          // outter loop
          auto nxtFor = reductionAxisGenerateForLoop(
              b, multiReductionOp, reductionAxis, reductionIdx + 1, vectorType,
              inductionVars, loopState, lastDimReduction, loc, loopStep);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });

  return forOp;
}

scf::ForOp parallelAxisGenerateForLoop(
    OpBuilder &opBuilder, vector::MultiDimReductionOp &multiReductionOp,
    const llvm::SmallVector<int64_t, 4> &parallelAxis, const size_t parallelIdx,
    const llvm::SmallVector<int64_t, 4> &reductionAxis,
    const size_t reductionIdx, const VectorType &vectorType,
    llvm::SmallVector<Value, 5> &inductionVars, const ValueRange &iterArgs,
    Value &originalWriteResult, bool lastDimReduction, Location &loc,
    const int loopStep) {
  auto zero = opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(), 0));
  auto forSteps = opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(), 1));
  auto numIter = opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(
          opBuilder.getIndexType(),
          vectorType.getShape()[parallelAxis[parallelIdx]]));
  // Create a loop and move vectorized operation into loops.
  auto forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);
        if (parallelIdx == parallelAxis.size() - 1) {

          // read operation
          IRMapping accReadMap;
          auto multiReductionAcc = multiReductionOp.getAcc();
          auto accReadOp = multiReductionAcc.getDefiningOp();
          accReadMap.map(accReadOp->getOperand(0), loopState.back());

          auto newAccReadOp = makeNewTransferReadOp(
              multiReductionAcc, b, accReadMap, parallelAxis, inductionVars,
              lastDimReduction, MultiReduceOpAxisKind::Parallel);
          auto resultElementType = vectorType.getElementType();
          // constructe next for loop
          // auto accVal = b.create<arith::ConstantOp>(
          //     loc, opBuilder.getZeroAttr(vectorType.getElementType()));
          Attribute initValueAttr;
          if (isa<FloatType>(resultElementType)) {
            initValueAttr = FloatAttr::get(resultElementType, 0.0);

          } else {
            initValueAttr = IntegerAttr::get(resultElementType, 0);
          }
          auto accVal = b.create<arith::ConstantOp>(
              loc, DenseElementsAttr::get(getVectorzedType(multiReductionOp),
                                          {initValueAttr}));

          ValueRange newIterArgs(accVal);
          auto nxtFor = reductionAxisGenerateForLoop(
              b, multiReductionOp, reductionAxis, reductionIdx, vectorType,
              inductionVars, newIterArgs, lastDimReduction, loc, loopStep);

          // insert accumulate value to original vector
          auto accRes = nxtFor->getResults()[0];

          Operation *reductionOp = b.create<vector::ReductionOp>(
              loc, multiReductionOp.getKind(), accRes);
          auto insertOp = b.create<vector::InsertOp>(
              loc, reductionOp->getResult(0), newAccReadOp->getResults()[0], 0);

          // write vector back to tensor
          vector::TransferWriteOp accWriteOp = nullptr;
          for (auto [idx, x] :
               llvm::enumerate(multiReductionOp->getResults()[0].getUsers())) {
            if (idx == 0 && mlir::isa<vector::TransferWriteOp>(x)) {
              accWriteOp = mlir::dyn_cast<vector::TransferWriteOp>(x);
              break;
            }
          }
          assert(accWriteOp &&
                 " Not transfer_write operation. Current multireduction "
                 "operation may have wrong analysis IR.");
          IRMapping accWriteindiceMap;
          accWriteindiceMap.map(accWriteOp.getOperand(0),
                                insertOp->getResults()[0]);
          auto writeResult = accWriteOp->getResults()[0];
          auto newAccWriteOp = makeNewTransferWriteOp(
              writeResult, accWriteindiceMap, b, parallelAxis, inductionVars);
          originalWriteResult = newAccWriteOp->getResult(0);

          maybeYieldValue(b, loc, newAccWriteOp->getResults());
        } else {
          auto nxtFor = parallelAxisGenerateForLoop(
              b, multiReductionOp, parallelAxis, parallelIdx + 1, reductionAxis,
              reductionIdx, vectorType, inductionVars, loopState,
              originalWriteResult, lastDimReduction, loc, loopStep);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
  return forOp;
}

scf::ForOp generateMultiReductionForLoop(
    OpBuilder &opBuilder, vector::MultiDimReductionOp &multiReductionOp,
    const llvm::SmallVector<int64_t, 4> &parallelAxis, const size_t parallelIdx,
    const llvm::SmallVector<int64_t, 4> &reductionAxis,
    const size_t reductionIdx, const VectorType &vectorType,
    llvm::SmallVector<Value, 5> &inductionVars, const ValueRange &iterArgs,
    Value &originalWriteResult, bool lastDimReduction) {
  const int loopStep = getDataTypeMAXSIMDLength(vectorType);
  auto loc = multiReductionOp->getLoc();

  scf::ForOp forOp = parallelAxisGenerateForLoop(
      opBuilder, multiReductionOp, parallelAxis, parallelIdx, reductionAxis,
      reductionIdx, vectorType, inductionVars, iterArgs, originalWriteResult,
      lastDimReduction, loc, loopStep);
  return forOp;
}

// mlir::FailureOr<scf::ForOp> generateTransposeForLoop(
//     OpBuilder &opBuilder, vector::TransposeOp &transposeOp,
//     const llvm::SmallVector<int64_t, 4> &parallelAxis, const size_t
//     parallelIdx, const llvm::SmallVector<int64_t, 4> &reductionAxis, const
//     size_t reductionIdx, const VectorType &vectorType,
//     llvm::SmallVector<Value, 5> &inductionVars, const ValueRange &iterArgs,
//     Value &originalWriteResult, bool lastDimReduction) {
//   const int loop_step = getDataTypeMAXSIMDLength(vectorType);
//   auto loc = transposeOp->getLoc();
//   auto zero = opBuilder.create<arith::ConstantOp>(
//       loc, opBuilder.getIndexType(),
//       opBuilder.getIntegerAttr(opBuilder.getIndexType(), 0));

//   scf::ForOp forOp = nullptr;
//   // parallel axis
//   if (parallelIdx < parallelAxis.size()) {
//     auto forSteps = opBuilder.create<arith::ConstantOp>(
//         loc, opBuilder.getIndexType(),
//         opBuilder.getIntegerAttr(opBuilder.getIndexType(), 1));
//     auto numIter = opBuilder.create<arith::ConstantOp>(
//         loc, opBuilder.getIndexType(),
//         opBuilder.getIntegerAttr(
//             opBuilder.getIndexType(),
//             vectorType.getShape()[parallelAxis[parallelIdx]]));
//     // Create a loop and move vectorized operation into loops.
//     forOp = opBuilder.create<scf::ForOp>(
//         loc, zero, numIter, forSteps, iterArgs,
//         [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
//           inductionVars.emplace_back(iv);
//           if (parallelIdx == parallelAxis.size() - 1) {
//             // move original transfer_read operation into parallel axis loop
//             // body
//             // get read operation
//             Value multiReductionAcc = multiReductionOp.getAcc();
//             auto accReadOp =
//                 multiReductionAcc.getDefiningOp<vector::TransferReadOp>();
//             assert(accReadOp &&
//                    " Not transfer_read operation. Current multireduction "
//                    "operation may have wrong analysis IR.");
//             // get write operation
//             vector::TransferWriteOp accWriteOp = nullptr;
//             for (auto [idx, x] : llvm::enumerate(
//                      multiReductionOp->getResults()[0].getUsers())) {
//               if (idx == 0 && mlir::isa<vector::TransferWriteOp>(x)) {
//                 accWriteOp = mlir::dyn_cast<vector::TransferWriteOp>(x);
//                 break;
//               }
//             }
//             assert(accWriteOp);
//             IRMapping accReadindiceMap;

//             IRRewriter bodyRewriter(b);
//             auto newAccReadOp = mlir::dyn_cast<vector::TransferReadOp>(
//                 b.clone(*accReadOp, accReadindiceMap));
//             bodyRewriter.replaceOp(accReadOp, newAccReadOp);
//             int offset = 1;
//             for (auto [idx, inductionVar] : llvm::enumerate(inductionVars)) {
//               if (idx >= parallelAxis.size()) {
//                 break;
//               }
//               newAccReadOp->setOperand(idx + offset, inductionVar);
//             }
//             auto newOperandType = getScalarType(newAccReadOp);
//             newAccReadOp->getResult(0).setType(newOperandType);
//             setOpVectorizationPermutationMap(
//                 newAccReadOp, b,
//                 newAccReadOp.getSource().getType().dyn_cast<RankedTensorType>(),
//                 newAccReadOp.getPermutationMap());
//             // constructe next for loop
//             auto accVal = b.create<arith::ConstantOp>(
//                 loc, opBuilder.getZeroAttr(vectorType.getElementType()));
//             ValueRange newIterArgs(accVal);
//             auto nxtFor = generateMultiReductionForLoop(
//                 b, multiReductionOp, parallelAxis, parallelIdx + 1,
//                 reductionAxis, reductionIdx, vectorType, inductionVars,
//                 newIterArgs, originalWriteResult, lastDimReduction);

//             // move original transfer_write into loop
//             auto accRes = nxtFor.value()->getResults()[0];

//             // replace the vector as the loop return vector value
//             llvm::SmallVector<int64_t> insertPos;
//             auto insertOp = b.create<vector::InsertOp>(
//                 loc, accRes, newAccReadOp->getResult(0), 0);
//             IRMapping accWriteindiceMap;
//             accWriteindiceMap.map(accWriteOp.getOperand(0),
//                                   insertOp->getResults()[0]);
//             auto newAccWriteOp = mlir::dyn_cast<vector::TransferWriteOp>(
//                 b.clone(*accWriteOp, accWriteindiceMap));
//             offset = 2;
//             for (auto [idx, inductionVar] : llvm::enumerate(inductionVars)) {
//               if (idx >= parallelAxis.size()) {
//                 break;
//               }
//               newAccWriteOp->setOperand(idx + offset, inductionVar);
//             }
//             setOpVectorizationPermutationMap(newAccWriteOp, b,
//                                              newAccWriteOp->getResult(0)
//                                                  .getType()
//                                                  .dyn_cast<RankedTensorType>(),
//                                              newAccWriteOp.getPermutationMap());
//             bodyRewriter.replaceOp(accWriteOp, newAccWriteOp);
//             originalWriteResult = newAccWriteOp->getResult(0);
//             maybeYieldValue(b, loc, newAccWriteOp->getResults());
//           } else {
//             auto nxtFor = generateMultiReductionForLoop(
//                 b, multiReductionOp, parallelAxis, parallelIdx + 1,
//                 reductionAxis, reductionIdx, vectorType, inductionVars,
//                 iterArgs, originalWriteResult, lastDimReduction);
//             maybeYieldValue(b, loc, nxtFor.value()->getResults());
//           }
//         });

//   } else {

//     auto forSteps = opBuilder.create<arith::ConstantOp>(
//         loc, opBuilder.getIndexType(),
//         opBuilder.getIntegerAttr(
//             opBuilder.getIndexType(),
//             reductionIdx == reductionAxis.size() - 1 ? loop_step : 1));
//     auto numIter = opBuilder.create<arith::ConstantOp>(
//         loc, opBuilder.getIndexType(),
//         opBuilder.getIntegerAttr(opBuilder.getIndexType(),
//                                  vectorType.getShape()[reductionIdx]));
//     forOp = opBuilder.create<scf::ForOp>(
//         loc, zero, numIter, forSteps, iterArgs,
//         [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
//           inductionVars.emplace_back(iv);

//           if (reductionIdx == reductionAxis.size() - 1) {
//             auto source = multiReductionOp->getOperand(0);

//             auto readOp =
//                 mlir::dyn_cast<vector::TransferReadOp>(source.getDefiningOp());
//             assert(readOp);
//             IRMapping indiceMap;
//             IRRewriter rewriter(b);
//             auto clonedOp = b.clone(*readOp, indiceMap);
//             int offset = 1;
//             auto newReadOp =
//             mlir::dyn_cast<vector::TransferReadOp>(clonedOp);

//             for (auto [idx, inductionVar] : enumerate(inductionVars)) {
//               newReadOp->setOperand(idx + offset, inductionVar);
//             }

//             auto newOperandType = lastDimReduction ?
//             getVectorzedType(newReadOp)
//                                                    :
//                                                    getScalarType(newReadOp);
//             newReadOp->getResult(0).setType(newOperandType);
//             setOpVectorizationPermutationMap(
//                 newReadOp, b,
//                 newReadOp.getSource().getType().dyn_cast<RankedTensorType>(),
//                 newReadOp.getPermutationMap());
//             rewriter.replaceOp(readOp, newReadOp);
//             if (lastDimReduction) {
//               Operation *reductionOp = rewriter.create<vector::ReductionOp>(
//                   loc, multiReductionOp.getKind(), newReadOp->getResult(0),
//                   loopState.back());
//               maybeYieldValue(b, loc, reductionOp->getResults());
//             } else {
//               auto reductionResult =
//                   makeArithReduction(b, loc, multiReductionOp.getKind(),
//                                      newReadOp->getResult(0),
//                                      loopState.back());
//               maybeYieldValue(b, loc, reductionResult);
//             }
//           } else {
//             // outter loop
//             auto nxtFor = generateMultiReductionForLoop(
//                 b, multiReductionOp, parallelAxis, parallelIdx,
//                 reductionAxis, reductionIdx + 1, vectorType, inductionVars,
//                 iterArgs, originalWriteResult, lastDimReduction);
//             maybeYieldValue(b, loc, nxtFor.value()->getResults());
//           }
//         });
//   }
//   return forOp;
// }

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
class VectorFusionStrategy {
public:
  llvm::SmallVector<std::queue<Operation *>, 8> &getOpGroups() {
    return opGroups;
  }
  llvm::DenseMap<Operation *, size_t> &getOpGroupIndexMap() {
    return opGroupIndexMap;
  }

  func::FuncOp getFunc() { return func; }

  VectorFusionStrategy() = default;
  VectorFusionStrategy(func::FuncOp func) : func(func) {}

  void
  classifyOperations(func::FuncOp func,
                     llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
                     llvm::DenseMap<Operation *, size_t> &opGroupIndexMap);

  // run the vector fusion strategy
  void run();

private:
  llvm::SmallVector<std::queue<Operation *>, 8> opGroups;
  // query current operation in which group, return group index
  llvm::DenseMap<Operation *, size_t> opGroupIndexMap;

  func::FuncOp func;
};

void VectorFusionStrategy::run() {
  classifyOperations(func, opGroups, opGroupIndexMap);
}

enum CanonicalizerKind { OperationsGroup, Operations };

class CanonicalizerVectorOperation {
public:
  func::FuncOp func;
  IRRewriter rewriter;
  VectorFusionStrategy fusionStrategy;
  CanonicalizerKind kind;

  // analysis the operation's operands and results
  llvm::SmallVector<llvm::SetVector<Value>, 8> groupOpResults, groupOpIterArgs;

  // store read and write operations permutation maps in order to convenient
  // to replace loop induction var
  llvm::DenseMap<Operation *, AffineMap> opPermuationMap;

  CanonicalizerVectorOperation(
      func::FuncOp func,
      CanonicalizerKind kind = CanonicalizerKind::OperationsGroup)
      : func(func), rewriter(func), kind(kind) {
    // vector operation fusion
    if (kind == CanonicalizerKind::OperationsGroup) {
      fusionStrategy = VectorFusionStrategy(func);
      fusionStrategy.run();
    }
  }
  func::FuncOp getFunc() { return func; };

  void generateGroupOpVectorizedIR(
      const int idx, std::queue<Operation *> &grp,
      llvm::DenseMap<Operation *, size_t> &opGroupIndexMap);

  void analysisGroupOperaionOperandsResults(
      llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
      llvm::DenseMap<Operation *, size_t> &opGroupIndexMap);

  void analysisGroupOperationResults(
      llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
      llvm::DenseMap<Operation *, size_t> &opGroupIndexMap);

  void canonicalizeSpecialOperation();
  LogicalResult
  canonicalizeReductionOperation(vector::MultiDimReductionOp &multiReductionOp,
                                 IRRewriter &rewriter);
  LogicalResult canonicalizeTransposeOperation(vector::TransposeOp &transposeOp,
                                               IRRewriter &rewriter);

  void run();

private:
  llvm::SetVector<vector::MultiDimReductionOp> multiReductionOps;
  llvm::SetVector<vector::ShapeCastOp> shapeCastOps;
};

// LogicalResult CanonicalizerVectorOperation::canonicalizeTransposeOperation(
//     vector::TransposeOp &transposeOp, IRRewriter &rewriter) {
//   OpBuilder::InsertionGuard guard(rewriter);

//   auto srcVecType = multiReductionOp.getSourceVectorType();
//   auto srcRank = multiReductionOp.getSourceVectorType().getRank();

//   // Separate reduction and parallel dims
//   bool lastDimReduction = false;
//   auto reductionAxisRange =
//       multiReductionOp.getReductionDims().getAsValueRange<IntegerAttr>();
//   auto reductionRange = llvm::to_vector<4>(llvm::map_range(
//       reductionAxisRange, [](const APInt &a) { return a.getZExtValue(); }));
//   llvm::SmallVector<int64_t, 4> reductionAxis(reductionRange.begin(),
//                                               reductionRange.end());
//   llvm::SmallDenseSet<int64_t> reductionAxisSet(reductionAxis.begin(),
//                                                 reductionAxis.end());
//   if (reductionAxisSet.contains(srcRank - 1)) {
//     lastDimReduction = true;
//   }
//   SmallVector<int64_t, 4> parallelAxis;
//   for (int64_t i = 0; i < srcRank; ++i) {
//     if (!reductionAxisSet.contains(i)) {
//       parallelAxis.push_back(i);
//     }
//   }
//   /*
//    * The final IR may look like below:
//    * _for_(_fuseiter_i, 0, 1)
//    *  sum = 0;
//    *  _for_(_fuseiter_j, 0, 1)
//    *   _for_(_fuseiter_k, 0, 1)
//    *     sum += src[src_idx];
//    *  dst[dst_idx] = sum;
//    * */
//   Operation *newReduction;
//   Value multiReductionAcc = multiReductionOp.getAcc();
//   auto accTensorReadOp =
//       multiReductionAcc.getDefiningOp<vector::TransferReadOp>();
//   Value originalWriteResult;
//   ValueRange iterArgs(accTensorReadOp->getOperand(0));
//   llvm::SmallVector<Value, 5> inductionVars;
//   auto forOp = generateMultiReductionForLoop(
//       rewriter, multiReductionOp, parallelAxis, 0, reductionAxis, 0,
//       srcVecType, inductionVars, iterArgs, originalWriteResult,
//       lastDimReduction);
//   if (failed(forOp)) {
//     LDBG("MultiReduction Operation lowering failed");
//     return failure();
//   }
//   auto replaceIfFn = [&](OpOperand &use) {
//     return use.getOwner()->getBlock() !=
//            originalWriteResult.getDefiningOp()->getBlock();
//   };
//   newReduction = forOp.value();
//   rewriter.replaceOpUsesWithIf(originalWriteResult.getDefiningOp(),
//                                newReduction->getResults()[0], replaceIfFn);

//   rewriter.replaceOp(multiReductionOp, newReduction);
//   return success();
// }

LogicalResult CanonicalizerVectorOperation::canonicalizeReductionOperation(
    vector::MultiDimReductionOp &multiReductionOp, IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto srcVecType = multiReductionOp.getSourceVectorType();
  auto srcRank = multiReductionOp.getSourceVectorType().getRank();

  // Separate reduction and parallel dims
  bool lastDimReduction = false;
  auto reductionAxisRange =
      multiReductionOp.getReductionDims().getAsValueRange<IntegerAttr>();
  auto reductionRange = llvm::to_vector<4>(llvm::map_range(
      reductionAxisRange, [](const APInt &a) { return a.getZExtValue(); }));
  llvm::SmallVector<int64_t, 4> reductionAxis(reductionRange.begin(),
                                              reductionRange.end());
  llvm::SmallDenseSet<int64_t> reductionAxisSet(reductionAxis.begin(),
                                                reductionAxis.end());
  if (reductionAxisSet.contains(srcRank - 1)) {
    lastDimReduction = true;
  }
  SmallVector<int64_t, 4> parallelAxis;
  for (int64_t i = 0; i < srcRank; ++i) {
    if (!reductionAxisSet.contains(i)) {
      parallelAxis.push_back(i);
    }
  }
  /*
   * The final IR may look like below:
   * _for_(_fuseiter_i, 0, 1)
   *  sum = 0;
   *  _for_(_fuseiter_j, 0, 1)
   *   _for_(_fuseiter_k, 0, 1)
   *     sum += src[src_idx];
   *  dst[dst_idx] = sum;
   * */
  Operation *newReduction;
  Value multiReductionAcc = multiReductionOp.getAcc();
  auto accTensorReadOp =
      multiReductionAcc.getDefiningOp<vector::TransferReadOp>();
  Value originalWriteResult;
  ValueRange iterArgs(accTensorReadOp->getOperand(0));
  llvm::SmallVector<Value, 5> inductionVars;
  auto forOp = generateMultiReductionForLoop(
      rewriter, multiReductionOp, parallelAxis, 0, reductionAxis, 0, srcVecType,
      inductionVars, iterArgs, originalWriteResult, lastDimReduction);
  auto replaceIfFn = [&](OpOperand &use) {
    return use.getOwner()->getBlock() !=
           originalWriteResult.getDefiningOp()->getBlock();
  };
  newReduction = forOp;
  rewriter.replaceOpUsesWithIf(originalWriteResult.getDefiningOp(),
                               newReduction->getResults()[0], replaceIfFn);

  rewriter.replaceOp(multiReductionOp, newReduction);
  return success();
}

void CanonicalizerVectorOperation::canonicalizeSpecialOperation() {
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<vector::MultiDimReductionOp>(
            [&](vector::MultiDimReductionOp multiReductionOp) {
              multiReductionOps.insert(multiReductionOp);
            })
        .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp shapeCastOp) {
          shapeCastOps.insert(shapeCastOp);
        })
        .Default([&](Operation *) {});
  });
  // process reduction
  for (auto x : multiReductionOps) {
    IRRewriter rewriter(x);
    (void)canonicalizeReductionOperation(x, rewriter);
  }
  // process shapecast
  // for (auto x : shapeCastOps) {
  // }
  return;
}

void CanonicalizerVectorOperation::run() {

  if (kind == CanonicalizerKind::OperationsGroup) {
    // 1. Analysis the operation's operands and results
    // We need to analyze which operation results are needed by other
    // operations, and we need to pass these results correctly. Mapping the
    // operation result value to forloop yeild result value. We can replace the
    // operation operand as: map(operand, forloop yield result) -> operand =
    // loop yield result We put all the operation result into this map.

    // 1.a. Find what results should be generated by current group for
    // using as operands to other operations?

    // Traverse all operations. If the operand of operations in other groups or
    // outside the group is the result of the current group operation, then the
    // current operation needs to generate a result. We use `setvector` to save
    // the results that need to be generated by the current group.

    //  1.b. What operands are needed to find in the current group, and where
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

    // Query groupResultYeildSet to map operaion result value to scf.yield
    // result value.
    analysisGroupOperaionOperandsResults(fusionStrategy.getOpGroups(),
                                         fusionStrategy.getOpGroupIndexMap());

    // Speical Operation Canonicalization
    canonicalizeSpecialOperation();
    // 2.Generate vectorized IR for each operation group
    for (auto [idx, grp] : llvm::enumerate(fusionStrategy.getOpGroups())) {

      generateGroupOpVectorizedIR(idx, grp,
                                  fusionStrategy.getOpGroupIndexMap());
    }

    // 3. Some IR cleanup work
    DominanceInfo domInfo;
    eliminateCommonSubExpressions(rewriter, domInfo, func);
  } else {
    // TODO: need to add directly canonicalize operations
    // generateGroupOpVectorizedIR(idx, grp, fusionStrategy.opGroupIndexMap);
  }
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
  bool isCompatible = true;
  // from front to back
  for (long i = 0; i < min_rank; i++) {
    if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
      isCompatible = false;
      break;
    }
  }

  return isCompatible;
}

///  which axis do the shape cast in source shape a
void shapeCastSourceAxis(const ArrayRef<int64_t> &a, const ArrayRef<int64_t> &b,
                         llvm::SmallVector<int64_t> &res) {
  unsigned rankA = a.size();
  unsigned rankB = b.size();
  assert(rankA < rankB && "May be invalid shape cast operation.");

  auto isOne = [](int64_t v) { return v == 1; };

  // Special-case for n-D to 0-d shape cast. 'b' must be all ones to be shape
  // casted to a 0-d vector.
  if (rankA == 0 && llvm::all_of(b, isOne)) {
    for (size_t i = 0; i < a.size(); i++) {
      res.emplace_back(i);
    }
    return;
  }

  unsigned i = 0;
  unsigned j = 0;
  while (i < rankA && j < rankB) {
    int64_t dimA = a[i];
    int64_t dimB = 1;
    int64_t bAxisBegin = j;
    while (dimB < dimA && j < rankB)
      dimB *= b[j++];
    if (dimA != dimB) {
      assert(false && " Invalid shape cast operation.");
      break;
    }
    if (bAxisBegin != j) {
      res.emplace_back(i);
    }
    ++i;

    // Handle the case when trailing dimensions are of size 1.
    // Include them into the contiguous sequence.
    if (i < rankA && llvm::all_of(a.slice(i), isOne))
      i = rankA;
    if (j < rankB && llvm::all_of(b.slice(j), isOne))
      j = rankB;
  }

  assert(i == rankA && j == rankB && "Invalid shapecast operation.");
}

void getOperationDataAxis(Operation *op, llvm::SmallVector<int64_t> &dataAxis) {
  return TypeSwitch<Operation *>(op)
      .Case<vector::MultiDimReductionOp>(
          [&](vector::MultiDimReductionOp multiReductionOp) {
            auto rdDimsRange = multiReductionOp.getReductionDims()
                                   .getAsValueRange<IntegerAttr>();
            auto reductionDims = llvm::to_vector<4>(llvm::map_range(
                rdDimsRange, [](const APInt &a) { return a.getZExtValue(); }));
            dataAxis.assign(reductionDims.begin(), reductionDims.end());
          })
      .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp shapeCastOp) {
        auto srcType = shapeCastOp.getSourceVectorType();
        auto dstType = shapeCastOp.getResultVectorType();
        auto srcShape = srcType.getShape();
        auto dstShape = dstType.getShape();
        shapeCastSourceAxis(srcShape, dstShape, dataAxis);
      })
      .Default([&](Operation *op) {
        // default is last axis
        dataAxis.emplace_back(
            op->getResultTypes().front().cast<ShapedType>().getRank() - 1);
      });
}

bool hasDataDependency(Operation *op1, Operation *op2) {
  // op1 must be special operation
  if (!isSpecialOp(op1)) {
    return hasDataDependency(op2, op1);
  }
  auto hasSameAxis = [](const llvm::SmallVector<int64_t> &dims1,
                        const llvm::SmallVector<int64_t> &dims2) {
    llvm::DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());
    for (auto x : dims1) {
      if (checkSet.contains(x)) {
        return true;
      }
    }
    return false;
  };
  auto res =
      TypeSwitch<Operation *, bool>(op1)
          .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp shapeCastOp) {
            llvm::SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            return hasSameAxis(dims1, dims2);
          })
          .Case<>([&](vector::MultiDimReductionOp multiReductionOp) {
            // op1 is special operation, op2 is normal operation
            // op1 and op2 is both speicial operation
            auto rdDimsRange = multiReductionOp.getReductionDims()
                                   .getAsValueRange<IntegerAttr>();
            auto reductionDims = llvm::to_vector(
                llvm::map_range(rdDimsRange, [](const APInt &a) {
                  return (int64_t)a.getZExtValue();
                }));
            llvm::SmallVector<int64_t> dims2;
            getOperationDataAxis(op2, dims2);
            llvm::DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());

            for (auto x : dims2) {
              if (!checkSet.contains(x)) {
                return true;
              }
            }
            return false;
          })
          .Default([&](Operation *op) { return false; });

  return res;
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
      // special operation need to check data dependency axis
      if (hasDataDependency(prevOp, op)) {
        return true;
      }
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
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap, Operation *op) {
  //
  if (isNeedNewGroup(opGroups, op)) {
    opGroups.emplace_back(std::queue<Operation *>());
  }
  opGroups.back().push(op);
  opGroupIndexMap[op] = opGroups.size() - 1;
}

// We classify the operations we are interested in after filtering. Operations
// of in the same group have no data dependencies. Those operations can generate
// a same outter for loop.
void VectorFusionStrategy::classifyOperations(
    func::FuncOp func, llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap) {
  if (opGroups.empty()) {
    // dummpy
    opGroups.emplace_back(std::queue<Operation *>());
  }
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *>(op).Default([&](Operation *op) {
      if (filterOperation(op)) {
        addOperationToGroup(opGroups, opGroupIndexMap, op);
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
                   if (isa<FloatType>(resultElementType)) {
                     initValueAttr = FloatAttr::get(resultElementType, 0.0);

                   } else {
                     initValueAttr = IntegerAttr::get(resultElementType, 0);
                   }
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

/// Rewrite the operations in the group to vectorized form.
void rewriteOperationAsVectorize(
    const std::queue<Operation *> &groupOps,
    const llvm::DenseMap<Operation *, size_t> &opMap, IRRewriter &rewriter,
    llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  std::queue<Operation *> transformQueue(groupOps);

  while (!transformQueue.empty()) {
    auto op = transformQueue.front();
    transformQueue.pop();
    auto lowerResult =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<vector::TransferWriteOp>(
                [&](vector::TransferWriteOp transferWriteOp) {
                  IRRewriter rewriter(transferWriteOp);
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
                            .dyn_cast<RankedTensorType>(),
                        transferWriteOp.getPermutationMap());
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
                            .dyn_cast<RankedTensorType>(),
                        transferReadOp.getPermutationMap());
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
// void analysisOperaionOperandSource(
//     size_t idx, std::queue<Operation *> &grp,
//     llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
//     llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOperandNeedSet) {
//   auto tmpOpQueue(grp);
//   llvm::SetVector<Value> opOperands;
//   while (!tmpOpQueue.empty()) {
//     auto t = tmpOpQueue.front();
//     for (auto x : t->getOperands()) {
//       // not in the same group
//       if (opGroupIndexMap.contains(x.getDefiningOp()) &&
//           opGroupIndexMap[x.getDefiningOp()] != idx) {
//         groupOperandNeedSet[idx].insert(x);
//       } else {
//         groupOperandNeedSet[idx].insert(x);
//       }
//     }
//     tmpOpQueue.pop();
//   }
// }

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
// void analysisGroupOperationOperands(
//     llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
//     llvm::DenseMap<Operation *, size_t> &opGroupIndexMap,
//     llvm::SmallVector<llvm::SetVector<Value>, 8> &groupOperandNeedSet) {

//   for (auto [idx, grp] : enumerate(opGroups)) {
//     analysisOperaionOperandSource(idx, grp, opGroupIndexMap,
//                                   groupOperandNeedSet);
//   }
// }

// TODO: need to rewrite reduce
// llvm::SmallVector<int64_t, 5> &
// getreductionAxis(vector::MultiDimReductionOp &reductionOp,
//                  llvm::SmallVector<int64_t, 5> &rdDims) {
//   auto rdDimsAttr = reductionOp.getreductionAxis().getValue();
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
void CanonicalizerVectorOperation::analysisGroupOperationResults(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap) {
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
              groupOpIterArgs[sourceOpGid].insert(init);
              groupOpResults[sourceOpGid].insert(result);
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
              groupOpIterArgs[sourceOpGid].insert(dstRet.value());
              groupOpResults[sourceOpGid].insert(opd);
            }
          }
        }
      }
    }
  });
  // If the group operations do not have result need to be returned, these are
  // useless code.
  for (auto [idx, grp] : enumerate(opGroups)) {
    if (groupOpResults[idx].empty()) {
      std::queue<Operation *>().swap(grp);
    }
  }
  LDBG("Complete analysis group operation results\n");
}

void CanonicalizerVectorOperation::analysisGroupOperaionOperandsResults(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap) {
  // prepare
  if (opGroups.size() != groupOpResults.size()) {
    for (size_t i = 0; i < opGroups.size(); i++) {
      groupOpResults.emplace_back(llvm::SetVector<Value>());
      groupOpIterArgs.emplace_back(llvm::SetVector<Value>());
    }
    LDBG("Size of groupOpResults is : " << groupOpResults.size());
  }

  // Operands
  // analysisGroupOperationOperands(opGroups, opGroupIndexMap);

  // Results
  analysisGroupOperationResults(opGroups, opGroupIndexMap);
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

void updateLoopResultUses(llvm::SetVector<Value> &opResults,
                          scf::ForOp *forOp) {
  if (opResults.empty()) {
    return;
  }
  IRRewriter rewriter(*forOp);
  OpBuilder::InsertionGuard g(rewriter);
  // Only different group operation operand need to be replaced due to same
  // group operation should directly use original operand.
  auto producerOp = opResults.front().getDefiningOp();
  auto needToReplaced = [&](OpOperand &operand) {
    return producerOp->getBlock() != operand.getOwner()->getBlock();
  };
  // update loop result uses
  for (auto [retIdx, rt] : llvm::enumerate(opResults)) {
    producerOp = rt.getDefiningOp();
    rewriter.replaceUsesWithIf(rt, forOp->getResult(retIdx), needToReplaced);
  }
}

bool hasSpecialOperation(std::queue<Operation *> &grp) {
  std::queue<Operation *> tmpQ(grp);
  while (!tmpQ.empty()) {
    auto curOp = tmpQ.front();
    if (mlir::isa<vector::MultiDimReductionOp>(curOp) or
        mlir::isa<vector::ShapeCastOp>(curOp)) {
      return true;
    }
    tmpQ.pop();
  }
  return false;
}

void CanonicalizerVectorOperation::generateGroupOpVectorizedIR(
    const int idx, std::queue<Operation *> &grp,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap) {
  if (grp.empty()) {
    LDBG("Current operation Group is empty.");
    return;
  }
  // TODO: special operation better fusion
  if (hasSpecialOperation(grp)) {
    return;
  }
  auto getType = getOperationVectorType(grp.front());
  if (failed(getType)) {
    LDBG("Failed to get vector type for operation: " << *grp.front() << "\n");
    return;
  }
  auto opShapes = getType.value();
  IRRewriter rewriter(grp.back());
  rewriter.setInsertionPointAfter(grp.back());
  // 1. Rewrite operation as vectorized form
  rewriteOperationAsVectorize(grp, opGroupIndexMap, rewriter, opPermuationMap);
  // 2. Generate loop
  auto forOp = generateVectorizedForLoop(rewriter, groupOpResults[idx],
                                         groupOpIterArgs[idx], opShapes, grp,
                                         opPermuationMap);
  // special operation do not need to change anything
  if (failed(forOp)) {
    return;
  }
  // 3 Update loop result uses
  updateLoopResultUses(groupOpResults[idx], &forOp.value());
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
    // canonicalize vector operation, default use vector-based fusion strategy.
    CanonicalizerVectorOperation canonicalizer(func);
    canonicalizer.run();
  }
};
} // namespace

} // namespace gc
} // namespace mlir