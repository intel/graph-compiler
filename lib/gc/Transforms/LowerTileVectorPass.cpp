//===- LowerTileVectorPass.cpp.cpp - OneDNNGraph To Linalg
// Lowering -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iostream>

namespace mlir {
namespace gc {

#define GEN_PASS_DEF_LOWERTOTILEVECTOR
#include "gc/Transforms/Passes.h.inc"
namespace {
#define DEBUG_TYPE "lower-to-tile-vector-pass"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

bool is_innermost_ir(Operation *op) {
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

/// Need to check if the reassociation are static/constant.
LogicalResult lowerExpandOpPrecondition(tensor::ExpandShapeOp expandOp) {
  //
  auto outputShape = expandOp.getStaticOutputShape();
  if (llvm::any_of(outputShape,
                   [](int64_t x) { return x == ShapedType::kDynamic; })) {
    LDBG("Output shape must be static: " << expandOp << "\n");
    return failure();
  }

  return success();
}

LogicalResult lowerBitcastOpPrecondition(tensor::BitcastOp bitCastOp) {
  if (bitCastOp.getSource().getType().getNumDynamicDims()) {
    LDBG("Type must be static: " << bitCastOp << "\n");
    return failure();
  }
  return success();
}

/// Need to check if the reassociation are static/constant.
LogicalResult
lowerCollapseShapeOpPrecondition(tensor::CollapseShapeOp expandOp) {

  if (llvm::any_of(expandOp.getReassociation(), [](Attribute x) {
        return !getConstantIntValue(x).has_value();
      })) {
    LDBG("Reassociation must be constant: " << expandOp << "\n");
    return failure();
  }
  return success();
}

LogicalResult lowerConcatOpPrecondition(tensor::ConcatOp concatOp) {
  for (auto x : concatOp->getOperands()) {
    auto tensorType = mlir::dyn_cast<TensorType>(x.getType());
    if (!tensorType) {
      LDBG("Operation type error: " << concatOp << "\n");
      return failure();
    }
    if (tensorType.getNumDynamicDims()) {
      LDBG("Type must be static: " << concatOp << "\n");
      return failure();
    }
  }

  return success();
}

LogicalResult lowerTargetOpPrecondition(Operation *op) {

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<tensor::ExpandShapeOp>([&](auto expandShapeOp) {
        return lowerExpandOpPrecondition(expandShapeOp);
      })
      .Case<tensor::CollapseShapeOp>([&](auto collapseShapeOp) {
        return lowerCollapseShapeOpPrecondition(collapseShapeOp);
      })
      .Case<tensor::BitcastOp>(
          [&](auto bitCastOp) { return lowerBitcastOpPrecondition(bitCastOp); })
      .Case<tensor::ConcatOp>(
          [&](auto concatOp) { return lowerConcatOpPrecondition(concatOp); })
      .Default([](auto) { return failure(); });
}

/// Create a TransferReadOp from `source` with static shape `readShape`.
Value createTransferRead(OpBuilder &builder, Location loc, Value source,
                         ArrayRef<int64_t> readShape) {
  assert(llvm::none_of(readShape,
                       [](int64_t s) { return s == ShapedType::kDynamic; }));
  assert(source && " source null.");
  auto shapedType = mlir::dyn_cast<ShapedType>(source.getType());
  auto sourceShape = shapedType.getShape();
  auto vectorType = VectorType::get(readShape, shapedType.getElementType());

  auto padValue = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(shapedType.getElementType()));
  assert(sourceShape.size() == readShape.size());
  int64_t readRank = readShape.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<bool> inBoundsVal(readRank, true);
  auto transferReadOp = builder.create<vector::TransferReadOp>(
      loc,
      /*vectorType=*/vectorType,
      /*source=*/source,
      /*indices=*/SmallVector<Value>(readRank, zero),
      /*padding=*/padValue,
      /*inBounds=*/inBoundsVal);

  if (llvm::equal(readShape, sourceShape)) {
    return transferReadOp;
  } else {
    assert(false && "wrong shape.");
  }
}

/// create an empty destination tensor and create a TransferWriteOp from the
/// input to the empty tensor.
Operation *createTransferWrite(OpBuilder &builder, Location loc, Value input,
                               SmallVector<OpFoldResult> destSizes,
                               ArrayRef<int64_t> inputVectorSizes) {
  auto inputType = cast<VectorType>(input.getType());
  Value dest = builder.create<tensor::EmptyOp>(loc, destSizes,
                                               inputType.getElementType());
  int64_t rank = cast<ShapedType>(dest.getType()).getRank();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Operation *write = builder.create<vector::TransferWriteOp>(
      loc,
      /*vector=*/input,
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/SmallVector<bool>(rank, true));
  auto destShape = cast<ShapedType>(dest.getType()).getShape();
  assert(llvm::none_of(
             destShape.drop_front(inputVectorSizes.size()),
             [](int64_t size) { return size == ShapedType::kDynamic; }) &&
         "InputVectorSizes may be dynamic");
  return write;
}

/// Vectorize a `tensor::expandshape` to these 3 Ops:
///   Vector::TransferReadOp - Reads a vector from the source tensor
///   ShapeCastOp - Reshape the data based on the target.
///   vector::TransferWriteOp. - Write the result vector back to the destination
///   tensor
template <class T>
LogicalResult lowerTensorExpandShapeOp(RewriterBase &rewriter, T expandShapeOp,
                                       SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(expandShapeOp);

  RankedTensorType expandShapeTensorType = expandShapeOp.getSrcType();

  SmallVector<int64_t> readMaskShape;
  ArrayRef<int64_t> sourceShape = expandShapeTensorType.getShape();
  ArrayRef<int64_t> resultShape = expandShapeOp.getResultType().getShape();
  readMaskShape.append(sourceShape.begin(), sourceShape.end());

  ReifiedRankedShapedTypeDims reifiedRetShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(expandShapeOp.getOperation())
          .reifyResultShapes(rewriter, reifiedRetShapes);
  if (status.failed()) {
    LDBG("Unable to reify result shapes of " << expandShapeOp << "\n");
    return failure();
  }
  Location loc = expandShapeOp->getLoc();

  // Read result, mask if necessary. If transferReadOp shape is not equal
  // to shape of source, then a mask is necessary.
  Value readResult = createTransferRead(
      rewriter, loc, expandShapeOp.getSrc(),
      ArrayRef<int64_t>(readMaskShape.begin(), readMaskShape.end()));

  auto resultVectorType =
      VectorType::get(resultShape, expandShapeTensorType.getElementType());
  vector::ShapeCastOp shapeCastOp =
      rewriter.create<vector::ShapeCastOp>(loc, resultVectorType, readResult);

  SmallVector<int64_t> writeMaskShape(
      shapeCastOp.getResultVectorType().getShape());
  Operation *write = createTransferWrite(rewriter, loc, shapeCastOp.getResult(),
                                         reifiedRetShapes[0], writeMaskShape);
  newResults.push_back(write->getResult(0));
  return success();
}

/// Vectorize a `tensor::bitcast` to these 3 Ops:
///   vector::TransferReadOp - Reads a vector from the source tensor
///   vector.Bitcast - Bitcast the data based on the target.
///   vector::TransferWriteOp. - Write the result vector back to the destination
///   tensor
LogicalResult lowerTensorBitcastOp(RewriterBase &rewriter,
                                   tensor::BitcastOp bitCastOp,
                                   SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(bitCastOp);

  auto sourceType = bitCastOp.getSource().getType();
  auto sourceShape = sourceType.getShape();
  auto resultType = bitCastOp.getResult().getType();
  auto resultShape = resultType.getShape();

  SmallVector<int64_t> readMaskShape;
  readMaskShape.append(sourceShape.begin(), sourceShape.end());
  Location loc = bitCastOp->getLoc();

  // Read result, mask if necessary. If transferReadOp shape is not equal
  // to shape of source, then a mask is necessary.
  Value readResult = createTransferRead(
      rewriter, loc, bitCastOp->getOperand(0),
      ArrayRef<int64_t>(readMaskShape.begin(), readMaskShape.end()));

  auto resultVectorType =
      VectorType::get(resultShape, resultType.getElementType());
  vector::BitCastOp vectorbitCastOp =
      rewriter.create<vector::BitCastOp>(loc, resultVectorType, readResult);

  SmallVector<int64_t> writeMaskShape(
      vectorbitCastOp.getResultVectorType().getShape());
  llvm::SmallVector<OpFoldResult> destSizes;
  for (auto size : resultShape)
    destSizes.emplace_back(rewriter.getIndexAttr(size));
  auto write =
      createTransferWrite(rewriter, loc, vectorbitCastOp->getResults()[0],
                          destSizes, writeMaskShape);
  newResults.push_back(write->getResults()[0]);
  return success();
}

/// Vectorize a `tensor::concat` to these 3 Ops:
///   Tensor::EmptyOp - The result tensor.
///   Vector::TransferWriteOp - Write the result vector back to the destination
///   tensor.
///   Vector::TransferWriteOp - Write the result vector back to the destination
///   tensor.
LogicalResult lowerTensorConcatOp(RewriterBase &rewriter,
                                  tensor::ConcatOp concatOp,
                                  SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(concatOp);

  Location loc = concatOp.getLoc();
  FailureOr<Value> dest =
      tensor::getOrCreateDestination(rewriter, loc, concatOp->getResult(0));
  if (failed(dest))
    return failure();

  auto empty = dest->getDefiningOp<tensor::EmptyOp>();
  if (!empty)
    return failure();

  // Compute the partial sums for the slice offsets.
  int64_t dim = concatOp.getDim();
  Value dimValue =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dim));

  int64_t rank = concatOp.getResultType().getRank();

  // Construct the chain of insert_slice ops into the destination.
  Value result = *dest;
  Value previous_offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (auto input : concatOp.getInputs()) {

    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, input);
    SmallVector<int64_t> readMaskShape;
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto sourceShape = inputType.getShape();

    readMaskShape.append(sourceShape.begin(), sourceShape.end());
    Value readResult = createTransferRead(
        rewriter, loc, input,
        ArrayRef<int64_t>(readMaskShape.begin(), readMaskShape.end()));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(rank, zero);
    indices[dim] = previous_offset;
    result = rewriter
                 .create<vector::TransferWriteOp>(
                     loc, readResult, result, indices,
                     rewriter.getMultiDimIdentityMap(rank))
                 ->getResults()[0];
    auto dimOp = rewriter.create<tensor::DimOp>(loc, input, dimValue);
    previous_offset =
        rewriter.create<arith::AddIOp>(loc, dimOp, previous_offset);
  }

  newResults.push_back(result);
  return success();
}

LogicalResult convert2TargetOperation(RewriterBase &rewriter, Operation *op) {
  LDBG("Attempting to vectorize:\n" << *op << "\n");

  if (failed(lowerTargetOpPrecondition(op))) {
    std::cout << "FAILED TO LOWER TARGET OP\n" << std::endl;
    LDBG("Vectorization pre-conditions failed\n");
    return failure();
  }

  SmallVector<Value> results;
  auto lowerResult =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<tensor::ExpandShapeOp>([&](auto expandShapeOp) {
            return lowerTensorExpandShapeOp<tensor::ExpandShapeOp>(
                rewriter, expandShapeOp, results);
          })
          .Case<tensor::CollapseShapeOp>([&](auto collapseShapeOp) {
            return lowerTensorExpandShapeOp<tensor::CollapseShapeOp>(
                rewriter, collapseShapeOp, results);
          })
          .Case<tensor::BitcastOp>([&](auto bitCastOp) {
            return lowerTensorBitcastOp(rewriter, bitCastOp, results);
          })
          .Case<tensor::ConcatOp>([&](auto concatOp) {
            return lowerTensorConcatOp(rewriter, concatOp, results);
          })
          .Default([](auto) { return failure(); });

  if (failed(lowerResult)) {
    LDBG("Lower failed\n");
    return failure();
  }

  if (!results.empty())
    rewriter.replaceOp(op, results);
  else
    rewriter.eraseOp(op);

  return success();
}

bool is_required_tensorOp(Operation *operation) {
  return llvm::isa<tensor::ExpandShapeOp>(operation) ||
         llvm::isa<tensor::CollapseShapeOp>(operation) ||
         llvm::isa<tensor::BitcastOp>(operation) ||
         llvm::isa<tensor::ConcatOp>(operation);
}

template <class T = linalg::LinalgOp>
struct OperationConvertTileVectorPass : public RewritePattern {

  explicit OperationConvertTileVectorPass(MLIRContext *context,
                                          bool vectorizeNDExtract = false,
                                          bool flatten1DDepthwiseConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorizeNDExtract(vectorizeNDExtract),
        flatten1DDepthwiseConv(flatten1DDepthwiseConv) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto targetOp = llvm::dyn_cast<T>(op);
    if (!targetOp || !is_innermost_ir(op))
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    return linalg::vectorize(rewriter, op, /*inputVectorSizes=*/{},
                             /*scalableVecDims=*/{}, vectorizeNDExtract,
                             flatten1DDepthwiseConv);
  }

private:
  bool vectorizeNDExtract, flatten1DDepthwiseConv;
};

struct TensorUnpackConvertVectorPass : public RewritePattern {

  explicit TensorUnpackConvertVectorPass(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    tensor::UnPackOp tensorUnPackOp = dyn_cast<tensor::UnPackOp>(op);

    if (!tensorUnPackOp || !is_innermost_ir(op))
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    Value resultValue = op->getResult(0);
    auto resultTy = dyn_cast<RankedTensorType>(resultValue.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    llvm::ArrayRef<int64_t> inputShape = resultTy.getShape();
    std::vector<int64_t> targetVectorSizes = inputShape.vec();
    llvm::SmallVector<bool, 5> targetVecDims(inputShape.size(), false);
    return linalg::vectorize(rewriter, op,
                             /*inputVectorSizes=*/targetVectorSizes,
                             /*scalableVecDims=*/targetVecDims, false, false);
  }
};

struct TensorOpConvertVectorPass : public RewritePattern {

  explicit TensorOpConvertVectorPass(MLIRContext *context,
                                     bool vectorizeExtract = false,
                                     bool flatten1DDepthwiseConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    bool is_target = is_required_tensorOp(op);
    if (!is_target || !is_innermost_ir(op))
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    return convert2TargetOperation(rewriter, op);
  }
};

/// Pass that lower to tile vector.
void populateLowerToTileVectorPatterns(RewritePatternSet &patterns) {
  patterns.add<OperationConvertTileVectorPass<linalg::LinalgOp>,
               OperationConvertTileVectorPass<tensor::PackOp>>(
      patterns.getContext());
  patterns.add<TensorUnpackConvertVectorPass>(patterns.getContext());
  patterns.add<TensorOpConvertVectorPass>(patterns.getContext());
}

struct LowerTileVectorPass
    : public impl::LowerToTileVectorBase<LowerTileVectorPass> {
  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    tensor::ControlFoldFn defaultControlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer && producer->hasOneUse();
    };
    tensor::populateRewriteAsConstantPatterns(patterns, defaultControlFn);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    tensor::populateFoldTensorSubsetOpPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns, true);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    populateLowerToTileVectorPatterns(patterns);
    linalg::populatePadOpVectorizationPatterns(patterns);

    // vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateSinkVectorBroadcastPatterns(patterns);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    auto curOp = getOperation();
    IRRewriter reWriter(curOp);
    DominanceInfo domInfo(curOp);
    eliminateCommonSubExpressions(reWriter, domInfo, curOp);
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileVectorPass() {
  return std::make_unique<LowerTileVectorPass>();
}
} // namespace gc
} // namespace mlir