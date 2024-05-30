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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

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

  // if (llvm::any_of(expandOp.getReassociation(), [](ArrayAttr res) {
  //       if (llvm::any_of(res, [](Attribute x) {
  //             return !getConstantIntValue(x).has_value();
  //           })) {
  //         return false;
  //       }
  //       return true;
  //     })) {
  //   LDBG("Reassociation must be constant: " << expandOp << "\n");
  //   return failure();
  // }

  return success();
}

LogicalResult lowerTargetOpPrecondition(Operation *op,
                                        ArrayRef<int64_t> inputVectorSizes,
                                        ArrayRef<bool> inputScalableVecDims,
                                        bool vectorizeNDExtract,
                                        bool flatten1DDepthwiseConv) {

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<tensor::ExpandShapeOp>([&](auto expandShapeOp) {
        return lowerExpandOpPrecondition(expandShapeOp);
      })
      .Case<tensor::CollapseShapeOp>(
          [&](auto collapseShapeOp) { return success(); })
      .Case<tensor::BitcastOp>([&](auto collapseShapeOp) { return success(); })
      .Case<tensor::ConcatOp>([&](auto concatOp) { return success(); })
      .Default([](auto) { return failure(); });
}

/// Create a TransferReadOp from `source` with static shape `readShape`.
Value createTransferRead(OpBuilder &builder, Location loc, Value source,
                         ArrayRef<int64_t> readShape, Value padValue) {
  assert(llvm::none_of(readShape,
                       [](int64_t s) { return s == ShapedType::kDynamic; }));
  assert(source && " source null.");
  auto sourceShape = dyn_cast<ShapedType>(source.getType()).getShape();
  assert(sourceShape.size() == readShape.size());
  auto vectorType = VectorType::get(readShape, padValue.getType());
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

/// Given an input, the mixed destSizes, and the vector sizes for vectorization,
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
         "Only dims aligned with inputVectorSizes may be dynamic");
  return write;
}

/// Vectorize a `tensor::expandshape` to these 3 Ops:
///   Vector::TransferReadOp - Reads a vector from the source tensor
///   ShapeCastOp - Reshape the data based on the target.
///   vector::TransferWriteOp. - Write the result vector back to the destination
///   tensor
template <class T>
LogicalResult lowerTensorExpandShapeOp(RewriterBase &rewriter, T expandShapeOp,
                                       ArrayRef<int64_t> inputVectorSizes,
                                       SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(expandShapeOp);

  RankedTensorType expandShapeTensorType = expandShapeOp.getSrcType();

  SmallVector<int64_t> readMaskShape(inputVectorSizes.begin(),
                                     inputVectorSizes.end());
  ArrayRef<int64_t> sourceShape = expandShapeTensorType.getShape();
  ArrayRef<int64_t> resultShape = expandShapeOp.getResultType().getShape();

  readMaskShape.append(sourceShape.begin() + inputVectorSizes.size(),
                       sourceShape.end());

  ReifiedRankedShapedTypeDims reifiedRetShapes;
  LogicalResult status =
      cast<ReifyRankedShapedTypeOpInterface>(expandShapeOp.getOperation())
          .reifyResultShapes(rewriter, reifiedRetShapes);
  if (status.failed()) {
    LDBG("Unable to reify result shapes of " << expandShapeOp << "\n");
    return failure();
  }
  Location loc = expandShapeOp->getLoc();

  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(expandShapeTensorType.getElementType()));

  // Read result, mask if necessary. If transferReadOp shape is not equal
  // to shape of source, then a mask is necessary.
  Value readResult = createTransferRead(
      rewriter, loc, expandShapeOp.getSrc(),
      ArrayRef<int64_t>(readMaskShape.begin(), readMaskShape.end()), padValue);

  auto resultVectorType =
      VectorType::get(resultShape, expandShapeTensorType.getElementType());
  vector::ShapeCastOp shapeCastOp =
      rewriter.create<vector::ShapeCastOp>(loc, resultVectorType, readResult);

  SmallVector<int64_t> writeMaskShape(
      expandShapeOp.getResultType().hasStaticShape()
          ? inputVectorSizes
          : shapeCastOp.getResultVectorType().getShape());
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
                                   ArrayRef<int64_t> inputVectorSizes,
                                   SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(bitCastOp);

  auto sourceType = bitCastOp.getSource().getType();
  auto sourceShape = sourceType.getShape();
  auto resultType = bitCastOp.getResult().getType();
  auto resultShape = resultType.getShape();

  SmallVector<int64_t> readMaskShape(inputVectorSizes.begin(),
                                     inputVectorSizes.end());

  readMaskShape.append(sourceShape.begin() + inputVectorSizes.size(),
                       sourceShape.end());

  Location loc = bitCastOp->getLoc();

  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(sourceType.getElementType()));

  // Read result, mask if necessary. If transferReadOp shape is not equal
  // to shape of source, then a mask is necessary.
  Value readResult = createTransferRead(
      rewriter, loc, bitCastOp->getOperand(0),
      ArrayRef<int64_t>(readMaskShape.begin(), readMaskShape.end()), padValue);

  auto resultVectorType =
      VectorType::get(resultShape, resultType.getElementType());
  vector::BitCastOp vectorbitCastOp =
      rewriter.create<vector::BitCastOp>(loc, resultVectorType, readResult);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(resultType.getRank(), zero);
  Value dest = rewriter.create<tensor::EmptyOp>(loc, resultShape,
                                                resultType.getElementType());
  Operation *write = rewriter.create<vector::TransferWriteOp>(
      loc, vectorbitCastOp, dest, indices,
      rewriter.getMultiDimIdentityMap(resultType.getRank()));
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
                                  ArrayRef<int64_t> inputVectorSizes,
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
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

  // Construct the chain of insert_slice ops into the destination.
  Value result = *dest;
  Value previous_offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (auto input : concatOp.getInputs()) {

    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, input);
    SmallVector<int64_t> readMaskShape(inputVectorSizes.begin(),
                                       inputVectorSizes.end());
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto sourceShape = inputType.getShape();

    readMaskShape.append(sourceShape.begin() + inputVectorSizes.size(),
                         sourceShape.end());
    auto padValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inputType.getElementType()));
    Value readResult = createTransferRead(
        rewriter, loc, input,
        ArrayRef<int64_t>(readMaskShape.begin(), readMaskShape.end()),
        padValue);
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

/// Emit a suitable vector form for an operation. If provided,
/// `inputVectorSizes` are used to vectorize this operation.
/// `inputVectorSizes` must match the rank of the iteration space of the
/// operation and the input vector sizes must be greater than or equal to
/// their counterpart iteration space sizes, if static. `inputVectorShapes`
/// also allows the vectorization of operations with dynamic shapes.
LogicalResult convert2TargetOperation(RewriterBase &rewriter, Operation *op,
                                      ArrayRef<int64_t> inputVectorSizes,
                                      ArrayRef<bool> inputScalableVecDims,
                                      bool vectorizeNDExtract,
                                      bool flatten1DDepthwiseConv) {
  LDBG("Attempting to vectorize:\n" << *op << "\n");
  LDBG("Input vector sizes: ");
  LLVM_DEBUG(llvm::interleaveComma(inputVectorSizes, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");
  LDBG("Input scalable vector dims: ");
  LLVM_DEBUG(llvm::interleaveComma(inputScalableVecDims, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (failed(lowerTargetOpPrecondition(op, inputVectorSizes,
                                       inputScalableVecDims, vectorizeNDExtract,
                                       flatten1DDepthwiseConv))) {
    LDBG("Vectorization pre-conditions failed\n");
    return failure();
  }

  SmallVector<Value> results;
  auto lowerResult =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<tensor::ExpandShapeOp>([&](auto expandShapeOp) {
            return lowerTensorExpandShapeOp(rewriter, expandShapeOp,
                                            inputVectorSizes, results);
          })
          .Case<tensor::CollapseShapeOp>([&](auto collapseShapeOp) {
            return lowerTensorExpandShapeOp(rewriter, collapseShapeOp,
                                            inputVectorSizes, results);
          })
          .Case<tensor::BitcastOp>([&](auto bitCastOp) {
            return lowerTensorBitcastOp(rewriter, bitCastOp, inputVectorSizes,
                                        results);
          })
          .Case<tensor::ConcatOp>([&](auto concatOp) {
            return lowerTensorConcatOp(rewriter, concatOp, inputVectorSizes,
                                       results);
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

struct LinalgConvertTileVectorPass : public RewritePattern {

  explicit LinalgConvertTileVectorPass(MLIRContext *context,
                                       bool vectorizeExtract = false,
                                       bool flatten1DDepthwiseConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || !is_innermost_ir(op))
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    return linalg::vectorize(rewriter, op, /*inputVectorSizes=*/{},
                             /*scalableVecDims=*/{}, true, false);
  }
};

struct TensorPackConvertVectorPass : public RewritePattern {

  explicit TensorPackConvertVectorPass(MLIRContext *context,
                                       bool vectorizeExtract = false,
                                       bool flatten1DDepthwiseConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    tensor::PackOp tensorPackOp = dyn_cast<tensor::PackOp>(op);
    if (!tensorPackOp || !is_innermost_ir(op))
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    return linalg::vectorize(rewriter, op, /*inputVectorSizes=*/{},
                             /*scalableVecDims=*/{}, true, false);
  }
};

struct TensorUnpackConvertVectorPass : public RewritePattern {

  explicit TensorUnpackConvertVectorPass(MLIRContext *context,
                                         bool vectorizeExtract = false,
                                         bool flatten1DDepthwiseConv = false)
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
    llvm::SmallVector<bool, 8> targetVecDims(inputShape.size(), false);
    return linalg::vectorize(rewriter, op,
                             /*inputVectorSizes=*/targetVectorSizes,
                             /*scalableVecDims=*/targetVecDims, true, false);
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

    return convert2TargetOperation(rewriter, op, /*inputVectorSizes=*/{},
                                   /*scalableVecDims=*/{}, true, false);
  }
};

/// Pass that lower to tile vector.
void populateLowerToTileVectorPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgConvertTileVectorPass>(patterns.getContext());
  patterns.add<TensorUnpackConvertVectorPass>(patterns.getContext());
  patterns.add<TensorPackConvertVectorPass>(patterns.getContext());
  patterns.add<TensorOpConvertVectorPass>(patterns.getContext());
}

struct LowerTileVectorPass
    : public impl::LowerToTileVectorBase<LowerTileVectorPass> {
  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    tensor::populateRewriteAsConstantPatterns(patterns);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    populateLowerToTileVectorPatterns(patterns);
    linalg::populatePadOpVectorizationPatterns(patterns);
    tensor::populateFoldTensorSubsetOpPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns, true);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    vector::VectorTransformsOptions vectorTransformOptions;
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vectorTransformOptions.vectorMultiReductionLowering);
    // vector::populateVectorShapeCastLoweringPatterns(patterns);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileVectorPass() {
  return std::make_unique<LowerTileVectorPass>();
}
} // namespace gc
} // namespace mlir