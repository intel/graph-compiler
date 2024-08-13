//===- LowerTileVectorPass.cpp.cpp - OneDNNGraph To Linalg
// Lowering -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Dialect/Linalgx/LinalgxOps.h"
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

#define IMPLEMENTED_MATMUL                                                     \
  linalgx::BatchReduceMatmulVnniOp, linalgx::MultiBatchMatmulOp,               \
      linalg::BatchReduceMatmulOp, linalgx::Mm2DVnniOp, linalgx::Mm4DVnniOp,   \
      linalg::MatmulOp, linalg::BatchMatmulOp,                                 \
      linalg::BatchMatmulTransposeAOp, linalg::BatchMatmulTransposeBOp,        \
      linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,                  \
      linalg::QuantizedBatchMatmulOp, linalg::QuantizedMatmulOp

#define SUPPORT_TENSOR_OP                                                      \
  tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::BitcastOp,           \
      tensor::ConcatOp

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

static bool isMatchedOperationUsage(Operation *op) {
  if (isa<IMPLEMENTED_MATMUL>(op)) {
    return true;
  }
  // operation produce for matmul can't lower
  if (!isa<linalg::FillOp>(op)) {
    return false;
  }

  for (auto x : op->getUsers()) {
    if (isa<IMPLEMENTED_MATMUL>(x)) {
      return true;
    }
  }

  return false;
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
lowerCollapseShapeOpPrecondition(tensor::CollapseShapeOp collapseOp) {
  auto isShapeStatic = [](Value v) {
    auto type = mlir::dyn_cast<ShapedType>(v.getType());
    if (!type) {
      LDBG("Operation type error: " << v << "\n");
      return false;
    }
    return type.hasStaticShape();
  };
  if (!isShapeStatic(collapseOp->getResults()[0])) {
    LDBG("Output shape must be static: " << collapseOp << "\n");
    return failure();
  }
  if (!isShapeStatic(collapseOp.getSrc())) {
    LDBG("Input shape must be static: " << collapseOp << "\n");
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

Operation *createWriteOrMaskedWrite(OpBuilder &builder, Location loc,
                                    Value input,
                                    SmallVector<OpFoldResult> destSizes,
                                    ArrayRef<int64_t> inputVectorSizes,
                                    bool useInBoundsInsteadOfMasking) {

  auto inputType = cast<VectorType>(input.getType());
  Value dest = builder.create<tensor::EmptyOp>(loc, destSizes,
                                               inputType.getElementType());
  int64_t rank = cast<ShapedType>(dest.getType()).getRank();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto destShape = cast<ShapedType>(dest.getType()).getShape();
  SmallVector<bool> inBoundsVal(rank, true);
  if (useInBoundsInsteadOfMasking) {
    // Update the inBounds attribute.
    for (unsigned i = 0; i < rank; i++)
      inBoundsVal[i] = (destShape[i] == inputVectorSizes[i]) &&
                       !ShapedType::isDynamic(destShape[i]);
  }
  Operation *write = builder.create<vector::TransferWriteOp>(
      loc,
      /*vector=*/input,
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/inBoundsVal);
  assert(llvm::none_of(
             destShape.drop_front(inputVectorSizes.size()),
             [](int64_t size) { return size == ShapedType::kDynamic; }) &&
         "Only dims aligned with inputVectorSizes may be dynamic");
  if (useInBoundsInsteadOfMasking)
    return write;
  bool needMaskForWrite = !llvm::equal(
      inputVectorSizes, destShape.take_front(inputVectorSizes.size()));
  if (needMaskForWrite) {
    SmallVector<int64_t> writeMaskShape;
    writeMaskShape.append(inputVectorSizes.begin(), inputVectorSizes.end());
    writeMaskShape.append(destShape.begin() + inputVectorSizes.size(),
                          destShape.end());
    auto writeMaskType = VectorType::get(writeMaskShape, builder.getI1Type());
    Value maskForWrite =
        builder.create<vector::CreateMaskOp>(loc, writeMaskType, destSizes);
    write = mlir::vector::maskOperation(builder, write, maskForWrite);
  }
  return write;
}

Value createReadOrMaskedRead(OpBuilder &builder, Location loc, Value source,
                             ArrayRef<int64_t> readShape, Value padValue,
                             bool useInBoundsInsteadOfMasking) {
  assert(llvm::none_of(readShape,
                       [](int64_t s) { return s == ShapedType::kDynamic; }) &&
         "expected static shape");
  auto sourceShapedType = cast<ShapedType>(source.getType());
  auto sourceShape = sourceShapedType.getShape();
  assert(sourceShape.size() == readShape.size() && "expected same ranks.");
  auto maskType = VectorType::get(readShape, builder.getI1Type());
  auto vectorType = VectorType::get(readShape, padValue.getType());
  assert(padValue.getType() == sourceShapedType.getElementType() &&
         "expected same pad element type to match source element type");
  int64_t readRank = readShape.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<bool> inBoundsVal(readRank, true);
  if (useInBoundsInsteadOfMasking) {
    // Update the inBounds attribute.
    for (unsigned i = 0; i < readRank; i++)
      inBoundsVal[i] = (sourceShape[i] == readShape[i]) &&
                       !ShapedType::isDynamic(sourceShape[i]);
  }
  auto transferReadOp = builder.create<vector::TransferReadOp>(
      loc,
      /*vectorType=*/vectorType,
      /*source=*/source,
      /*indices=*/SmallVector<Value>(readRank, zero),
      /*padding=*/padValue,
      /*inBounds=*/inBoundsVal);

  if (llvm::equal(readShape, sourceShape) || useInBoundsInsteadOfMasking)
    return transferReadOp;
  SmallVector<OpFoldResult> mixedSourceDims =
      tensor::getMixedSizes(builder, loc, source);
  Value mask =
      builder.create<vector::CreateMaskOp>(loc, maskType, mixedSourceDims);
  return mlir::vector::maskOperation(builder, transferReadOp, mask)
      ->getResult(0);
}

/// Vectorize a `tensor::expandshape` to these 3 Ops:
///   Vector::TransferReadOp - Reads a vector from the source tensor
///   ShapeCastOp - Reshape the data based on the target.
///   vector::TransferWriteOp. - Write the result vector back to the destination
///   tensor
template <class T>
LogicalResult lowerTensorExpandShapeOp(RewriterBase &rewriter,
                                       Operation *inputOp,
                                       SmallVectorImpl<Value> &newResults) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(inputOp);
  auto src = inputOp->getOperand(0);
  auto srcType = mlir::dyn_cast<ShapedType>(src.getType());
  auto result = inputOp->getResults()[0];
  auto resultType = mlir::dyn_cast<ShapedType>(result.getType());

  ArrayRef<int64_t> resultShape = resultType.getShape();
  Location loc = inputOp->getLoc();

  // read
  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(srcType.getElementType()));
  Value readResult = createReadOrMaskedRead(
      rewriter, loc, src, srcType.getShape(), padValue, false);

  auto shapeCastType =
      VectorType::get(resultType.getShape(), resultType.getElementType());
  vector::ShapeCastOp shapeCastOp =
      rewriter.create<vector::ShapeCastOp>(loc, shapeCastType, readResult);

  // write
  SmallVector<OpFoldResult> destSizes;
  for (auto size : resultShape) {
    destSizes.emplace_back(rewriter.getIndexAttr(size));
  }
  Operation *write =
      createWriteOrMaskedWrite(rewriter, loc, shapeCastOp->getResults()[0],
                               destSizes, resultShape, false);
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
  auto resultType = bitCastOp.getResult().getType();
  auto resultShape = resultType.getShape();
  Location loc = bitCastOp->getLoc();

  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(sourceType.getElementType()));
  Value readResult = createReadOrMaskedRead(
      rewriter, loc, bitCastOp.getSource(), resultShape, padValue, false);

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
      createWriteOrMaskedWrite(rewriter, loc, vectorbitCastOp->getResult(0),
                               destSizes, resultShape, false);
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
  auto srcType =
      mlir::dyn_cast<RankedTensorType>(concatOp->getResultTypes()[0]);
  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(srcType.getElementType()));

  // Construct the chain of insert_slice ops into the destination.
  Value result = *dest;
  Value previous_offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (auto input : concatOp.getInputs()) {

    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, input);
    SmallVector<int64_t> readMaskShape;
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto sourceShape = inputType.getShape();

    readMaskShape.append(sourceShape.begin(), sourceShape.end());
    Value readResult = createReadOrMaskedRead(rewriter, loc, input, sourceShape,
                                              padValue, false);
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
  return isa<SUPPORT_TENSOR_OP>(operation);
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

    // linalg.fill + linalgx.batch_mutmul should not be lower to vector
    // because these two operation is needed by brgemm optimization.
    if (isMatchedOperationUsage(op)) {
      return rewriter.notifyMatchFailure(
          op, "linalg.fill + linalgx.batch_matmul can't do lowering.");
    }

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

struct EliminateWriteReadOpPass
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceOp = op->getOperand(0).getDefiningOp();
    if (isa_and_nonnull<vector::TransferWriteOp>(sourceOp)) {
      rewriter.replaceOp(op, sourceOp->getOperand(0));
      return success();
    }
    return failure();
  }
};

void eliminateWriteReadOperation(Operation *op) {
  if (!isa_and_nonnull<vector::TransferReadOp>(op)) {
    return;
  }
  auto sourceOp = op->getOperand(0).getDefiningOp();
  if (isa_and_nonnull<vector::TransferWriteOp>(sourceOp)) {
    IRRewriter rewriter(op);
    rewriter.replaceOp(op, sourceOp->getOperand(0));
  }
}

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
    auto funcOp = getOperation();

    tensor::ControlFoldFn defaultControlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer && producer->hasOneUse();
    };
    // some operation convert as constant, this pattern can help us to improve
    // the performance
    // tensor::populateRewriteAsConstantPatterns(patterns, defaultControlFn);
    // remove unnessary operation
    // tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    // tensor::populateFoldTensorSubsetOpPatterns(patterns);
    // tensor::populateFoldTensorEmptyPatterns(patterns, true);
    // tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    populateLowerToTileVectorPatterns(patterns);
    linalg::populatePadOpVectorizationPatterns(patterns);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns), config);
    // error case:
    // due to insert slice tensor<1x32xf32> to tensor<1x128x1x32xf32>
    // linalg.copy : <1x32xf32>
    // -> transfer_write : permutation map = (d0, d1, d2, d3) -> (d0, d3)
    // Inorder to avoid the fold greedily bug (fold wrong permution map for the
    // transfer_write operation). Give it the new full IR to fold second time
    // can fold correctly.
    RewritePatternSet secondPattern(ctx);
    // secondPattern.add<EliminateWriteReadOpPass>(patterns.getContext());
    // ensure read and write on last dimension
    vector::populateVectorTransferPermutationMapLoweringPatterns(secondPattern);
    // remove unnessary broadcast operation
    vector::populateSinkVectorBroadcastPatterns(secondPattern);
    // vector::TransferReadOp::getCanonicalizationPatterns(secondPattern, ctx);
    // vector::TransferWriteOp::getCanonicalizationPatterns(secondPattern, ctx);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(secondPattern);

    (void)applyPatternsAndFoldGreedily(funcOp, std::move(secondPattern));
    // DominanceInfo domInfo;
    // IRRewriter rewriter(funcOp);
    // eliminateCommonSubExpressions(rewriter, domInfo, funcOp);
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileVectorPass() {
  return std::make_unique<LowerTileVectorPass>();
}
} // namespace gc
} // namespace mlir