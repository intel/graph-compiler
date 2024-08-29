//===-- LowerToTileVector.cpp - Lower Op to vector --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace gc {

#define GEN_PASS_DEF_LOWERTOTILEVECTOR
#include "gc/Transforms/Passes.h.inc"
namespace {
#define DEBUG_TYPE "lower-to-tile-vector-pass"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define SAFE_EXPAND(X) X
#define LDBG(X) LLVM_DEBUG(DBGS() << SAFE_EXPAND(X) << "\n")

#define IMPLEMENTED_MATMUL                                                     \
  linalgx::BatchReduceMatmulVnniOp, linalgx::MultiBatchMatmulOp,               \
      linalg::BatchReduceMatmulOp, linalgx::Mm2DVnniOp, linalgx::Mm4DVnniOp,   \
      linalg::MatmulOp, linalg::BatchMatmulOp,                                 \
      linalg::BatchMatmulTransposeAOp, linalg::BatchMatmulTransposeBOp,        \
      linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,                  \
      linalg::QuantizedBatchMatmulOp, linalg::QuantizedMatmulOp

#define SUPPORT_TENSOR_OP                                                      \
  tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ConcatOp

template <typename T, typename U>
struct decay_equiv : std::is_same<typename std::decay<T>::type, U>::type {};

static inline bool isRequiredTensorOp(Operation *operation) {
  return isa<SUPPORT_TENSOR_OP>(operation);
}

/// matmul operation or fill + matmul operation
static bool isMatchedOperationSet(Operation *op) {
  if (isa<IMPLEMENTED_MATMUL>(op))
    return true;

  // Operation produce for matmul can't lower.
  // Currently only the fill operation need to check this.
  if (!isa<linalg::FillOp>(op))
    return false;

  return llvm::any_of(op->getUsers(),
                      [](Operation *x) { return isa<IMPLEMENTED_MATMUL>(x); });
}

static bool isContainsDynamicSize(ArrayRef<int64_t> sizes) {
  return llvm::any_of(sizes,
                      [](int64_t x) { return x == ShapedType::kDynamic; });
}

/// Reshape operation like expand_shape process helper class.
/// Inorder to avoid pass too many parameters to function.
struct ReshapeVectorizeHelper {
  /// The transfer_read operation read result. We calculate this shape based on
  /// the specified input vector size.
  SmallVector<int64_t> srcVectorizedShape;
  /// The ratio of the size of a certain dimension specified by the user to the
  /// size of the op result dimension
  llvm::SmallDenseMap<int64_t, int64_t> shapeScales;
  /// operation result shape
  SmallVector<int64_t> resultShape;
  /// input operand shape
  SmallVector<int64_t> srcShape;

  ReshapeVectorizeHelper() = default;
  ReshapeVectorizeHelper(ArrayRef<int64_t> srcVectorizedShape,
                         llvm::SmallDenseMap<int64_t, int64_t> &shapeScales,
                         ArrayRef<int64_t> resultShape,
                         ArrayRef<int64_t> srcShape)
      : srcVectorizedShape(srcVectorizedShape), shapeScales(shapeScales),
        resultShape(resultShape), srcShape(srcShape) {}

  /// Get the magnification factor of dimension size of the shape
  void getScalesDim(ArrayRef<int64_t> inputVectorSizes);
};

void ReshapeVectorizeHelper::getScalesDim(ArrayRef<int64_t> inputVectorSizes) {
  for (auto [idx, vs] : llvm::enumerate(inputVectorSizes)) {
    if (vs != resultShape[idx])
      shapeScales[idx] = vs / resultShape[idx];
  }
}

/// Get proper input vector size for the operation.
/// Currently only expandshape and collaspeshape need to handle this.
template <class T, typename = typename std::enable_if<
                       decay_equiv<T, tensor::ExpandShapeOp>::value ||
                           decay_equiv<T, tensor::CollapseShapeOp>::value,
                       T>>
void getReshapeOperationVectorizeShape(ReshapeVectorizeHelper &reshapeHelper) {
  reshapeHelper.srcVectorizedShape.clear();
  bool isCollapseOp = decay_equiv<T, tensor::CollapseShapeOp>::value;
  int64_t cur = 1, resultIdx = 0;

  for (auto [srcIdx, ss] : llvm::enumerate(reshapeHelper.srcShape)) {
    cur *= ss;
    // collapse operation need to keep each of the original shape.
    if (isCollapseOp)
      reshapeHelper.srcVectorizedShape.emplace_back(ss);

    // Only when the scaled dimension appears, it is necessary to infer the
    // corresponding multiple of the src shape.
    if (cur != reshapeHelper.resultShape[resultIdx])
      continue;

    // expand_shape op only need to keep the total vectorized result shape.
    if (!isCollapseOp)
      reshapeHelper.srcVectorizedShape.emplace_back(cur);

    // The corresponding dimension is expanded by the multiple specified by the
    // user.
    if (isCollapseOp and reshapeHelper.shapeScales.count(resultIdx))
      reshapeHelper.srcVectorizedShape.back() *=
          reshapeHelper.shapeScales[resultIdx];
    if (!isCollapseOp and reshapeHelper.shapeScales.count(srcIdx))
      reshapeHelper.srcVectorizedShape.back() *=
          reshapeHelper.shapeScales[srcIdx];

    cur = 1;
    resultIdx++;
  }
}

/// Need to check whether the reassociation, input, output and input vectorize
/// size are valid.
template <class T, typename = typename std::enable_if<
                       decay_equiv<T, tensor::ExpandShapeOp>::value ||
                           decay_equiv<T, tensor::CollapseShapeOp>::value,
                       T>>
LogicalResult
lowerReshapeOpPrecondition(T reshapeOp,
                           ArrayRef<int64_t> inputVectorSizes = {}) {

  Type resultType = reshapeOp->getResultTypes()[0];
  auto resultShapeType = cast<ShapedType>(resultType);
  RankedTensorType srcShapeType = reshapeOp.getSrcType();

  // check reassociation
  SmallVector<int64_t> associateIndices;

  for (const Attribute &attr : reshapeOp.getReassociation())
    llvm::transform(
        cast<ArrayAttr>(attr), std::back_inserter(associateIndices),
        [](Attribute indice) { return cast<IntegerAttr>(indice).getInt(); });

  if (isContainsDynamicSize(associateIndices)) {
    LDBG("Reassociation must be static: " << reshapeOp << "\n");
    return failure();
  }

  // check input and output shape
  bool isStaticInputOutput =
      resultShapeType.hasStaticShape() && srcShapeType.hasStaticShape();
  if (!isStaticInputOutput) {
    LDBG("Input and output shape must be static: " << reshapeOp << "\n");
    return failure();
  }
  // ensure specify input vector size is valid
  if (!inputVectorSizes.empty() &&
      failed(vector::isValidMaskedInputVector(resultShapeType.getShape(),
                                              inputVectorSizes)))
    return failure();

  if (!llvm::all_of(llvm::zip(resultShapeType.getShape(), inputVectorSizes),
                    [](std::tuple<int64_t, int64_t> sizePair) {
                      int64_t staticSize = std::get<0>(sizePair);
                      int64_t inputSize = std::get<1>(sizePair);
                      return inputSize % staticSize == 0;
                    })) {
    LDBG("Input vector sizes must be an integer multiple or equal to "
         "space static sizes");
    return failure();
  }

  return success();
}

LogicalResult
lowerConcatOpPrecondition(tensor::ConcatOp concatOp,
                          ArrayRef<int64_t> inputVectorSizes = {}) {
  if (!inputVectorSizes.empty())
    LDBG("Concat operation does not support specify inputVectorSizes: "
         << concatOp << "\n");

  // check input operand shape type
  if (llvm::any_of(concatOp.getOperandTypes(), [](Type x) {
        return not cast<ShapedType>(x).hasStaticShape();
      })) {
    LDBG("Type must be static: " << concatOp << "\n");
    return failure();
  }
  // check valid dimension
  uint64_t dim = concatOp.getDim();
  if (dim >= (uint64_t)concatOp.getResultType().getRank()) {
    LDBG("Invalid dim: " << concatOp << "\n");
    return failure();
  }

  return success();
}

LogicalResult lowerTargetOpPrecondition(Operation *op,
                                        ArrayRef<int64_t> inputVectorSizes) {

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<tensor::ExpandShapeOp>([&](auto expandShapeOp) {
        return lowerReshapeOpPrecondition<tensor::ExpandShapeOp>(
            expandShapeOp, inputVectorSizes);
      })
      .Case<tensor::CollapseShapeOp>([&](auto collapseShapeOp) {
        return lowerReshapeOpPrecondition<tensor::CollapseShapeOp>(
            collapseShapeOp, inputVectorSizes);
      })
      .Case<tensor::ConcatOp>([&](auto concatOp) {
        return lowerConcatOpPrecondition(concatOp, inputVectorSizes);
      })
      .Default([](auto) { return failure(); });
}

Operation *createWriteOrMaskedWrite(OpBuilder &builder, Location loc,
                                    Value input,
                                    ArrayRef<OpFoldResult> destSizes,
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
  return vector::maskOperation(builder, transferReadOp, mask)->getResult(0);
}

/// Vectorize a `tensor::expandshape` to these 3 Ops:
///   Vector::TransferReadOp - Reads a vector from the source tensor
///   ShapeCastOp - Reshape the data based on the target.
///   vector::TransferWriteOp. - Write the result vector back to the destination
///   tensor
template <class T>
LogicalResult lowerTensorReshapeOp(RewriterBase &rewriter, Operation *inputOp,
                                   SmallVectorImpl<Value> &newResults,
                                   ArrayRef<int64_t> inputVectorSizes) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(inputOp);

  auto src = inputOp->getOperand(0);
  auto srcType = cast<ShapedType>(src.getType());
  OpResult result = inputOp->getResults()[0];
  auto resultType = cast<ShapedType>(result.getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  Location loc = inputOp->getLoc();

  SmallVector<int64_t> srcVectorizedShape(srcType.getRank());
  llvm::SmallDenseMap<int64_t, int64_t> shapeScales;
  ReshapeVectorizeHelper reshapeHelper(srcVectorizedShape, shapeScales,
                                       resultShape, srcShape);

  srcVectorizedShape.assign(srcShape.begin(), srcShape.end());
  if (!inputVectorSizes.empty()) {
    reshapeHelper.getScalesDim(inputVectorSizes);
    getReshapeOperationVectorizeShape<T>(reshapeHelper);
  }
  // generate read operation
  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(srcType.getElementType()));
  Value readResult = vector::createReadOrMaskedRead(
      rewriter, loc, src,
      inputVectorSizes.empty() ? srcType.getShape() : srcVectorizedShape,
      padValue, true);

  auto shapeCastType =
      VectorType::get(inputVectorSizes.empty() ? resultShape : inputVectorSizes,
                      resultType.getElementType());
  vector::ShapeCastOp shapeCastOp =
      rewriter.create<vector::ShapeCastOp>(loc, shapeCastType, readResult);

  // generate write operation
  SmallVector<OpFoldResult> destSizes(resultShape.size());
  llvm::transform(resultShape, std::begin(destSizes), [&rewriter](size_t size) {
    return rewriter.getIndexAttr(size);
  });

  Operation *write = createWriteOrMaskedWrite(
      rewriter, loc, shapeCastOp->getResults()[0], destSizes,
      inputVectorSizes.empty() ? resultShape : inputVectorSizes, true);
  newResults.push_back(write->getResult(0));
  return success();
}

/// Mainly Vectorize a `tensor::concat` to these Ops:
/// Tensor::EmptyOp - The result tensor.
/// Vector::TransferReadOp - Reads a vector from the source tensor
/// Vector::TransferWriteOp - Write the result vector back to the destination
/// tensor. Repeat the read and write operation until all input tensor are
/// completed.
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
  auto srcType = cast<ShapedType>(concatOp->getResultTypes()[0]);
  auto padValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(srcType.getElementType()));

  // Construct the chain of insert_slice ops into the destination.
  Value result = *dest;
  Value previous_offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (auto input : concatOp.getInputs()) {

    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, input);
    auto inputType = cast<ShapedType>(input.getType());

    Value readResult = createReadOrMaskedRead(
        rewriter, loc, input, inputType.getShape(), padValue, true);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(rank, zero);
    // update write position
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

/// Lowring some tensor operation
LogicalResult convert2TargetOperation(RewriterBase &rewriter, Operation *op,
                                      ArrayRef<int64_t> inputVectorSizes = {}) {
  LDBG("Attempting to vectorize:\n" << *op << "\n");
  LLVM_DEBUG(llvm::interleaveComma(inputVectorSizes, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (failed(lowerTargetOpPrecondition(op, inputVectorSizes))) {
    LDBG("Vectorization pre-conditions failed\n");
    return failure();
  }

  SmallVector<Value> results;
  auto lowerResult =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<tensor::ExpandShapeOp>([&](auto expandShapeOp) {
            return lowerTensorReshapeOp<tensor::ExpandShapeOp>(
                rewriter, expandShapeOp, results, inputVectorSizes);
          })
          .Case<tensor::CollapseShapeOp>([&](auto collapseShapeOp) {
            return lowerTensorReshapeOp<tensor::CollapseShapeOp>(
                rewriter, collapseShapeOp, results, inputVectorSizes);
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

/// Use the methods provided by the MLIR community, mainly to lower linalg
/// related ops.
template <class T = linalg::LinalgOp>
struct OperationConvertTileVectorPass : public RewritePattern {

private:
  /// specify vectorize size
  SmallVector<int64_t, 5> inputVectorSizes;
  /// keep those parameters for future use
  bool vectorizeNDExtract, flatten1DDepthwiseConv;

public:
  explicit OperationConvertTileVectorPass(
      MLIRContext *context, ArrayRef<int64_t> inputVectorSizes = {},
      bool vectorizeNDExtract = false, bool flatten1DDepthwiseConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        inputVectorSizes(inputVectorSizes),
        vectorizeNDExtract(vectorizeNDExtract),
        flatten1DDepthwiseConv(flatten1DDepthwiseConv) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto targetOp = dyn_cast<T>(op);
    if (!targetOp)
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    // linalg.fill + linalgx.batch_mutmul should not be lower to vector
    // because these two operation is needed by brgemm kernel.
    if (isMatchedOperationSet(op))
      return rewriter.notifyMatchFailure(
          op, "linalg.fill + linalg.matmul can't do lowering.");

    SmallVector<bool> scalableVecDims(inputVectorSizes.size(), false);
    if (failed(linalg::vectorize(rewriter, op,
                                 /*inputVectorSizes=*/inputVectorSizes,
                                 /*inputScalableVecDims=*/scalableVecDims,
                                 vectorizeNDExtract, flatten1DDepthwiseConv)))
      return rewriter.notifyMatchFailure(op, "Fail to vectorize.");

    return success();
  }
};

/// Lower tensor.unpack operation to vector.
///
/// The reason why we don't use `OperationConvertTileVectorPass` is we
/// need to specify input vector size due to unpack operation does not support
/// empty vector size. It's logic is not consistent with other tensor operation.
/// It would be better we split this process logic as a standalone class to
/// notify unpack operation is not support empty vector size. We need to support
/// it like other operation in the future.
///
/// TODO: Need to support upstream to handle empty vector size. Currently
/// upstream folks don't allow me to do this. It's weird, I can't find reason.
struct TensorUnpackConvertVectorPass : public RewritePattern {

  explicit TensorUnpackConvertVectorPass(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto tensorUnPackOp = dyn_cast<tensor::UnPackOp>(op);
    if (!tensorUnPackOp)
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    auto resultTy = cast<ShapedType>(op->getResultTypes()[0]);
    // TODO: Need to support upstream to handle empty vector size. Currently
    // upstream folks don't allow me to do this.
    ArrayRef<int64_t> inputShape = resultTy.getShape();
    SmallVector<bool, 5> targetVecDims(inputShape.size(), false);

    if (failed(linalg::vectorize(rewriter, op,
                                 /*inputVectorSizes=*/inputShape.vec(),
                                 /*inputScalableVecDims=*/targetVecDims, false,
                                 false)))
      return rewriter.notifyMatchFailure(op, "Fail to vectorize.");

    return success();
  }
};

/// Some tensor operation lowering to vector.
///
/// Currently support expand_shape, collapse_shape and concat_shape.
/// May need support other operation in the future.
struct TensorOpConvertVectorPass : public RewritePattern {
private:
  SmallVector<int64_t> inputVectorSizes;

public:
  explicit TensorOpConvertVectorPass(MLIRContext *context,
                                     ArrayRef<int64_t> inputVectorSizes = {})
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        inputVectorSizes(inputVectorSizes) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    bool is_target = isRequiredTensorOp(op);
    if (!is_target)
      return rewriter.notifyMatchFailure(op, "Not expected operations.");

    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
      return rewriter.notifyMatchFailure(op, "Don't need to lower this op.");
    }

    if (failed(convert2TargetOperation(rewriter, op, inputVectorSizes)))
      return rewriter.notifyMatchFailure(op, "Fail to vectorize.");

    return success();
  }
};

/// Patterns that lower to tile (virtual) vector.
void populateLowerToTileVectorPatterns(RewritePatternSet &patterns) {
  patterns.add<OperationConvertTileVectorPass<linalg::LinalgOp>,
               OperationConvertTileVectorPass<tensor::PackOp>>(
      patterns.getContext());
  patterns.add<TensorUnpackConvertVectorPass>(patterns.getContext());
  patterns.add<TensorOpConvertVectorPass>(patterns.getContext());
}

/// LowerToTileVectorPass is a pass that lowers operations to tile (virtual)
/// vector. We must aware that this pass do not support dynamic shape currently.
struct LowerToTileVectorPass
    : public impl::LowerToTileVectorBase<LowerToTileVectorPass> {
  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet patternsInit(ctx);
    auto funcOp = getOperation();

    // Pad operation will lower to linalg.fill. We lower it in init patterns
    // then lower the fill operation in first patterns.
    linalg::populatePadOpVectorizationPatterns(patternsInit);

    GreedyRewriteConfig configInit;
    // Init patterns use to remove useless tensor operation like extract or
    // insert slice.
    configInit.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patternsInit),
                                       configInit);

    RewritePatternSet firstPatterns(ctx);
    // All the dynamic shape will reject to lower.
    populateLowerToTileVectorPatterns(firstPatterns);
    GreedyRewriteConfig configFirstPn;
    // We only apply the lowering pattern on existing operations
    configFirstPn.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(firstPatterns),
                                       configFirstPn);
    // Error case:
    // ```
    // linalg.copy : <1x32xf32>
    // tensor.insert_slice tensor<1x32xf32> to tensor<1x128x1x32xf32>
    // (The permutation map (permutation map = (d0, d1, d2, d3) -> (d0,
    // d3)) appears in the context of the current insert slice.)
    // --> lowering as:
    // transfer_write : permutation map = (d0, d1, d2, d3) -> (d0, d3) (This
    // permutation map should not appear because copy is just a direct write and
    // has no other permutation semantics. )
    // ```
    // Inorder to avoid the fold greedily bug (fold wrong
    // permution map for the transfer_write operation). Give it the new full IR
    // to fold second time can fold correctly. This is due to fold the existing
    // operation and new operation together.
    RewritePatternSet secondPattern(ctx);
    // Ensure each operation has a clear semantics, rather than a composite
    // semantics. Instead of leaving it to the subsequent passes to handle these
    // complex semantics, it reduces the difficulty of handling operations in
    // the subsequent passes. Like transfer_read and transfer_write may have
    // transpose or braodcast semantic etc.
    vector::populateVectorTransferPermutationMapLoweringPatterns(secondPattern);
    // Remove unnessary broadcast operation
    // TODO: disable this pattern until the following support is ready
    // vector::populateSinkVectorBroadcastPatterns(secondPattern);
    // Second fold (with the help of the `applyPatternsAndFoldGreedily`
    // function) can help us to eliminate redundant operation like consecutive
    // read and write.
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(secondPattern));
    // may need other patterns to reduce redundant operations
  }
};
} // namespace
} // namespace gc
} // namespace mlir