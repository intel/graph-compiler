//===-- OneDNNGraphOps.cpp - OneDNN input dialect ops -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Debug.h"

#define GET_OP_CLASSES
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.cpp.inc"

namespace mlir {
namespace onednn_graph {

//===----------------------------------------------------------------------===//
// Binary ops shape infer
//===----------------------------------------------------------------------===//

#define BINARY_OP_SHAPE_INFER(OP)                                              \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      OP::Adaptor adaptor,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    auto inputTy0 = dyn_cast<ShapedType>(adaptor.getInputA().getType());       \
    auto inputTy1 = dyn_cast<ShapedType>(adaptor.getInputB().getType());       \
    if (!adaptor.getAutoBroadcast() && (inputTy0 != inputTy1)) {               \
      return failure();                                                        \
    }                                                                          \
    llvm::SmallVector<int64_t> outShape;                                       \
    auto ret = OpTrait::util::getBroadcastedShape(                             \
        inputTy0.getShape(), inputTy1.getShape(), outShape);                   \
    inferredReturnShapes.push_back(                                            \
        ShapedTypeComponents(outShape, inputTy0.getElementType()));            \
    return LogicalResult::success(ret);                                        \
  }

BINARY_OP_SHAPE_INFER(onednn_graph::AddOp)
BINARY_OP_SHAPE_INFER(onednn_graph::MulOp)
BINARY_OP_SHAPE_INFER(onednn_graph::SubOp)
BINARY_OP_SHAPE_INFER(onednn_graph::DivOp)

//===----------------------------------------------------------------------===//
// Reduce ops shape infer
//===----------------------------------------------------------------------===//

SmallVector<int64_t> canonicalizeReduceAxes(ArrayRef<int64_t> axes,
                                            int64_t rank) {
  SmallVector<int64_t> ret(axes.size());
  for (size_t i = 0; i < axes.size(); i++) {
    ret[i] = axes[i] < 0 ? axes[i] + rank : axes[i];
  }
  llvm::sort(ret);
  ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
  return ret;
}

SmallVector<int64_t> getReducedShape(ShapeAdaptor operandShape,
                                     ArrayRef<int64_t> axes, bool keep_dims) {
  SmallVector<int64_t> outputShape;
  // get reduce axis one by one
  size_t index = 0;
  auto getNextReduceAxis = [&]() {
    return (index >= axes.size()) ? -1 : axes[index++];
  };
  // get reduced shape
  auto rank = operandShape.getRank();
  auto axis = getNextReduceAxis();
  for (int64_t idx = 0; idx < rank; idx++) {
    if (idx == axis) {
      axis = getNextReduceAxis();
      if (keep_dims) {
        outputShape.push_back(1);
      }
    } else {
      outputShape.push_back(operandShape.getDimSize(idx));
    }
  }
  return outputShape;
}

static LogicalResult InferReduceReturnTypes(
    ShapeAdaptor operandShape, Type elemType, ArrayRef<int64_t> axes,
    bool keep_dims,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // no reduce axes
  if (axes.empty()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(operandShape));
    return success();
  }
  inferredReturnShapes.push_back(ShapedTypeComponents(
      getReducedShape(operandShape, axes, keep_dims), elemType));
  return success();
}

template <typename ReduceOp>
struct CanonicalizeReduceOp : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto rank = dyn_cast<ShapedType>(op.getOperand().getType()).getRank();
    // consider canonicalized if all axes are non-negative in ascending order
    // Note: disable tidy here due to dangling reference in OperationState
    // NOLINTBEGIN
    bool canonicalized = true;
    int64_t last = -1;
    for (const auto axis : op.getAxes()) {
      if (axis <= last) {
        canonicalized = false;
        break;
      }
      last = axis;
    }
    if (canonicalized) {
      return failure();
    }
    // canonicalize the reduce axes
    auto new_axes = canonicalizeReduceAxes(op.getAxes(), rank);
    auto new_op = rewriter.create<ReduceOp>(
        op.getLoc(), op.getType(), op.getOperand(), new_axes, op.getKeepDims());
    rewriter.replaceOp(op, new_op);
    // NOLINTEND
    return success();
  }
};

#define REDUCE_OP_SHAPE_CANONICALIZE(OP)                                       \
  void OP::getCanonicalizationPatterns(RewritePatternSet &results,             \
                                       MLIRContext *context) {                 \
    using CanonicalizeOp = CanonicalizeReduceOp<OP>;                           \
    results.add<CanonicalizeOp>(context);                                      \
  }

#define REDUCE_OP_SHAPE_INFER(OP)                                              \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      OP::Adaptor adaptor,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    llvm::SmallVector<int64_t> outShape;                                       \
    auto operandTy = dyn_cast<ShapedType>(adaptor.getOperand().getType());     \
    auto rank = operandTy.getRank();                                           \
    ShapeAdaptor inputShape(operandTy);                                        \
    return InferReduceReturnTypes(                                             \
        inputShape, operandTy.getElementType(),                                \
        canonicalizeReduceAxes(adaptor.getAxes(), rank),                       \
        adaptor.getKeepDims(), inferredReturnShapes);                          \
  }

#define REDUCE_OP_VERIFY(OP)                                                   \
  LogicalResult OP::verify() {                                                 \
    auto operandTy = dyn_cast<ShapedType>(getOperand().getType());             \
    if (!operandTy.hasRank()) {                                                \
      return emitOpError("Invalid operand shape!\n");                          \
    }                                                                          \
    int64_t rank = operandTy.getRank();                                        \
    for (const auto axis : canonicalizeReduceAxes(getAxes(), rank)) {          \
      if (axis >= rank || axis < 0) {                                          \
        return emitOpError("Reduce axis not valid!\n");                        \
      }                                                                        \
    }                                                                          \
    return success();                                                          \
  }

#define REDUCE_OP_DEFINE(OP)                                                   \
  REDUCE_OP_SHAPE_CANONICALIZE(OP)                                             \
  REDUCE_OP_SHAPE_INFER(OP)                                                    \
  REDUCE_OP_VERIFY(OP)

REDUCE_OP_DEFINE(onednn_graph::ReduceSumOp)
REDUCE_OP_DEFINE(onednn_graph::ReduceMeanOp)

//===----------------------------------------------------------------------===//
// Matmul ops shape infer
//===----------------------------------------------------------------------===//

LogicalResult onednn_graph::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    MatMulOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // get batch dims from shape
  auto extractBatch = [](const ShapeAdaptor &lhsShape,
                         const ShapeAdaptor &rhsShape, int64_t range,
                         int64_t diff, SmallVector<int64_t> &outDims) {
    for (int64_t i = 0; i < range; i++) {
      // TODO(longsheng): add OpTrait::util::getBroadcastedShape for batch
      int64_t idx = i - diff;
      if ((idx >= 0) && (lhsShape.getDimSize(i) != rhsShape.getDimSize(idx))) {
        return failure();
      }
      outDims.push_back(lhsShape.getDimSize(i));
    }
    return success();
  };
  // get row col of 2d matrix according to transpose info
  auto getMatRowCol = [](const ShapeAdaptor &shape, bool transpose) {
    using pairRowCol = std::pair<int64_t, int64_t>;
    auto rank = shape.getRank();
    assert(rank > 1);
    return transpose ? pairRowCol(shape.getDimSize(rank - 1),
                                  shape.getDimSize(rank - 2))
                     : pairRowCol(shape.getDimSize(rank - 2),
                                  shape.getDimSize(rank - 1));
  };
  ShapeAdaptor lhsShape(adaptor.getInputA().getType());
  ShapeAdaptor rhsShape(adaptor.getInputB().getType());
  bool transposeA = adaptor.getTransposeA();
  bool transposeB = adaptor.getTransposeB();
  int64_t lRank = lhsShape.getRank();
  int64_t rRank = rhsShape.getRank();
  //
  SmallVector<int64_t> outShape;
  LogicalResult status = failure();
  if (lRank == 1 && rRank == 1) {
    // 1D x 1D
    if (lhsShape.getDimSize(0) != rhsShape.getDimSize(0)) {
      return failure();
    }
    outShape.push_back(1);
  } else if (lRank == 1 && rRank > 1) {
    // 1D x ND
    auto rMatRowCol = getMatRowCol(rhsShape, transposeB);
    status = extractBatch(rhsShape, rhsShape, rRank - 2, 0, outShape);
    if (lhsShape.getDimSize(0) != rMatRowCol.first) {
      return failure();
    }
    outShape.push_back(rhsShape.getDimSize(rMatRowCol.second));
  } else if (lRank > 1 && rRank == 1) {
    // ND x 1D
    auto lMatRowCol = getMatRowCol(lhsShape, transposeA);
    status = extractBatch(lhsShape, lhsShape, lRank - 2, 0, outShape);
    if (lMatRowCol.second != rhsShape.getDimSize(0)) {
      return failure();
    }
    outShape.push_back(lhsShape.getDimSize(lMatRowCol.first));
  } else if (lRank > 1 && rRank > 1) {
    if (lRank == rRank) {
      // ND x ND
      auto range = lRank - 2;
      status = extractBatch(lhsShape, rhsShape, range, 0, outShape);
    } else if (lRank > rRank) {
      // MD x ND (M > N)
      auto range = lRank - 2;
      auto diff = lRank - rRank;
      status = extractBatch(lhsShape, rhsShape, range, diff, outShape);
    } else {
      // ND x MD (M > N)
      auto range = rRank - 2;
      auto diff = rRank - lRank;
      status = extractBatch(rhsShape, lhsShape, range, diff, outShape);
    }
    //
    auto lMatRowCol = getMatRowCol(lhsShape, transposeA);
    auto rMatRowCol = getMatRowCol(rhsShape, transposeB);
    if (failed(status) || (lMatRowCol.second != rMatRowCol.first)) {
      return failure();
    }
    outShape.push_back(lMatRowCol.first);
    outShape.push_back(rMatRowCol.second);
  } else {
    // Not supported
    return failure();
  }
  // final shape
  auto retShape = ShapedTypeComponents(outShape, lhsShape.getElementType());
  inferredReturnShapes.push_back(retShape);
  // check for bias broadcasting
  if (adaptor.getBias()) {
    auto biasType = adaptor.getBias().getType();
    ShapeAdaptor biasShape(biasType);

    bool biasRankMatch = biasShape.getRank() == 1 ||
                         biasShape.getRank() == (int64_t)outShape.size();
    SmallVector<int64_t> resultShape;
    if (!biasRankMatch ||
        !OpTrait::util::getBroadcastedShape(
            retShape.getDims(), dyn_cast<ShapedType>(biasType).getShape(),
            resultShape)) {
      return failure();
    }
  }
  return success();
}

} // namespace onednn_graph
} // namespace mlir
