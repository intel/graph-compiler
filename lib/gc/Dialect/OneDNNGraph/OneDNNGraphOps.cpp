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

SmallVector<int64_t> canonicalizeKeepAxes(ArrayRef<int64_t> axes,
                                          int64_t rank) {
  auto reduceAxes = canonicalizeReduceAxes(axes, rank);
  SmallVector<int64_t> keepAxes;
  for (int64_t dim = 0, idx = 0; dim < rank; dim++) {
    if (idx < (int64_t)reduceAxes.size() && reduceAxes[idx] == dim) {
      idx++;
      continue;
    }
    keepAxes.push_back(dim);
  }
  return keepAxes;
}

SmallVector<int64_t> inferReducedShape(ShapedType operandShape,
                                       ArrayRef<int64_t> axes, bool keep_dims) {
  // get reduce axis one by one
  auto canonicalized = canonicalizeReduceAxes(axes, operandShape.getRank());
  size_t index = 0;
  auto getNextReduceAxis = [&]() {
    return (index >= canonicalized.size()) ? -1 : canonicalized[index++];
  };
  // get reduced shape
  SmallVector<int64_t> outputShape;
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
    ShapedType operandTy, ArrayRef<int64_t> axes, bool keep_dims,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // no reduce axes
  if (axes.empty()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(operandTy));
    return success();
  }
  inferredReturnShapes.push_back(
      ShapedTypeComponents(inferReducedShape(operandTy, axes, keep_dims),
                           operandTy.getElementType()));
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
    return InferReduceReturnTypes(operandTy, adaptor.getAxes(),                \
                                  adaptor.getKeepDims(),                       \
                                  inferredReturnShapes);                       \
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
  // get batch dims from 1 multi-batch mat shape
  auto extractBatch = [](ShapedType shape, SmallVector<int64_t> &outDims) {
    // assuming last 2 input dims are row and col
    assert(shape.getRank() >= 2);
    for (int64_t i = 0; i < shape.getRank() - 2; i++) {
      outDims.push_back(shape.getDimSize(i));
    }
    return success();
  };
  // get broadcasted batch dims from 2 multi-batch mat shape,
  auto extractBroadcastBatch = [](ShapedType lhsType, ShapedType rhsType,
                                  SmallVector<int64_t> &outDims) {
    SmallVector<int64_t> lhsShape(lhsType.getShape());
    SmallVector<int64_t> rhsShape(rhsType.getShape());
    // assuming last 2 input dims are row and col
    // 0xFF is just a random number > 1, replacing the row and col dims
    // so that getBroadcastedShape can match, will be removed after
    lhsShape[lhsShape.size() - 1] = 0xFF;
    lhsShape[lhsShape.size() - 2] = 0xFF;
    rhsShape[rhsShape.size() - 1] = 0xFF;
    rhsShape[rhsShape.size() - 2] = 0xFF;
    bool ret = OpTrait::util::getBroadcastedShape(lhsShape, rhsShape, outDims);
    // remove 0xFF
    assert(outDims.size() >= 2);
    outDims.pop_back();
    outDims.pop_back();
    return LogicalResult::success(ret);
  };
  // get row col of 2d matrix according to transpose info
  auto getMatRowCol = [](ShapedType shape, bool transpose) {
    using pairRowCol = std::pair<int64_t, int64_t>;
    auto rank = shape.getRank();
    assert(rank > 1);
    return transpose ? pairRowCol(shape.getDimSize(rank - 1),
                                  shape.getDimSize(rank - 2))
                     : pairRowCol(shape.getDimSize(rank - 2),
                                  shape.getDimSize(rank - 1));
  };
  auto lhsShape = cast<ShapedType>(adaptor.getInputA().getType());
  auto rhsShape = cast<ShapedType>(adaptor.getInputB().getType());
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
    status = extractBatch(rhsShape, outShape);
    if (lhsShape.getDimSize(0) != rMatRowCol.first) {
      return failure();
    }
    outShape.push_back(rhsShape.getDimSize(rMatRowCol.second));
  } else if (lRank > 1 && rRank == 1) {
    // ND x 1D
    auto lMatRowCol = getMatRowCol(lhsShape, transposeA);
    status = extractBatch(lhsShape, outShape);
    if (lMatRowCol.second != rhsShape.getDimSize(0)) {
      return failure();
    }
    outShape.push_back(lhsShape.getDimSize(lMatRowCol.first));
  } else if (lRank > 1 && rRank > 1) {
    // ND x ND
    // MD x ND (M > N)
    // ND x MD (M > N)
    status = extractBroadcastBatch(lhsShape, rhsShape, outShape);
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
    auto biasType = dyn_cast<ShapedType>(adaptor.getBias().getType());
    bool biasRankMatch = biasType.getRank() == 1 ||
                         biasType.getRank() == (int64_t)outShape.size();
    SmallVector<int64_t> resultShape;
    if (!biasRankMatch ||
        !OpTrait::util::getBroadcastedShape(retShape.getDims(),
                                            biasType.getShape(), resultShape)) {
      return failure();
    }
  }
  return success();
}

} // namespace onednn_graph
} // namespace mlir
