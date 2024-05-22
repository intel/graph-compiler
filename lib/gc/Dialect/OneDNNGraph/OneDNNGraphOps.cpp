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

LogicalResult onednn_graph::AddOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outShape;
  auto resultTy = dyn_cast<ShapedType>(operands.front().getType());
  auto getShapeIdx = [&operands](size_t i) {
    return operands.getTypes()[i].dyn_cast<ShapedType>().getShape();
  };

  auto ret = OpTrait::util::getBroadcastedShape(getShapeIdx(0), getShapeIdx(1),
                                                outShape);
  inferredReturnShapes.push_back(
      ShapedTypeComponents(outShape, resultTy.getElementType()));
  return LogicalResult::success(ret);
}

LogicalResult onednn_graph::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    MatMulOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // get batch dims from shape
  auto extractBatch = [](const ShapeAdaptor &lhsShape,
                         const ShapeAdaptor &rhsShape, int64_t range,
                         int64_t diff, SmallVector<int64_t> &outDims) {
    for (int64_t i = 0; i < range; i++) {
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
            retShape.getDims(), biasType.dyn_cast<ShapedType>().getShape(),
            resultShape)) {
      return failure();
    }
  }
  return success();
}

} // namespace onednn_graph
} // namespace mlir
