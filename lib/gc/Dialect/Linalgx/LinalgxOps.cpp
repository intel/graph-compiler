//===-- LinalgxOps.cpp - linalgx dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, Value query, Value key, Value value,
                        Value scale, Value output, ArrayAttr indexingMaps,
                        std::optional<Value> mask) {
  Value maskIn = mask.value_or(Value());
  build(odsBuilder, odsState, results, query, key, value, scale, maskIn, output,
        indexingMaps, DictionaryAttr());
}

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, ValueRange inputOperands,
                        ValueRange initOperands, ArrayAttr indexingMaps) {
  assert(inputOperands.size() < 6);
  assert(initOperands.size() == 1);
  Value mask = inputOperands.size() > 4 ? inputOperands[4] : Value();
  build(odsBuilder, odsState, results, inputOperands[0], inputOperands[1],
        inputOperands[2], inputOperands[3], mask, initOperands[0], indexingMaps,
        DictionaryAttr());
}

LogicalResult AttentionOp::verify() {
  AttentionOp attnOp = *this;

  // Check if indexing maps can represent attention.
  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();
  if (indexingMaps.size() != getOperation()->getNumOperands()) {
    return attnOp->emitOpError("expected an indexing map for each operand");
  }

  FloatType scaleElementType = dyn_cast<FloatType>(getScale().getType());
  if (!scaleElementType) {
    return attnOp->emitOpError("expected scale to be of floating point type");
  }

  // Check shape compatibility based on indexing maps.
  SmallVector<int64_t> shape(getIterationDomainRank());
  SmallVector<bool> foundDims(getIterationDomainRank(), false);
  auto checkShape = [&shape, &foundDims,
                     &attnOp](StringRef operandName, ArrayRef<int64_t> valShape,
                              AffineMap indexingMap) -> LogicalResult {
    if (indexingMap.getNumResults() != valShape.size()) {
      return attnOp->emitError("Rank Mismatch for ")
             << operandName << ". Expected: " << indexingMap.getNumResults()
             << " Got: " << valShape.size();
    }
    for (auto [i, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      AffineDimExpr dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (ShapedType::isDynamic(valShape[i])) {
        continue;
      }
      if (!foundDims[pos]) {
        foundDims[pos] = true;
        shape[pos] = valShape[i];
      }
      if (shape[pos] != valShape[i]) {
        return attnOp->emitError("Shape Mismatch for ")
               << operandName << " at position " << i
               << ". Expected: " << shape[pos] << " Got: " << valShape[i];
      }
    }
    return success();
  };

  if (failed(checkShape("Query", getQuery().getType().getShape(),
                        getQueryMap())) ||
      failed(checkShape("Key", getKey().getType().getShape(), getKeyMap())) ||
      failed(checkShape("Value", getValue().getType().getShape(),
                        getValueMap())) ||
      failed(checkShape("Output", getOutput().getType().getShape(),
                        getOutputMap()))) {
    return failure();
  }

  // Additional check case if mask exists
  if (auto maskMap = getMaskMap()) {
    if (failed(checkShape("Mask", getMask().getType().getShape(), *maskMap)))
      return failure();
  }

  unsigned expectedSymbols = getQueryMap().getNumInputs();
  auto checkDomain =
      [&attnOp, &expectedSymbols](StringRef operandName,
                                  AffineMap indexingMap) -> LogicalResult {
    if (expectedSymbols != indexingMap.getNumInputs()) {
      return attnOp->emitError("Mismatched map domain for ")
             << operandName << ". Expected: " << expectedSymbols
             << " Got: " << indexingMap.getNumInputs();
    }
    return success();
  };

  if (failed(checkDomain("Query", getQueryMap())) ||
      failed(checkDomain("Key", getKeyMap())) ||
      failed(checkDomain("Value", getValueMap())) ||
      failed(checkDomain("Scale", getScaleMap())) ||
      failed(checkDomain("Output", getOutputMap()))) {
    return failure();
  }

  // Additional check case if mask exists
  if (auto maskMap = getMaskMap()) {
    if (failed(checkDomain("Mask", *maskMap)))
      return failure();
  }

  return success();
}

ArrayRef<int64_t> AttentionOp::getShape(OpOperand* opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  Type t = opOperand->get().getType();
  // A VectorType is an elemental type, do not consider its rank for the operand.
  if (isa<VectorType>(t))
    return {};
  if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
    // Failsafe.
    assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
            "expected a ranked tensor or memref in LinalgInterface::getRank");
    return shapedType.getShape();
  }
  return {};
}

MutableOperandRange AttentionOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getMask() ? 5 : 4,
                             /*numInits=*/1);
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

void AttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  // FIXME: add canonicalization patterns
  // patterns.insert<StaticizeLinalgExtOp<AttentionOp>>(ctx);
}

AffineMap AttentionOp::getMatchingIndexingMap(OpOperand *operand) {
  return *(getIndexingMaps().getAsValueRange<AffineMapAttr>().begin() +
           operand->getOperandNumber());
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"
