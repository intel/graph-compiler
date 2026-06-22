//===-- TensorMaskingOpInterface.cpp -----------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/TensorMaskingOpInterface.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir::gc {

#include "gc/Transforms/TensorMaskingOpInterface.cpp.inc" // IWYU pragma: keep

namespace {

static bool isPaddingNeeded(OpBuilder &b, TilingInterface op,
                            ArrayRef<OpFoldResult> padMultiples) {
  SmallVector<Range> iterationDomain = op.getIterationDomain(b);
  assert(iterationDomain.size() == padMultiples.size() &&
         "expected padMultiples to match the number of iteration dimensions");

  for (auto [range, padMultiple] :
       llvm::zip_equal(iterationDomain, padMultiples)) {
    std::optional<int64_t> padSize = getConstantIntValue(padMultiple);
    if (!padSize || *padSize == 0) continue;
    std::optional<int64_t> dimSize = getConstantIntValue(range.size);
    if (!dimSize) return true;
    if (*dimSize % *padSize != 0) return true;
  }
  return false;
}

static FailureOr<Attribute> getZeroPadAttr(OpBuilder &b, Type operandType) {
  Type elemTy = getElementTypeOrSelf(operandType);
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return b.getFloatAttr(floatTy, 0.0);
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return b.getIntegerAttr(intTy, 0);
  return failure();
}

struct LinalgGenericOpMaskingInterface final
    : TensorMaskingOpInterface::ExternalModel<LinalgGenericOpMaskingInterface,
                                              linalg::GenericOp> {

  FailureOr<SmallVector<Value>>
  getMaskedImplementation(Operation *op, OpBuilder &builder,
                          ArrayRef<OpFoldResult> padMultiples) const {
    auto genericOp = cast<linalg::GenericOp>(op);
    auto tilingOp = cast<TilingInterface>(op);

    if (!isPaddingNeeded(builder, tilingOp, padMultiples)) return failure();

    SmallVector<Attribute> padValues;
    for (Value operand : op->getOperands()) {
      auto attr = getZeroPadAttr(builder, operand.getType());
      if (failed(attr))
        return op->emitError(
            "linalg.generic masking: unsupported element type");
      padValues.push_back(*attr);
    }

    linalg::PadTilingInterfaceOptions options =
        linalg::PadTilingInterfaceOptions()
            .setPaddingSizes(padMultiples)
            .setPaddingValues(padValues)
            .setPadToMultipleOf(false);

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(op);
    FailureOr<linalg::PadTilingInterfaceResult> result =
        linalg::rewriteAsPaddedOp(builder, tilingOp, options);
    if (failed(result)) return failure();

    auto paddedGeneric = cast<linalg::GenericOp>(result->paddedOp);

    if (failed(maskReductions(builder, genericOp, paddedGeneric)))
      return failure();

    return result->replacements;
  }

private:
  static LogicalResult maskReductions(OpBuilder &builder,
                                      linalg::GenericOp origOp,
                                      linalg::GenericOp paddedOp) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(origOp);
    SmallVector<Range> origDomain = cast<TilingInterface>(origOp.getOperation())
                                        .getIterationDomain(builder);
    builder.setInsertionPoint(paddedOp);
    SmallVector<Range> paddedDomain =
        cast<TilingInterface>(paddedOp.getOperation())
            .getIterationDomain(builder);

    SmallVector<OpFoldResult> guardBounds;
    for (auto [origR, paddedR, iterTy] : llvm::zip_equal(
             origDomain, paddedDomain, origOp.getIteratorTypesArray())) {
      if (iterTy == utils::IteratorType::reduction &&
          origR.size != paddedR.size)
        guardBounds.push_back(origR.size);
      else guardBounds.push_back(OpFoldResult{});
    }

    bool anyGuard =
        llvm::any_of(guardBounds, [](OpFoldResult ofr) { return bool(ofr); });
    if (!anyGuard) return success();

    Location loc = paddedOp.getLoc();
    Block &body = paddedOp.getRegion().front();
    builder.setInsertionPointToStart(&body);

    OpFoldResult isInBounds = builder.getBoolAttr(true);
    for (auto [idx, bound] : llvm::enumerate(guardBounds)) {
      if (!bound) continue;
      Value loopIdx = linalg::IndexOp::create(builder, loc, idx);
      Value cmp = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ult, loopIdx,
          getValueOrCreateConstantIndexOp(builder, loc, bound));
      isInBounds = builder.createOrFold<arith::AndIOp>(
          loc, getValueOrCreateConstantIntOp(builder, loc, isInBounds), cmp);
    }

    for (auto [initIdx, initOpOperand] :
         llvm::enumerate(paddedOp.getDpsInitsMutable())) {
      SmallVector<Operation *> combinerOps;
      matchReduction(paddedOp.getRegionOutputArgs(), initIdx, combinerOps);
      for (Operation *combiner : combinerOps) {
        std::optional<TypedAttr> neutral = arith::getNeutralElement(combiner);
        if (!neutral) continue;
        builder.setInsertionPoint(combiner);
        Value neutralVal = arith::ConstantOp::create(builder, loc, *neutral);
        for (OpOperand &opOperand : combiner->getOpOperands()) {
          auto bbArg = dyn_cast<BlockArgument>(opOperand.get());
          if (bbArg && paddedOp.isDpsInit(paddedOp.getMatchingOpOperand(bbArg)))
            continue;
          Value masked = builder.createOrFold<arith::SelectOp>(
              loc, getValueOrCreateConstantIntOp(builder, loc, isInBounds),
              opOperand.get(), neutralVal);
          opOperand.set(masked);
        }
      }
    }
    return success();
  }
};

} // namespace

void registerTensorMaskingOpInterfaceForLinalg(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::GenericOp::attachInterface<LinalgGenericOpMaskingInterface>(*ctx);
  });
}

} // namespace mlir::gc
