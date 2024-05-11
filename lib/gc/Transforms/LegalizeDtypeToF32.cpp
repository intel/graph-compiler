//===- LegalizeDtypeToF32.cpp - Promote low-precision to f32 ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include "gc/Transforms/LegalizeUtils.h"
#include "gc/Transforms/Passes.h"
namespace mlir {
namespace gc {
#define GEN_PASS_DEF_LEGALIZEDTYPETOF32
#include "gc/Transforms/Passes.h.inc"
CPUFlags cpuf;
} // namespace gc

struct LegalizeToF32RewritePattern final : ConversionPattern {
  LegalizeToF32RewritePattern(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, 1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult LegalizeToF32RewritePattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  const TypeConverter *converter = getTypeConverter();
  FailureOr<Operation *> legalized =
      convertOpResultTypes(op, operands, *converter, rewriter);
  if (failed(legalized))
    return failure();

  SmallVector<Value> results = (*legalized)->getResults();
  for (auto [result, newType, origType] : llvm::zip_equal(
           results, (*legalized)->getResultTypes(), op->getResultTypes())) {
    if (newType != origType)
      result = rewriter.create<arith::TruncFOp>(loc, origType, result);
  }
  rewriter.replaceOp(op, results);
  return success();
}

namespace gc {
template <typename T>
void populateLegalizeToF32TypeConverter(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](Type type) -> std::optional<Type> { return type; });
  typeConverter.addConversion([](FloatType type) -> std::optional<Type> {
    if (isa<T>(type))
      return Float32Type::get(type.getContext());
    return std::nullopt;
  });
  typeConverter.addConversion([](ShapedType type) -> std::optional<Type> {
    if (auto elemTy = dyn_cast<T>(type.getElementType()))
      return type.clone(Float32Type::get(type.getContext()));
    return std::nullopt;
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &b, Type target, ValueRange input, Location loc) {
        return b.create<arith::ExtFOp>(loc, target, input);
      });
}

void populateBfloat16ToF32ConversionTarget(ConversionTarget &target,
                                           TypeConverter &typeConverter) {
  target.addDynamicallyLegalDialect<mlir::math::MathDialect>(
      [&typeConverter](Operation *op) -> bool {
        return typeConverter.isLegal(op);
      });
  target.addDynamicallyLegalOp<
      arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
      arith::MaximumFOp, arith::MinimumFOp, arith::CmpFOp, arith::SelectOp>(
      [&typeConverter](Operation *op) -> bool {
        return typeConverter.isLegal(op);
      });
  target.addLegalOp<arith::ExtFOp, arith::TruncFOp>();
}

void populateFloat16ToF32ConversionTarget(ConversionTarget &target,
                                          TypeConverter &typeConverter) {
  target.addDynamicallyLegalDialect<mlir::math::MathDialect>(
      [&typeConverter](Operation *op) -> bool {
        return typeConverter.isLegal(op);
      });
  if (!cpuf.fAVX512FP16)
    target.addDynamicallyLegalOp<
        arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
        arith::MaximumFOp, arith::MinimumFOp, arith::CmpFOp, arith::SelectOp>(
        [&typeConverter](Operation *op) -> bool {
          return typeConverter.isLegal(op);
        });
  else
    target.addLegalOp<math::AbsFOp, math::CeilOp, math::FloorOp, math::RoundOp,
                      math::SqrtOp, math::RsqrtOp, math::Exp2Op>();
  target.addLegalOp<arith::ExtFOp, arith::TruncFOp>();
}

void populateLegalizeToF32Patterns(RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<LegalizeToF32RewritePattern>(typeConverter,
                                            patterns.getContext());
}

class LegalizeDTypeToF32
    : public impl::LegalizeDTypeToF32Base<LegalizeDTypeToF32> {
public:
  using impl::LegalizeDTypeToF32Base<
      LegalizeDTypeToF32>::LegalizeDTypeToF32Base;
  void runOnOperation() final;
};

void LegalizeDTypeToF32::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext &ctx = getContext();

  // for bf16 legalization
  {
    TypeConverter bf16Converter;
    populateLegalizeToF32TypeConverter<BFloat16Type>(bf16Converter);
    ConversionTarget target(ctx);
    populateBfloat16ToF32ConversionTarget(target, bf16Converter);
    RewritePatternSet patterns(&ctx);
    populateLegalizeToF32Patterns(patterns, bf16Converter);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }

  // for fp16 legalization
  {
    TypeConverter f16Converter;
    populateLegalizeToF32TypeConverter<Float16Type>(f16Converter);
    ConversionTarget target(ctx);
    populateFloat16ToF32ConversionTarget(target, f16Converter);
    RewritePatternSet patterns(&ctx);
    populateLegalizeToF32Patterns(patterns, f16Converter);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
}

} // namespace gc
} // namespace mlir
