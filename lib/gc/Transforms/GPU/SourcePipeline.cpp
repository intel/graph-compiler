// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;

namespace mlir::gc {
namespace {
struct SourceAttention : OpRewritePattern<linalgx::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgx::AttentionOp op,
                                PatternRewriter &b) const override {
    auto q = dyn_cast<RankedTensorType>(op.getQuery().getType());
    auto k = dyn_cast<RankedTensorType>(op.getKey().getType());
    auto v = dyn_cast<RankedTensorType>(op.getValue().getType());
    auto out = dyn_cast<RankedTensorType>(op.getOutput().getType());
    auto sameIfStatic = [](int64_t a, int64_t c) {
      return ShapedType::isDynamic(a) || ShapedType::isDynamic(c) || a == c;
    };
    if (!q || !k || !v || !out || (out.getRank() != 3 && out.getRank() != 4) ||
        q.getRank() != out.getRank() || k.getRank() != out.getRank() ||
        v.getRank() != out.getRank() ||
        !sameIfStatic(q.getDimSize(out.getRank() - 1),
                      k.getDimSize(out.getRank() - 1)) ||
        !sameIfStatic(k.getDimSize(out.getRank() - 2),
                      v.getDimSize(out.getRank() - 2)) ||
        !sameIfStatic(q.getDimSize(out.getRank() - 2),
                      out.getDimSize(out.getRank() - 2)) ||
        !sameIfStatic(v.getDimSize(out.getRank() - 1),
                      out.getDimSize(out.getRank() - 1)))
      return failure();
    if ((!q.getElementType().isF16() && !q.getElementType().isF32()) ||
        k.getElementType() != q.getElementType() ||
        v.getElementType() != q.getElementType() ||
        (!out.getElementType().isF16() && !out.getElementType().isF32()) ||
        (!op.getScale().getType().isF16() && !op.getScale().getType().isF32()))
      return failure();

    unsigned rank = out.getRank();
    auto leading = [&](RankedTensorType type) {
      SmallVector<AffineExpr> dims;
      for (unsigned i = 0; i + 2 < rank; ++i) {
        if (type.getDimSize(i) == 1) dims.push_back(b.getAffineConstantExpr(0));
        else if (sameIfStatic(type.getDimSize(i), out.getDimSize(i)))
          dims.push_back(b.getAffineDimExpr(i));
        else return SmallVector<AffineExpr>();
      }
      return dims;
    };
    auto qDims = leading(q), kDims = leading(k), vDims = leading(v);
    if (qDims.size() != rank - 2 || kDims.size() != rank - 2 ||
        vDims.size() != rank - 2)
      return failure();
    if (auto mask = op.getMask()) {
      auto type = dyn_cast<RankedTensorType>(mask.getType());
      if (!type || type.getRank() != 2 ||
          (!type.getElementType().isF16() && !type.getElementType().isF32()) ||
          (type.getDimSize(0) != 1 &&
           !sameIfStatic(type.getDimSize(0), q.getDimSize(rank - 2))) ||
          (type.getDimSize(1) != 1 &&
           !sameIfStatic(type.getDimSize(1), k.getDimSize(rank - 2))))
        return failure();
    }

    Location loc = op.getLoc();
    auto f32 = b.getF32Type();
    auto zero = arith::ConstantOp::create(b, loc, f32, b.getF32FloatAttr(0));
    auto minusInf = arith::ConstantOp::create(
        b, loc, f32,
        b.getFloatAttr(f32, APFloat::getInf(APFloat::IEEEsingle(), true)));
    auto convert = [&](OpBuilder &builder, Location l, Value x) {
      return x.getType() == f32
                 ? x
                 : arith::ExtFOp::create(builder, l, f32, x, {}).getResult();
    };
    auto empty = [&](ArrayRef<int64_t> shape, ArrayRef<Value> dimensions,
                     Type type = Type()) {
      SmallVector<Value> dynamic;
      for (size_t i = 0; i < shape.size(); ++i)
        if (ShapedType::isDynamic(shape[i])) dynamic.push_back(dimensions[i]);
      return tensor::EmptyOp::create(b, loc, shape, type ? type : f32, dynamic)
          .getResult();
    };
    auto filled = [&](ArrayRef<int64_t> shape, ArrayRef<Value> dimensions,
                      Value value) {
      return linalg::FillOp::create(b, loc, ValueRange{value},
                                    ValueRange{empty(shape, dimensions)})
          .getResult(0);
    };
    SmallVector<int64_t> scoresShape(out.getShape());
    scoresShape.back() = k.getDimSize(rank - 2);
    SmallVector<Value> outDims;
    for (unsigned i = 0; i < rank; ++i)
      outDims.push_back(tensor::DimOp::create(b, loc, op.getOutput(), i));
    SmallVector<Value> scoresDims(outDims);
    scoresDims.back() = tensor::DimOp::create(b, loc, op.getKey(), rank - 2);
    SmallVector<int64_t> rowShape(scoresShape.begin(), scoresShape.end() - 1);
    SmallVector<Value> rowDims(scoresDims.begin(), scoresDims.end() - 1);
    SmallVector<AffineExpr> outputDims;
    for (unsigned i = 0; i < rank; ++i)
      outputDims.push_back(b.getAffineDimExpr(i));
    auto identity = b.getMultiDimIdentityMap(rank);
    auto rowMap = AffineMap::get(rank, 0, ArrayRef(outputDims).drop_back(),
                                 b.getContext());
    auto contractionOut =
        AffineMap::get(rank + 1, 0, outputDims, b.getContext());
    auto map = [&](SmallVector<AffineExpr> dims, AffineExpr a, AffineExpr c) {
      dims.append({a, c});
      return AffineMap::get(rank + 1, 0, dims, b.getContext());
    };
    SmallVector<utils::IteratorType> parallel(rank,
                                              utils::IteratorType::parallel);
    auto reduction = parallel;
    reduction.back() = utils::IteratorType::reduction;
    auto generic = [&](ValueRange inputs, Value output,
                       ArrayRef<AffineMap> maps,
                       ArrayRef<utils::IteratorType> iterators, auto body) {
      return linalg::GenericOp::create(
                 b, loc, TypeRange{output.getType()}, inputs,
                 ValueRange{output}, maps, iterators,
                 [&](OpBuilder &builder, Location l, ValueRange args) {
                   linalg::YieldOp::create(builder, l,
                                           ValueRange{body(builder, l, args)});
                 })
          .getResult(0);
    };
    auto dot = [&](OpBuilder &builder, Location l, ValueRange args) {
      auto product =
          arith::MulFOp::create(builder, l, convert(builder, l, args[0]),
                                convert(builder, l, args[1]));
      return arith::AddFOp::create(builder, l, product, args[2]).getResult();
    };
    SmallVector<utils::IteratorType> contraction = parallel;
    contraction.push_back(utils::IteratorType::reduction);
    Value scores = generic(
        ValueRange{op.getQuery(), op.getKey()},
        filled(scoresShape, scoresDims, zero),
        {map(qDims, b.getAffineDimExpr(rank - 2), b.getAffineDimExpr(rank)),
         map(kDims, b.getAffineDimExpr(rank - 1), b.getAffineDimExpr(rank)),
         contractionOut},
        contraction, dot);

    SmallVector<Value> scaleInputs{scores, op.getScale()};
    SmallVector<AffineMap> scaleMaps{
        identity, AffineMap::get(rank, 0, {}, b.getContext())};
    if (auto mask = op.getMask()) {
      auto type = cast<RankedTensorType>(mask.getType());
      SmallVector<AffineExpr> dims{
          type.getDimSize(0) == 1 ? b.getAffineConstantExpr(0)
                                  : b.getAffineDimExpr(rank - 2),
          type.getDimSize(1) == 1 ? b.getAffineConstantExpr(0)
                                  : b.getAffineDimExpr(rank - 1)};
      scaleInputs.push_back(mask);
      scaleMaps.push_back(AffineMap::get(rank, 0, dims, b.getContext()));
    }
    scaleMaps.push_back(identity);
    scores = generic(
        scaleInputs, empty(scoresShape, scoresDims), scaleMaps, parallel,
        [&](OpBuilder &builder, Location l, ValueRange args) {
          Value scaled = arith::MulFOp::create(builder, l, args[0],
                                               convert(builder, l, args[1]));
          return op.getMask()
                     ? arith::AddFOp::create(builder, l, scaled,
                                             convert(builder, l, args[2]))
                           .getResult()
                     : scaled;
        });
    Value maximum =
        generic(ValueRange{scores}, filled(rowShape, rowDims, minusInf),
                {identity, rowMap}, reduction,
                [](OpBuilder &builder, Location l, ValueRange args) {
                  return arith::MaximumFOp::create(builder, l, args[0], args[1])
                      .getResult();
                });
    Value exps = generic(
        ValueRange{scores, maximum}, empty(scoresShape, scoresDims),
        {identity, rowMap, identity}, parallel,
        [](OpBuilder &builder, Location l, ValueRange args) {
          return math::ExpOp::create(
              builder, l, arith::SubFOp::create(builder, l, args[0], args[1]));
        });
    Value sum = generic(
        ValueRange{exps}, filled(rowShape, rowDims, zero), {identity, rowMap},
        reduction, [](OpBuilder &builder, Location l, ValueRange args) {
          return arith::AddFOp::create(builder, l, args[0], args[1])
              .getResult();
        });
    Value weights =
        generic(ValueRange{exps, sum}, empty(scoresShape, scoresDims),
                {identity, rowMap, identity}, parallel,
                [](OpBuilder &builder, Location l, ValueRange args) {
                  return arith::DivFOp::create(builder, l, args[0], args[1])
                      .getResult();
                });
    SmallVector<AffineExpr> weightsDims(outputDims.begin(),
                                        outputDims.end() - 2);
    Value result = generic(
        ValueRange{weights, op.getValue()},
        filled(out.getShape(), outDims, zero),
        {map(weightsDims, b.getAffineDimExpr(rank - 2),
             b.getAffineDimExpr(rank)),
         map(vDims, b.getAffineDimExpr(rank), b.getAffineDimExpr(rank - 1)),
         contractionOut},
        contraction, dot);
    if (out.getElementType().isF16())
      result = generic(ValueRange{result},
                       empty(out.getShape(), outDims, b.getF16Type()),
                       {identity, identity}, parallel,
                       [](OpBuilder &builder, Location l, ValueRange args) {
                         return arith::TruncFOp::create(
                             builder, l, builder.getF16Type(), args[0]);
                       });
    b.replaceOp(op, result);
    return success();
  }
};

struct SourceGroupedMatmul : OpRewritePattern<linalgx::GroupedMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgx::GroupedMatmulOp op,
                                PatternRewriter &b) const override {
    auto lhs = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto weights = dyn_cast<RankedTensorType>(op.getWeights().getType());
    auto out = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!lhs || !weights || !out || weights.isDynamicDim(0) ||
        !lhs.getElementType().isF32() ||
        weights.getElementType() != lhs.getElementType() ||
        out.getElementType() != lhs.getElementType())
      return failure();
    auto loc = op.getLoc();
    auto m = b.getAffineDimExpr(0), n = b.getAffineDimExpr(1),
         k = b.getAffineDimExpr(2);
    auto result =
        linalg::GenericOp::create(
            b, loc, TypeRange{out}, ValueRange{op.getLhs()},
            ValueRange{op.getOutput()},
            ArrayRef<AffineMap>{AffineMap::get(3, 0, {m, k}, b.getContext()),
                                AffineMap::get(3, 0, {m, n}, b.getContext())},
            ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                          utils::IteratorType::parallel,
                                          utils::IteratorType::reduction},
            [&](OpBuilder &builder, Location l, ValueRange args) {
              Value row = linalg::IndexOp::create(builder, l, 0);
              Value col = linalg::IndexOp::create(builder, l, 1);
              Value inner = linalg::IndexOp::create(builder, l, 2);
              Value expert = arith::ConstantIndexOp::create(builder, l, 0);
              for (int64_t i = 0; i + 1 < weights.getDimSize(0); ++i) {
                Value index = arith::ConstantIndexOp::create(builder, l, i);
                Value end = tensor::ExtractOp::create(
                    builder, l, op.getOffsets(), ValueRange{index});
                end = arith::IndexCastOp::create(builder, l,
                                                 builder.getIndexType(), end);
                Value next = arith::ConstantIndexOp::create(builder, l, i + 1);
                Value past = arith::CmpIOp::create(
                    builder, l, arith::CmpIPredicate::sge, row, end);
                expert =
                    arith::SelectOp::create(builder, l, past, next, expert);
              }
              Value weight = tensor::ExtractOp::create(
                  builder, l, op.getWeights(), ValueRange{expert, col, inner});
              Value product =
                  arith::MulFOp::create(builder, l, args[0], weight);
              Value sum = arith::AddFOp::create(builder, l, product, args[1]);
              linalg::YieldOp::create(builder, l, ValueRange{sum});
            })
            .getResult(0);
    b.replaceOp(op, result);
    return success();
  }
};

struct DecomposeSourceAttention
    : PassWrapper<DecomposeSourceAttention, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeSourceAttention)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SourceAttention, SourceGroupedMatmul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    getOperation()->walk([&](linalgx::AttentionOp op) {
      op.emitError(
          "Unsupported attention shape or precision in OpenCL source backend");
      signalPassFailure();
    });
    getOperation()->walk([&](linalgx::GroupedMatmulOp op) {
      op.emitError("Unsupported grouped matmul in OpenCL source backend");
      signalPassFailure();
    });
  }
};
} // namespace

void populateOclSourcePipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(std::make_unique<DecomposeSourceAttention>());
  pm.addNestedPass<func::FuncOp>(createDecomposition());
  pm.addNestedPass<func::FuncOp>(createTensorConcatToLinalg());
  pm.addPass(createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createCanonicalizerPass());
  bufferization::OneShotBufferizePassOptions opts;
  opts.bufferizeFunctionBoundaries = true;
  opts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(opts));
  pm.addPass(createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}
} // namespace mlir::gc
