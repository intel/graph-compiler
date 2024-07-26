//===- OneDNNGraphToLinalg.cpp - OneDNNGraph To Linalg Lowering -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <vector>

#include "gc/Dialect/Linalgx/IR/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/IR/LinalgxOps.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"
#include<iostream>
using namespace mlir::onednn_graph;

namespace mlir {
namespace onednn_graph {
SmallVector<int64_t> canonicalizeReduceAxes(ArrayRef<int64_t>, int64_t);
SmallVector<int64_t> canonicalizeKeepAxes(ArrayRef<int64_t>, int64_t, bool);
SmallVector<int64_t> inferReducedShape(ShapedType, ArrayRef<int64_t>, bool,
                                       bool);
} // namespace onednn_graph
namespace gc {
#define GEN_PASS_DEF_CONVERTONEDNNGRAPHTOLINALG
#include "gc/Transforms/Passes.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Util funcs
//===----------------------------------------------------------------------===//

SmallVector<SmallVector<int64_t, 2>>
computeReassociationByAnchor(ArrayRef<int64_t> anchorDims, int64_t rank) {
  // Compute reassociation indices according to
  // sorted anchor dims [a, b, ..., c] and original rank n, i.e. a<b<...<c<n
  // so reassociation = [[0,...a...][...b...][...c...,n-1]]
  SmallVector<SmallVector<int64_t, 2>> reassociation(anchorDims.size());
  for (int64_t dim = 0, pos = 0; dim < rank; dim++) {
    assert(anchorDims[pos] < rank);
    reassociation[pos].push_back(dim);
    if ((pos + 1 < (int64_t)anchorDims.size()) && anchorDims[pos] == dim) {
      pos++;
    }
  }
  return reassociation;
}

Value createBroadcastOperand(Location loc, PatternRewriter &rewriter,
                             ShapedType ty, Value op) {
  auto opTy = dyn_cast<ShapedType>(op.getType());
  llvm::ArrayRef<int64_t> bcastShape = ty.getShape();
  llvm::ArrayRef<int64_t> opShape = opTy.getShape();
  int64_t diff = bcastShape.size() - opShape.size();

  if (bcastShape.equals(opShape)) {
    return op;
  } else {
    // get broadcast dimensions
    llvm::SmallVector<int64_t> keepDims;
    llvm::SmallVector<int64_t> bcastDims;
    llvm::SmallVector<int64_t> collapseShape;
    for (int64_t i = 0; i < (int64_t)bcastShape.size(); i++) {
      int64_t idxOp = i - diff;
      if (idxOp < 0) {
        bcastDims.push_back(i);
      } else if (bcastShape[i] != opShape[idxOp]) {
        assert(opShape[idxOp] == 1);
        bcastDims.push_back(i);
      } else {
        keepDims.push_back(idxOp);
        collapseShape.push_back(opShape[idxOp]);
      }
    }
    // If dimensions with size 1 need to be broadcasted, need to collapse
    Value newVal = op;
    if (collapseShape.size() < opShape.size()) {
      // consider kept dims [a, b,..., c] as anchor so
      // reassociation = [[...a...], [...b...], ..., [...c...]]
      // e.g. for rank = 5, kept dims = [0, 2, 4]
      // [16, 1, 32, 1, 64] --collapse-> [16, 32, 64]
      // reassociation = [[0], [1, 2], [3, 4]]
      assert(collapseShape.size() + bcastDims.size() == bcastShape.size());
      auto reassociation =
          computeReassociationByAnchor(keepDims, opTy.getRank());
      ShapedType collapseTy =
          RankedTensorType::get(collapseShape, opTy.getElementType());
      newVal = rewriter.create<tensor::CollapseShapeOp>(loc, collapseTy, newVal,
                                                        reassociation);
    }
    // create a new output tensor
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, bcastShape, ty.getElementType());
    return rewriter
        .create<linalg::BroadcastOp>(
            /*location=*/loc,
            /*inputs=*/newVal,
            /*inits=*/initTensor,
            /*dimensions=*/bcastDims)
        .getResults()
        .front();
  }
}

// Typedef for function to get operands for transformed op
using GetOperandFn = mlir::Value (*)(Operation *, PatternRewriter &,
                                     ShapedType);

// Functions to get operands for from original op
struct OriginalOperand {
  template <unsigned I>
  static Value getIdx(Operation *op, PatternRewriter &b, ShapedType ty) {
    if (I >= op->getNumOperands()) {
      op->emitError("Index exceeds operand num.\n");
      return nullptr;
    }
    return createBroadcastOperand(op->getLoc(), b, ty, op->getOperand(I));
  }
};

// Functions to get constant operands
struct ConstantOperand {
  template <int64_t I>
  static Value getConst(Operation *op, PatternRewriter &b, ShapedType ty) {
    const auto loc = op->getLoc();
    const auto elemTy = ty.getElementType();
    if (llvm::isa<IntegerType>(elemTy)) {
      return b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(ty, b.getIntegerAttr(elemTy, I)));
    } else if (llvm::isa<FloatType>(elemTy)) {
      return b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(ty, b.getFloatAttr(elemTy, I)));
    } else {
      op->emitError("Not a supported element type for constant.\n");
      return nullptr;
    }
  }
  template <const char *Attr>
  static Value getAttr(Operation *op, PatternRewriter &b, ShapedType ty) {
    const auto loc = op->getLoc();
    const auto elemTy = ty.getElementType();
    if (!op->hasAttr(Attr)) {
      op->emitError() << "no attr found: " << Attr;
      return nullptr;
    }
    Attribute attr = op->getAttr(Attr);
    if (llvm::isa<IntegerType>(elemTy) && isa<IntegerAttr>(attr)) {
      auto val = cast<IntegerAttr>(attr).getInt();
      return b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(ty, b.getIntegerAttr(elemTy, val)));
    } else if (llvm::isa<FloatType>(elemTy) && isa<FloatAttr>(attr)) {
      auto val = cast<FloatAttr>(attr).getValueAsDouble();
      return b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(ty, b.getFloatAttr(elemTy, val)));
    } else {
      op->emitError() << "invalid attr data type: " << Attr;
      return nullptr;
    }
  }
};

//===----------------------------------------------------------------------===//
// Elemwise lowering
//===----------------------------------------------------------------------===//

// Generate elementwise op using linalg named ops
template <typename LoweredOp>
Operation *createElemwiseOp(Location loc, PatternRewriter &rewriter,
                            ShapedType ty, llvm::ArrayRef<Value> inputs) {
  // create a new output tensor
  Value outTensor =
      rewriter.create<tensor::EmptyOp>(loc, ty.getShape(), ty.getElementType());
  return rewriter.create<LoweredOp>(
      /*location=*/loc,
      /*resultTensorTypes=*/outTensor.getType(),
      /*inputs=*/inputs,
      /*outputs=*/outTensor);
}

template <typename UnaryOp, typename LoweredOp, GetOperandFn GetOperand>
struct UnaryElemwiseLowering : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto resultTy = dyn_cast<ShapedType>(op->getResultTypes().front());
    auto inOp = GetOperand(op, rewriter, resultTy);
    if (!inOp) {
      return rewriter.notifyMatchFailure(op, "Fail to get operand.");
    }
    auto unaryOp = createElemwiseOp<LoweredOp>(loc, rewriter, resultTy, {inOp});
    rewriter.replaceOp(op, unaryOp);
    return success();
  }
};

template <typename BinaryOp, typename LoweredOp, GetOperandFn GetOperandLHS,
          GetOperandFn GetOperandRHS>
struct BinaryElemwiseLowering : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto resultTy = dyn_cast<ShapedType>(op->getResultTypes().front());
    auto lhsOp = GetOperandLHS(op, rewriter, resultTy);
    auto rhsOp = GetOperandRHS(op, rewriter, resultTy);
    if (!lhsOp || !rhsOp) {
      return rewriter.notifyMatchFailure(op, "Fail to get operand.");
    }
    auto binaryOp = createElemwiseOp<LoweredOp>(loc, rewriter, resultTy, //
                                                {lhsOp, rhsOp});
    rewriter.replaceOp(op, binaryOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduce lowering
//===----------------------------------------------------------------------===//

static TypedAttr createReductionInitialValue(Operation *op, Type elementTy,
                                             PatternRewriter &rewriter) {
  if (isa<ReduceSumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy, 0.0);
  }
  if (isa<ReduceMeanOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy, 0.0);
  }
  op->emitError("invalid op for reduction initial value!");
  return {};
}

static Value createReductionBodyCalculation(Operation *op, ValueRange args,
                                            Type elementTy,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  if (isa<ReduceSumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }
  if (isa<ReduceMeanOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }
  op->emitError("invalid op for reduction body calculation!");
  return {};
}

static Operation *createLoweredReduceOp(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> reducedShape,
    ArrayRef<int64_t> keepAxes, ArrayRef<int64_t> reduceAxes, bool keepDims) {
  auto loc = op->getLoc();
  auto inputTy = cast<ShapedType>(op->getOperand(0).getType());
  auto resultTy = cast<ShapedType>(op->getResult(0).getType());
  auto elementTy = resultTy.getElementType();
  // Create init tensor
  auto valueAttr = createReductionInitialValue(op, elementTy, rewriter);
  auto fillValue = rewriter.create<arith::ConstantOp>(loc, valueAttr);
  auto linalgEmpty = rewriter.create<tensor::EmptyOp>( //
      loc, reducedShape, elementTy);
  auto linalgFilled = rewriter.create<linalg::FillOp>( //
      loc, ValueRange{fillValue}, ValueRange{linalgEmpty});
  // Create ReduceOp based on reduceAxes
  Operation *newOp = rewriter.create<linalg::ReduceOp>(
      loc, op->getOperand(0), linalgFilled.result(), reduceAxes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto result =
            createReductionBodyCalculation(op, blockArgs, elementTy, rewriter);
        nestedBuilder.create<linalg::YieldOp>(loc, result);
      });
  // If reduce mean, need div
  if (isa<ReduceMeanOp>(op)) {
    assert(llvm::isa<FloatType>(elementTy));
    int64_t count = 1;
    for (auto ax : reduceAxes) {
      count *= inputTy.getDimSize(ax);
    }
    auto reducedVal = newOp->getResult(0);
    auto reducedTy = cast<ShapedType>(reducedVal.getType());
    auto divValue = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(reducedTy,
                                    rewriter.getFloatAttr(elementTy, count)));
    newOp = createElemwiseOp<linalg::DivOp>(loc, rewriter, reducedTy,
                                            {reducedVal, divValue});
  }
  // If keep dims, create tensor::ExpandShapeOp
  if (keepDims) {
    // consider kept dims [a, b,..., c] as anchor so
    // reassociation = [[...a...], [...b...], ..., [...c...]]
    // e.g. for rank = 5, kept dims = [0, 2, 4]
    // [16, 32, 64] --expand-> [16, 1, 32, 1, 64]
    // reassociation = [[0], [1, 2], [3, 4]]
    assert(keepAxes.size() == reducedShape.size());
    auto reassociation =
        computeReassociationByAnchor(keepAxes, resultTy.getRank());
    newOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, resultTy, newOp->getResult(0), reassociation);
  }
  return newOp;
}

template <typename ReduceOp>
struct ReduceOpLowering : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const final {
    // If no reduce, just return the original
    if (op.getAxes().empty()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }
    // Get params
    auto operandTy = cast<ShapedType>(op.getOperand().getType());
    auto reduceAxes = canonicalizeReduceAxes(op.getAxes(), operandTy.getRank());
    auto keepAxes = canonicalizeKeepAxes(reduceAxes, operandTy.getRank(), true);
    auto reducedShape = inferReducedShape(operandTy, reduceAxes, false, true);
    // replace Op with linalg/tensor
    auto newOp = createLoweredReduceOp(rewriter, op, reducedShape, keepAxes,
                                       reduceAxes, op.getKeepDims());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Op lowering
//===----------------------------------------------------------------------===//
namespace attrs {
static const char beta[] = "beta";
} // namespace attrs

using SigmoidOpLowering =
    UnaryElemwiseLowering<onednn_graph::SigmoidOp, linalgx::SigmoidOp, //
                          OriginalOperand::getIdx<0>>;
using TypeCastLowering =
    UnaryElemwiseLowering<onednn_graph::TypeCastOp, linalg::CopyOp, //
                          OriginalOperand::getIdx<0>>;
using PowOpLowering =
    BinaryElemwiseLowering<onednn_graph::PowOp, linalg::PowFOp, //
                           OriginalOperand::getIdx<0>,
                           ConstantOperand::getAttr<attrs::beta>>;
using ReLUOpLowering =
    BinaryElemwiseLowering<onednn_graph::ReLUOp, linalg::MaxOp, //
                           OriginalOperand::getIdx<0>,
                           ConstantOperand::getConst<0>>;
using AddOpLowering =
    BinaryElemwiseLowering<onednn_graph::AddOp, linalg::AddOp, //
                           OriginalOperand::getIdx<0>,
                           OriginalOperand::getIdx<1>>;
using MulOpLowering =
    BinaryElemwiseLowering<onednn_graph::MulOp, linalg::MulOp, //
                           OriginalOperand::getIdx<0>,
                           OriginalOperand::getIdx<1>>;
using SubOpLowering =
    BinaryElemwiseLowering<onednn_graph::SubOp, linalg::SubOp, //
                           OriginalOperand::getIdx<0>,
                           OriginalOperand::getIdx<1>>;
using DivOpLowering =
    BinaryElemwiseLowering<onednn_graph::DivOp, linalg::DivOp, //
                           OriginalOperand::getIdx<0>,
                           OriginalOperand::getIdx<1>>;

using ReduceSumOpLowering = ReduceOpLowering<onednn_graph::ReduceSumOp>;
using ReduceMeanOpLowering = ReduceOpLowering<onednn_graph::ReduceMeanOp>;

//===----------------------------------------------------------------------===//
// MatMulOp lowering
//===----------------------------------------------------------------------===//

struct MatMulOpBatchFlatten : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto typeA = dyn_cast<ShapedType>(op.getInputA().getType());
    auto typeB = dyn_cast<ShapedType>(op.getInputB().getType());
    auto resultTy = dyn_cast<ShapedType>(op->getResultTypes().front());
    // Format batch dims in 3dx2d so inputs comply with linalg
    if (typeA.getRank() > 2 && typeB.getRank() == 2 && !op.getTransposeA()) {
      // flatten batched 3d inputA to 2d shape
      int64_t dim = 0;
      int64_t collapse = 1;
      SmallVector<SmallVector<int64_t, 2>> reassociation(2);
      for (; dim < typeA.getRank() - 1; dim++) {
        collapse *= typeA.getDimSize(dim);
        reassociation[0].push_back(dim);
      }
      reassociation[1].push_back(dim);
      // reassociation = {[0, 1, ...], [r-1]}
      // collapseShape = [D(0)*D(1)*..., D(r-1)]
      ShapedType collapseTy = RankedTensorType::get(
          {collapse, typeA.getShape().back()}, typeA.getElementType());
      // transform to collapse + matmul + expand
      auto newCollapse = rewriter.create<tensor::CollapseShapeOp>(
          loc, collapseTy, op.getInputA(), reassociation);
      auto newMatmul = rewriter.create<onednn_graph::MatMulOp>(
          loc, newCollapse->getResult(0), op.getInputB(), op.getBias(),
          op.getTransposeA(), op.getTransposeB());
      auto newExpand = rewriter.create<tensor::ExpandShapeOp>(
          loc, resultTy, newMatmul->getResult(0), reassociation);
      rewriter.replaceOp(op, newExpand);
      return success();
    }
    return failure();
  }
};

struct MatMulOpLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    // Make empty tensor
    auto getEmptyTensor = [&](ShapedType tensorTy) -> Value {
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(tensorTy.getElementType()));
      Value newTensor = rewriter.create<tensor::EmptyOp>(
          loc, tensorTy.getShape(), tensorTy.getElementType());
      return rewriter.create<linalg::FillOp>(loc, zero, newTensor).getResult(0);
    };
    // Get inputs
    auto transposeA = op.getTransposeA();
    auto transposeB = op.getTransposeB();
    auto resultTy = dyn_cast<ShapedType>(op->getResultTypes().front());
    // Lower to matmul
    Operation *newOp = nullptr;
    if (!transposeA && !transposeB) {
      // (A * B)
      auto outTensor = getEmptyTensor(resultTy);
      newOp = rewriter.create<linalg::MatmulOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputA(), op.getInputB()},
          /*outputs=*/outTensor);
    } else if (transposeA && !transposeB) {
      // T(A) * B
      auto outTensor = getEmptyTensor(resultTy);
      newOp = rewriter.create<linalg::MatmulTransposeAOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputA(), op.getInputB()},
          /*outputs=*/outTensor);
    } else if (!transposeA && transposeB) {
      // A * T(B)
      auto outTensor = getEmptyTensor(resultTy);
      newOp = rewriter.create<linalg::MatmulTransposeBOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputA(), op.getInputB()},
          /*outputs=*/outTensor);
    } else {
      // T(B * A)
      const auto &resultShape = resultTy.getShape();
      SmallVector<int64_t> transShape{resultShape[1], resultShape[0]};
      SmallVector<int64_t> permutation{1, 0};
      auto transTy = resultTy.clone(transShape);
      auto transTensor = getEmptyTensor(transTy);
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/transTy,
          /*inputs=*/ValueRange{op.getInputB(), op.getInputA()},
          /*outputs=*/transTensor);
      auto outTensor = getEmptyTensor(resultTy);
      newOp = rewriter.create<linalg::TransposeOp>(
          /*location=*/loc,
          /*inputs=*/matmulOp.getResult(0),
          /*outputs=*/outTensor,
          /*permutation=*/permutation);
    }
    // Lowering bias add
    if (op.getBias()) {
      Value bias =
          createBroadcastOperand(loc, rewriter, resultTy, op.getBias());
      Value outBias = rewriter.create<tensor::EmptyOp>(
          loc, resultTy.getShape(), resultTy.getElementType());
      newOp = rewriter.create<linalg::AddOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/outBias.getType(),
          /*inputs=*/ValueRange{newOp->getResult(0), bias},
          /*outputs=*/outBias);
    }

    // Passing mutmal configs to linalg.matmul
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass define
//===----------------------------------------------------------------------===//

struct ConvertOneDNNGraphToLinalg
    : public impl::ConvertOneDNNGraphToLinalgBase<ConvertOneDNNGraphToLinalg> {
  void runOnOperation() final {
    auto *ctx = &getContext();
    // ==========================================
    // perform pre conversion
    // ==========================================
    RewritePatternSet patternsPre(ctx);
    patternsPre.add<
        // clang-format off
        MatMulOpBatchFlatten
        // clang-format on
        >(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patternsPre)))) {
      signalPassFailure();
    }
    // ==========================================
    // perform conversion
    // ==========================================
    ConversionTarget target(getContext());
    target.addIllegalDialect<onednn_graph::OneDNNGraphDialect>();
    target.addLegalDialect<
        // clang-format off
        BuiltinDialect, 
        math::MathDialect,
        func::FuncDialect, 
        arith::ArithDialect, 
        tensor::TensorDialect,
        linalg::LinalgDialect, 
        linalgx::LinalgxDialect
        // clang-format on
        >();
    RewritePatternSet patterns(ctx);
    patterns.add<
        // clang-format off
        AddOpLowering, 
        MulOpLowering, 
        SubOpLowering, 
        DivOpLowering,
        PowOpLowering, 
        ReLUOpLowering, 
        MatMulOpLowering, 
        TypeCastLowering,
        SigmoidOpLowering, 
        ReduceSumOpLowering, 
        ReduceMeanOpLowering
        // clang-format on
        >(ctx);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir
