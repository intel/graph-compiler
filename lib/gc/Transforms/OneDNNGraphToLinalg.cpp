//===- OneDNNGraphToLinalg.cpp - OneDNN Graph To Linalg Lowering --*- C++ -*-=//
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <vector>

#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphOps.h"
#include "gc/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir::onednn_graph;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CONVERTONEDNNGRAPHTOLINALG
#include "gc/Transforms/Passes.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Util funcs
//===----------------------------------------------------------------------===//

Value createBroadcastOperand(Location loc, PatternRewriter &rewriter,
                             TensorType ty, Value op) {
  auto opTy = dyn_cast<TensorType>(op.getType());
  llvm::ArrayRef<int64_t> bcastShape = ty.getShape();
  llvm::ArrayRef<int64_t> opShape = opTy.getShape();
  int64_t diff = bcastShape.size() - opShape.size();

  if (bcastShape.equals(opShape)) {
    return op;
  } else {
    // get broadcast dimensions
    llvm::SmallVector<int64_t> bcastDims;
    for (int64_t i = 0; i < (int64_t)bcastShape.size(); i++) {
      int64_t idxOp = i - diff;
      if (idxOp < 0) {
        bcastDims.push_back(i);
      } else if (bcastShape[i] != opShape[idxOp]) {
        bcastDims.push_back(i);
      }
    }
    // create a new output tensor
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, bcastShape, ty.getElementType());
    return rewriter
        .create<linalg::BroadcastOp>(
            /*location=*/loc,
            /*inputs=*/op,
            /*inits=*/initTensor,
            /*dimensions=*/bcastDims)
        .getResults()
        .front();
  }
}

// Typedef for function to get operands for transformed op
typedef mlir::Value (*GetOperandFn)(Operation *, PatternRewriter &, TensorType);

// Function to get operands for from original op
struct OriginalOperand {
  template <unsigned I>
  static Value getIdx(Operation *op, PatternRewriter &b, TensorType ty) {
    if (I >= op->getNumOperands()) {
      op->emitError("Index exceeds operand num.\n");
      return nullptr;
    }
    return createBroadcastOperand(op->getLoc(), b, ty, op->getOperand(I));
  }
};

// Function to get constant operands
struct ConstantOperand {
  template <int64_t I>
  static Value getConst(Operation *op, PatternRewriter &b, TensorType ty) {
    const auto loc = op->getLoc();
    if (llvm::isa<IntegerType>(ty.getElementType())) {
      return b.create<arith::ConstantOp>( //
          loc, DenseElementsAttr::get(ty, int64_t(I)));
    } else if (llvm::isa<FloatType>(ty.getElementType())) {
      return b.create<arith::ConstantOp>( //
          loc, DenseElementsAttr::get(ty, float(I)));
    } else {
      op->emitError("Not a supported element type for constant.\n");
      return nullptr;
    }
  }
};

//===----------------------------------------------------------------------===//
// Elemwise lowering
//===----------------------------------------------------------------------===//

// Generate elementwise op using linalg named ops
template <typename LoweredOp>
Value createElemwiseOp(Location loc, PatternRewriter &rewriter, TensorType ty,
                       llvm::ArrayRef<Value> inputs) {
  // create a new output tensor
  Value outTensor =
      rewriter.create<tensor::EmptyOp>(loc, ty.getShape(), ty.getElementType());

  auto elemwiseOp = rewriter.create<LoweredOp>(
      /*location=*/loc,
      /*resultTensorTypes=*/outTensor.getType(),
      /*inputs=*/inputs,
      /*outputs=*/outTensor);

  return elemwiseOp.getResult(0);
}

template <typename UnaryOp, typename LoweredOp, GetOperandFn GetOperand>
struct UnaryElemwiseLowering : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto resultTy = dyn_cast<TensorType>(op->getResultTypes().front());
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
    auto resultTy = dyn_cast<TensorType>(op->getResultTypes().front());
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
// Op lowering
//===----------------------------------------------------------------------===//

using ReLUOpLowering =
    BinaryElemwiseLowering<onednn_graph::ReLUOp, linalg::MaxOp, //
                           OriginalOperand::getIdx<0>,
                           ConstantOperand::getConst<0>>;

using AddOpLowering =
    BinaryElemwiseLowering<onednn_graph::AddOp, linalg::AddOp, //
                           OriginalOperand::getIdx<0>,
                           OriginalOperand::getIdx<1>>;

//===----------------------------------------------------------------------===//
// MatMulOp lowering
//===----------------------------------------------------------------------===//

struct MatMulOpLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto resultTy = dyn_cast<TensorType>(op->getResultTypes().front());
    auto typeA = dyn_cast<TensorType>(op.getInputA().getType());
    auto typeB = dyn_cast<TensorType>(op.getInputB().getType());
    //
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultTy.getElementType()));
    Value newTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType());
    Value outTensor =
        rewriter.create<linalg::FillOp>(loc, zero, newTensor).getResult(0);

    if (typeA.getRank() != 2 || typeB.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Currently not support multi batch matmul.");
    }
    bool transposeA = op.getTransposeA();
    bool transposeB = op.getTransposeB();
    Operation *newOp;
    if (!transposeA && !transposeB) {
      // (A * B)
      newOp = rewriter.create<linalg::MatmulOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputA(), op.getInputB()},
          /*outputs=*/outTensor);
    } else if (transposeA && !transposeB) {
      // T(A) * B
      newOp = rewriter.create<linalg::MatmulTransposeAOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputA(), op.getInputB()},
          /*outputs=*/outTensor);
    } else if (!transposeA && transposeB) {
      // A * T(B)
      newOp = rewriter.create<linalg::MatmulTransposeBOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputA(), op.getInputB()},
          /*outputs=*/outTensor);
    } else {
      // T(B * A)
      SmallVector<int64_t> permutation{1, 0};
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/resultTy,
          /*inputs=*/ValueRange{op.getInputB(), op.getInputA()},
          /*outputs=*/outTensor);
      newOp = rewriter.create<linalg::TransposeOp>(
          /*location=*/loc,
          /*inputs=*/matmulOp.getResult(0),
          /*outputs=*/outTensor,
          /*permutation=*/permutation);
    }

    if (op.getBias()) {
      auto bias = createBroadcastOperand(loc, rewriter, resultTy, op.getBias());
      newOp = rewriter.create<linalg::AddOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/outTensor.getType(),
          /*inputs=*/newOp->getResult(0),
          /*outputs=*/bias);
    }

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
    // add lowering target
    ConversionTarget target(getContext());
    target.addIllegalDialect<onednn_graph::OneDNNGraphDialect>();
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect,
                           linalg::LinalgDialect, func::FuncDialect,
                           tensor::TensorDialect>();
    // set pattern
    RewritePatternSet patterns(ctx);
    patterns.add<AddOpLowering>(ctx);
    patterns.add<ReLUOpLowering>(ctx);
    patterns.add<MatMulOpLowering>(ctx);
    // perform conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace gc
} // namespace mlir
