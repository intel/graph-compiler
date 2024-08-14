//===- ConvertLinalgToMicrokernel.cpp - Linalg To Microkernel -*- C++ -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <tuple>

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Utils/StructuredOpMatcher.h"
#include "gc/Transforms/Utils/ValueUtils.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_EXPANDMICROKERNEL
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "expand-microkernel"

struct BrgemmInfo {
  enum BrgemmMode { STRIDE_MODE, LIST_MODE };
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batchSize;
  int64_t addrLen;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t strideA;
  int64_t strideB;

  bool isInitOutput;
  BrgemmMode mode;
};

static FailureOr<BrgemmInfo> inferBrgemmInfo(microkernel::BrgemmOp brgemmOp) {
  Value operandA = brgemmOp.getOperandA();
  Value operandB = brgemmOp.getOperandB();
  Value operandC = brgemmOp.getOperandC();

  auto checkTypeAndGetShape =
      [&](Value operand) -> FailureOr<ArrayRef<int64_t>> {
    auto operandTy = operand.getType();
    if (!llvm::isa<MemRefType>(operandTy))
      return failure();
    return dyn_cast<MemRefType>(operandTy).getShape();
  };

  auto checkAndGetDimSize =
      [&](int64_t batchDim, int64_t leadingDim,
          Value operand) -> std::tuple<FailureOr<int64_t>, FailureOr<int64_t>,
                                       FailureOr<int64_t>> {
    auto operandShape = checkTypeAndGetShape(operand);
    if (failed(operandShape))
      return {failure(), failure(), failure()};
    int64_t batchDimSize = (*operandShape)[batchDim];
    int64_t leadingDimSize = (*operandShape)[leadingDim];
    // minorDim is always last dim (the 3rd dim in 3D shape)
    int64_t minorDimSize = (*operandShape)[2];
    if (operandShape->size() == 4)
      // Input B VNNI format exists, special treatment to align with non-VNNI
      // format
      leadingDimSize *= (*operandShape)[3];
    return {batchDimSize, leadingDimSize, minorDimSize};
  };

  auto checkAndGetLdStride = [&](int64_t leadingDim,
                                 Value operand) -> FailureOr<int64_t> {
    auto operandShape = checkTypeAndGetShape(operand);
    if (failed(operandShape))
      return failure();
    auto stridesOnOperand = utils::getStaticStrides(operand);
    if (failed(stridesOnOperand))
      return failure();
    auto leadingDimStride = (*stridesOnOperand)[leadingDim];
    if (operandShape->size() == 4)
      // Input B VNNI format exists, special treatment to align with non-VNNI
      // format
      return leadingDimStride / (*operandShape)[3];
    return leadingDimStride;
  };

  auto checkAndGetBatchStride = [&](int64_t batchDim,
                                    Value operand) -> FailureOr<int64_t> {
    auto stridesOnOperand = utils::getStaticStrides(operand);
    if (failed(stridesOnOperand))
      return failure();
    return (*stridesOnOperand)[batchDim];
  };

  // A(m, k)
  auto batchDimA = brgemmOp.getBatchDimA();
  auto leadingDimA = brgemmOp.getLeadingDimA();
  auto [batchA, M, KA] = checkAndGetDimSize(batchDimA, leadingDimA, operandA);
  auto lda = checkAndGetLdStride(leadingDimA, operandA);
  if (failed(batchA) || failed(M) || failed(KA) || failed(lda))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] M, K, Lda for A: " << *M << ", "
                          << *KA << ", " << *lda << "\n");

  // B(k, n)
  auto batchDimB = brgemmOp.getBatchDimB();
  auto leadingDimB = brgemmOp.getLeadingDimB();
  auto [batchB, KB, N] = checkAndGetDimSize(batchDimB, leadingDimB, operandB);
  auto ldb = checkAndGetLdStride(leadingDimB, operandB);
  if (failed(batchB) || failed(KB) || failed(N) || failed(ldb))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] K, N, Ldb for B: " << *KB
                          << ", " << *N << ", " << *ldb << "\n");
  assert(*batchA == *batchB && *KA == *KB &&
         "Expecting matching shapes of inputs");

  // C(m, n)
  auto ldc = checkAndGetLdStride(0, operandC);
  if (failed(ldc))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Ld stride on C: " << ldc
                          << "\n");

  auto strideA = checkAndGetBatchStride(brgemmOp.getBatchDimA(), operandA);
  if (failed(strideA))
    return failure();

  auto strideB = checkAndGetBatchStride(brgemmOp.getBatchDimB(), operandB);
  if (failed(strideB))
    return failure();

  bool isInit = false;
  auto flags = brgemmOp.getFlagsAttr();
  for (auto flag : flags) {
    auto brgemmFlag = dyn_cast_or_null<microkernel::BrgemmFlagsAttr>(flag);
    if (!brgemmFlag)
      return failure();
    if (brgemmFlag.getValue() == BrgemmFlags::LIST)
      return failure();
    if (brgemmFlag.getValue() == BrgemmFlags::BETA_0)
      isInit = true;
  }

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] final BrgemmInfo: m(" << *M
                          << "), n(" << *N << "), k(" << *KB << "), batch("
                          << *batchA << "), lda(" << *lda << "), ldb(" << *ldb
                          << "), ldc(" << *ldc << "), strideA(" << *strideA
                          << "), strideB(" << *strideB << ")\n");
  BrgemmInfo info{*M,
                  *N,
                  *KA,
                  *batchA,
                  0 /* addrLen useless under stride mode */,
                  *lda,
                  *ldb,
                  *ldc,
                  *strideA,
                  *strideB,
                  isInit,
                  BrgemmInfo::STRIDE_MODE};
  return info;
}

// Replace microkernel.BrgemmOp with a set of microkernel ops
static void replaceOpWithMicrokernelOpSet(PatternRewriter &rewriter,
                                          microkernel::BrgemmOp brgemmOp,
                                          const BrgemmInfo &info) {
  assert(brgemmOp.getDpsInputs().size() == 2);
  OpBuilder::InsertionGuard guard(rewriter);

  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  Location loc = brgemmOp.getLoc();
  SmallVector<Attribute> brgemmFlags;
  if (info.isInitOutput) {
    brgemmFlags.push_back(microkernel::BrgemmFlagsAttr::get(
        rewriter.getContext(), microkernel::BrgemmFlags::BETA_0));
  }
  if (info.mode == BrgemmInfo::STRIDE_MODE) {
    brgemmFlags.push_back(microkernel::BrgemmFlagsAttr::get(
        rewriter.getContext(), microkernel::BrgemmFlags::STRIDE));
  } else if (info.mode == BrgemmInfo::LIST_MODE) {
    brgemmFlags.push_back(microkernel::BrgemmFlagsAttr::get(
        rewriter.getContext(), microkernel::BrgemmFlags::LIST));
  }

  SmallVector<Attribute, 2> brgemmDtypes{
      TypeAttr::get(getElementTypeOrSelf(brgemmOp.getDpsInputs()[0].getType())),
      TypeAttr::get(
          getElementTypeOrSelf(brgemmOp.getDpsInputs()[1].getType()))};

  // create dispatch op
  auto flags = rewriter.getArrayAttr(brgemmFlags);
  auto dtypes = rewriter.getArrayAttr(brgemmDtypes);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{info.m, info.n, info.k, info.lda, info.ldb, info.ldc,
                        info.strideA, info.strideB});
  Value dispatched = rewriter.create<microkernel::BrgemmDispatchOp>(
      loc, integer64, dims, flags, dtypes);

  // create prologue op
  rewriter.create<microkernel::BrgemmPrologueOp>(loc, dispatched);

  // create brgemm invoke op
  Value batchDim = rewriter.create<arith::ConstantOp>(
      loc, integer64, rewriter.getIntegerAttr(integer64, info.batchSize));
  Value lenDim = rewriter.create<arith::ConstantOp>(
      loc, integer64, rewriter.getIntegerAttr(integer64, info.addrLen));
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(brgemmOp->getOperands().begin(),
                        brgemmOp->getOperands().end());
  invokeOperands.push_back(batchDim);
  invokeOperands.push_back(lenDim);
  rewriter.create<microkernel::BrgemmExecuteOp>(loc, invokeOperands);

  // create epilogue op & replace original op
  rewriter.replaceOpWithNewOp<microkernel::BrgemmEpilogueOp>(brgemmOp,
                                                             dispatched);
}

class ExpandMicrokernelBrgemmRewriter
    : public OpRewritePattern<microkernel::BrgemmOp> {
public:
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(microkernel::BrgemmOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.hasPureBufferSemantics())
      return failure();

    auto brgemmInfo = inferBrgemmInfo(op);
    if (failed(brgemmInfo))
      return failure();

    replaceOpWithMicrokernelOpSet(rewriter, op, *brgemmInfo);
    return success();
  }
};

class ExpandMicrokernel
    : public impl::ExpandMicrokernelBase<ExpandMicrokernel> {
public:
  using impl::ExpandMicrokernelBase<ExpandMicrokernel>::ExpandMicrokernelBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandMicrokernelBrgemmRewriter>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace mlir::microkernel
