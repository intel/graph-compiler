//===- ConvertLinalgToMicrokernel.cpp - Linalg To Microkernel -*- C++ -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Utils/StructuredOpMatcher.h"
#include "gc/Utils/ValueUtils.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_CONVERTLINALGTOMICROKERNEL
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "convert-linalg-to-microkernel"

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

// Return the position of `dim` in the codomain of `operand`.
static std::optional<unsigned>
getPosInCodomain(unsigned dim, OpOperand *operand, linalg::LinalgOp linalgOp) {
  assert(operand->getOwner() == linalgOp);
  return linalgOp.getMatchingIndexingMap(operand).getResultPosition(
      getAffineDimExpr(dim, linalgOp.getContext()));
}

static FailureOr<BrgemmInfo>
inferBrgemmInfo(linalg::LinalgOp linalgOp,
                const linalg::ContractionDimensions &dims) {
  unsigned mPos = dims.m[0];
  unsigned nPos = dims.n[0];
  unsigned kPos = dims.k.back();
  std::optional<unsigned> batchPos;
  if (dims.k.size() == 2)
    batchPos = dims.k.front();

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] m: " << mPos << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] n: " << nPos << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] k: " << kPos << "\n");
  if (batchPos)
    LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] batch: " << batchPos << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] no batch dim\n");

  auto checkStridesAndGetLda = [&](unsigned minorDim, unsigned majorDim,
                                   OpOperand *operand) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain = getPosInCodomain(minorDim, operand, linalgOp);
    auto majorDimPosInCodomain = getPosInCodomain(majorDim, operand, linalgOp);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain)
      return failure();
    auto stridesOnOperand = gcext::utils::getStaticStrides(operand->get());
    if (failed(stridesOnOperand) ||
        (*stridesOnOperand)[*minorDimPosInCodomain] != 1)
      return failure();
    return (*stridesOnOperand)[*majorDimPosInCodomain];
  };

  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = &linalgOp.getDpsInitsMutable()[0];

  // A(m, k)
  auto lda = checkStridesAndGetLda(kPos, mPos, operandA);
  if (failed(lda))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on A: OK\n");

  // B(k, n)
  auto ldb = checkStridesAndGetLda(nPos, kPos, operandB);
  if (failed(ldb))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda(nPos, mPos, operandC);
  if (failed(ldc))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on C: OK\n");

  int64_t strideA = 1;
  int64_t strideB = 1;
  if (batchPos) {
    auto batchPosCodomainA =
        getPosInCodomain(batchPos.value(), operandA, linalgOp);
    auto stridesOnA = gcext::utils::getStaticStrides(operandA->get());
    strideA = (*stridesOnA)[*batchPosCodomainA];

    auto batchPosCodomainB =
        getPosInCodomain(batchPos.value(), operandB, linalgOp);
    auto stridesOnB = gcext::utils::getStaticStrides(operandB->get());
    strideB = (*stridesOnB)[*batchPosCodomainB];
  }

  auto loops = linalgOp.computeStaticLoopSizes();
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;

  BrgemmInfo info{loops[mPos],
                  loops[nPos],
                  loops[kPos],
                  batchVal,
                  0 /* addrLen useless under stride mode */,
                  *lda,
                  *ldb,
                  *ldc,
                  strideA,
                  strideB};
  info.isInitOutput = false;
  info.mode = BrgemmInfo::STRIDE_MODE;

  return info;
}

static FailureOr<BrgemmInfo> getBrgemmInfo(linalg::LinalgOp linalgOp) {
  using namespace mlir::gcext::utils::structured_match;
  auto validBrgemmMatcher = StructuredOpMatcher::make<linalg::LinalgOp>()
                                .output(MatchAll(), HasStaticShape())
                                .input(MatchAll(), HasStaticShape())
                                .output(MatchAll(), HasStaticStrides())
                                .input(MatchAll(), HasStaticStrides())
                                .operation(NumOfLoops(GreaterThanOrEqualTo(3)));
  // clang-format on
  if (!validBrgemmMatcher.match(linalgOp))
    return failure();

  auto contractionDims = linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Not a valid contraction\n");
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 1) ||
      contractionDims->batch.size() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Wrong dimensions\n");
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (linalgOp.getNumLoops() != classifiedLoops) {
    LLVM_DEBUG(llvm::dbgs()
               << "[checkStructure] Not all loops are classified\n");
    return failure();
  }

  return inferBrgemmInfo(linalgOp, *contractionDims);
}

// Replace linalgOp with a set of microkernel ops
static void replaceOpWithMicrokernelOpSet(PatternRewriter &rewriter,
                                          linalg::LinalgOp linalgOp,
                                          const BrgemmInfo &info) {
  assert(linalgOp.getDpsInputs().size() == 2);
  OpBuilder::InsertionGuard guard(rewriter);

  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  Location loc = linalgOp.getLoc();
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
      TypeAttr::get(getElementTypeOrSelf(linalgOp.getDpsInputs()[0].getType())),
      TypeAttr::get(
          getElementTypeOrSelf(linalgOp.getDpsInputs()[1].getType()))};

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
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  invokeOperands.push_back(batchDim);
  invokeOperands.push_back(lenDim);
  rewriter.create<microkernel::BrgemmOp>(loc, invokeOperands);

  // create epilogue op & replace original op
  rewriter.replaceOpWithNewOp<microkernel::BrgemmEpilogueOp>(linalgOp,
                                                             dispatched);
}

class ConvertBatchReduceMatmulToBrgemmRewriter
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp op,
                                PatternRewriter &rewriter) const final {
    auto brgemmInfo = getBrgemmInfo(op);
    if (failed(brgemmInfo))
      return failure();
    replaceOpWithMicrokernelOpSet(rewriter, op, *brgemmInfo);
    return success();
  }
};

class ConvertLinalgToMicrokernel
    : public impl::ConvertLinalgToMicrokernelBase<ConvertLinalgToMicrokernel> {
public:
  using impl::ConvertLinalgToMicrokernelBase<
      ConvertLinalgToMicrokernel>::ConvertLinalgToMicrokernelBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertBatchReduceMatmulToBrgemmRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace mlir::microkernel
