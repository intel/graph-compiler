//===-- ConvertLinalgToMicrokernel.cpp - Linalg To Microkernel --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

FailureOr<linalg::ContractionDimensions>
customInferContractionDims(linalg::LinalgOp linalgOp) {
  auto dims = linalg::inferContractionDims(linalgOp);
  if (failed(dims))
    return failure();
  if (llvm::isa<linalgx::BatchReduceMatmulVnniOp>(linalgOp)) {
    // For VnniOp, the K reduction dims (dim index 3 & 4) cannot be infered by
    // linalg utils because they form complex affine in operand A; Manually add
    // them here
    dims->k.push_back(3);
    dims->k.push_back(4);
  }
  return dims;
}

static bool isMatchingAffineResult(linalg::LinalgOp linalgOp, AffineExpr expr,
                                   ArrayRef<unsigned> dimPos) {
  if (dimPos.size() > 2) {
    return false;
  }
  auto firstDim = getAffineDimExpr(dimPos[0], linalgOp.getContext());
  if (dimPos.size() == 1)
    return firstDim == expr;

  // If not regular dim affine, check for VNNI format K affine
  auto secondKPosDim = getAffineDimExpr(dimPos[1], linalgOp.getContext());
  // An K affine result for VNNI should be this format:
  // d{kPos[0]} * s{kPos[1]} + d{kPos[1]} (k0 * K_vnni + k1)
  auto add = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!add)
    return false;
  if (add.getKind() != AffineExprKind::Add)
    return false;
  auto lhs = add.getLHS();
  auto rhs = add.getRHS();
  if (rhs != secondKPosDim)
    return false;
  auto mul = dyn_cast<AffineBinaryOpExpr>(lhs);
  if (!mul || mul.getKind() != AffineExprKind::Mul || mul.getLHS() != firstDim)
    return false;

  auto cst_affine = dyn_cast<AffineConstantExpr>(mul.getRHS());
  return cst_affine &&
         (cst_affine.getValue() == 2 || cst_affine.getValue() == 4);
}

// Return the position of `dim` in the codomain of `operand`.
static std::optional<unsigned> getPosInCodomain(ArrayRef<unsigned> dimPos,
                                                OpOperand *operand,
                                                linalg::LinalgOp linalgOp) {
  assert(operand->getOwner() == linalgOp);
  auto map = linalgOp.getMatchingIndexingMap(operand);
  for (unsigned i = 0, numResults = map.getNumResults(); i < numResults; i++) {
    if (isMatchingAffineResult(linalgOp, map.getResult(i), dimPos))
      return i;
  }
  return std::nullopt;
}

static FailureOr<BrgemmInfo>
inferBrgemmInfo(linalg::LinalgOp linalgOp,
                const linalg::ContractionDimensions &dims) {
  unsigned mPos = dims.m[0];
  unsigned nPos = dims.n[0];
  // dims.k could be of 2 cases:
  //     1. dims.k.size() == 2: non-VNNI, K = dims.k[1]
  //     2. dims.k.size() == 3: VNNI, K = dims.k[1] * dims.k[2]
  unsigned batchPos = dims.k.front();
  SmallVector<unsigned, 2> kPos;
  if (dims.k.size() == 2) {
    kPos = {dims.k[1]};
  } else if (dims.k.size() == 3) {
    kPos = {dims.k[1], dims.k[2]};
  } else {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] m pos in affine: " << mPos
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] n pos in affine: " << nPos
                          << "\n");
  for (auto kp : kPos) {
    LLVM_DEBUG(llvm::dbgs()
               << "[inferBrgemmInfo] k pos in affine: " << kp << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] batch pos in affine: "
                          << batchPos << "\n");

  auto checkStridesAndGetLda =
      [&](ArrayRef<unsigned> minorDim, ArrayRef<unsigned> majorDim,
          OpOperand *operand, bool allowVnni) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain = getPosInCodomain(minorDim, operand, linalgOp);
    auto majorDimPosInCodomain = getPosInCodomain(majorDim, operand, linalgOp);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain)
      return failure();
    auto stridesOnOperand = utils::getStaticStrides(operand->get());
    if (failed(stridesOnOperand))
      return failure();
    auto minorDimLd = (*stridesOnOperand)[*minorDimPosInCodomain];
    auto majorDimLd = (*stridesOnOperand)[*majorDimPosInCodomain];
    if (minorDimLd != 1) {
      // VNNI format exists, special treatment to align LD with non-VNNI format
      if (!allowVnni || (minorDimLd != 2 && minorDimLd != 4))
        return failure();
      return majorDimLd / minorDimLd;
    }
    return majorDimLd;
  };

  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = &linalgOp.getDpsInitsMutable()[0];

  // A(m, k)
  auto lda = checkStridesAndGetLda(kPos, {mPos}, operandA, false);
  if (failed(lda))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on A: OK\n");

  // B(k, n)
  // note: B does not use VNNI format K affine
  auto ldb = checkStridesAndGetLda({nPos}, {kPos[0]}, operandB, true);
  if (failed(ldb))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda({nPos}, {mPos}, operandC, false);
  if (failed(ldc))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on C: OK\n");

  int64_t strideA = 1;
  int64_t strideB = 1;
  auto batchPosCodomainA = getPosInCodomain(batchPos, operandA, linalgOp);
  auto stridesOnA = utils::getStaticStrides(operandA->get());
  strideA = (*stridesOnA)[*batchPosCodomainA];

  auto batchPosCodomainB = getPosInCodomain(batchPos, operandB, linalgOp);
  auto stridesOnB = utils::getStaticStrides(operandB->get());
  strideB = (*stridesOnB)[*batchPosCodomainB];

  auto loops = linalgOp.computeStaticLoopSizes();
  auto kSize =
      kPos.size() == 1 ? loops[kPos[0]] : (loops[kPos[0]] * loops[kPos[1]]);

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] final BrgemmInfo: m("
                          << loops[mPos] << "), n(" << loops[nPos] << "), k("
                          << kSize << "), batch(" << loops[batchPos]
                          << "), lda(" << *lda << "), ldb(" << *ldb << "), ldc("
                          << *ldc << "), strideA(" << strideA << "), strideB("
                          << strideB << ")\n");
  BrgemmInfo info{loops[mPos],
                  loops[nPos],
                  kSize,
                  loops[batchPos],
                  0 /* addrLen useless under stride mode */,
                  *lda,
                  *ldb,
                  *ldc,
                  strideA,
                  strideB,
                  false,
                  BrgemmInfo::STRIDE_MODE};
  return info;
}

static FailureOr<BrgemmInfo> getBrgemmInfo(linalg::LinalgOp linalgOp) {
  using namespace mlir::structured_match;
  auto validBrgemmMatcher = StructuredOpMatcher::make<linalg::LinalgOp>()
                                .output(MatchAll(), HasStaticShape())
                                .input(MatchAll(), HasStaticShape())
                                .output(MatchAll(), HasStaticStrides())
                                .input(MatchAll(), HasStaticStrides())
                                .operation(NumOfLoops(GreaterThanOrEqualTo(3)));
  // clang-format on
  if (!validBrgemmMatcher.match(linalgOp))
    return failure();

  auto contractionDims = customInferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Not a valid contraction\n");
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      // batch-reduce dim for BRGEMM should be identified as one of k dim
      // including VNNI & non-VNNI cases
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 3) ||
      !contractionDims->batch.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Wrong dimensions\n");
    LLVM_DEBUG(llvm::dbgs()
               << "[checkStructure] " << contractionDims->m.size() << " "
               << contractionDims->n.size() << " " << contractionDims->k.size()
               << " " << contractionDims->batch.size() << "\n");
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

bool isZeroArithConstant(arith::ConstantOp op) {
  if (!op)
    return false;

  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(op.getValue())) {
    if (intAttr.getInt() != 0)
      return false;
  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(op.getValue())) {
    if (!floatAttr.getValue().isZero())
      return false;
  } else
    return false;

  return true;
}

template <typename ContractionOp>
class ConvertContractionOpToBrgemmRewriter
    : public OpRewritePattern<ContractionOp> {
public:
  using OpRewritePattern<ContractionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ContractionOp op,
                                PatternRewriter &rewriter) const final {
    auto brgemmInfo = getBrgemmInfo(op);
    if (failed(brgemmInfo))
      return failure();
    // Check for immediately preceding linalg::FillOp
    Operation *rawOp = op;
    auto block = rawOp->getBlock();
    auto opIter = Block::iterator(rawOp);
    if (block->begin() != opIter) {
      auto prevOp = &(*(--opIter));
      if (auto fillOp = dyn_cast<linalg::FillOp>(prevOp)) {
        auto inputCst = dyn_cast_or_null<arith::ConstantOp>(
            fillOp.getInputs()[0].getDefiningOp());
        auto fillOperand = fillOp.getOutputs()[0];
        auto contractionOperand = op.getOutputs()[0];
        if (isZeroArithConstant(inputCst) &&
            contractionOperand == fillOperand) {
          brgemmInfo->isInitOutput = true;
          rewriter.eraseOp(prevOp);
        }
      }
    }
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
    patterns
        .add<ConvertContractionOpToBrgemmRewriter<linalg::BatchReduceMatmulOp>>(
            &getContext());
    patterns.add<
        ConvertContractionOpToBrgemmRewriter<linalgx::BatchReduceMatmulVnniOp>>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace mlir::microkernel
