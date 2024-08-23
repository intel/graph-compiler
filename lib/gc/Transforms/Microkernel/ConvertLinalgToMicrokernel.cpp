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

struct BrgemmDims {
  int64_t batchDimA;
  int64_t leadingDimA;
  int64_t minorDimA;

  int64_t batchDimB;
  int64_t leadingDimB;
  int64_t minorDimB;

  BrgemmDims() = default;
  BrgemmDims(int64_t bdA, int64_t ldA, int64_t mdA, int64_t bdB, int64_t ldB,
             int64_t mdB)
      : batchDimA(bdA), leadingDimA(ldA), minorDimA(mdA), batchDimB(bdB),
        leadingDimB(ldB), minorDimB(mdB) {}
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
  // Expecting dimPos.size() == 1 for normal dim and == 2 for vnni dim
  if (dimPos.size() > 2)
    return false;

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

// Return the position of linalg loop `dim` in the domain of `operand`.
static std::optional<unsigned> getPosInDomain(ArrayRef<unsigned> dimPos,
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

static FailureOr<BrgemmDims> inferBrgemmDims(linalg::LinalgOp linalgOp) {
  using namespace mlir::structured_match;
  auto validBrgemmMatcher = StructuredOpMatcher::make<linalg::LinalgOp>()
                                .output(MatchAll(), HasStaticShape())
                                .input(MatchAll(), HasStaticShape())
                                .output(MatchAll(), HasStaticStrides())
                                .input(MatchAll(), HasStaticStrides())
                                .operation(NumOfLoops(GreaterThanOrEqualTo(3)));
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

  unsigned mAffinePos = contractionDims->m[0];
  unsigned nAffinePos = contractionDims->n[0];
  // contractionDims.k could be of 2 cases:
  //     1. dims.k.size() == 2: non-VNNI, K = dims.k[1]
  //     2. dims.k.size() == 3: VNNI, K = dims.k[1] * dims.k[2]
  unsigned batchAffinePos = contractionDims->k.front();
  SmallVector<unsigned, 2> kAffinePos;
  if (contractionDims->k.size() == 2)
    kAffinePos = {contractionDims->k[1]};
  else if (contractionDims->k.size() == 3)
    kAffinePos = {contractionDims->k[1], contractionDims->k[2]};
  else
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmDims] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmDims] m pos in affine: " << mAffinePos
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmDims] n pos in affine: " << nAffinePos
                          << "\n");
  for (auto kp : kAffinePos)
    LLVM_DEBUG(llvm::dbgs()
               << "[inferBrgemmDims] k pos in affine: " << kp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmDims] batch pos in affine: "
                          << batchAffinePos << "\n");

  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];

  BrgemmDims brgemmDims;

  auto checkAndGetPosInDomain = [&](int64_t &dim, ArrayRef<unsigned> dimPos,
                                    OpOperand *operand) {
    auto pos = getPosInDomain(dimPos, operand, linalgOp);
    assert(pos && "Cannot find position in codomain");
    dim = *pos;
  };

  // A(batch, m, k)
  checkAndGetPosInDomain(brgemmDims.batchDimA, batchAffinePos, operandA);
  checkAndGetPosInDomain(brgemmDims.leadingDimA, {mAffinePos}, operandA);
  checkAndGetPosInDomain(brgemmDims.minorDimA, kAffinePos, operandA);
  // B(batch, k, n) or B(batch, k/vnni_step, n, vnni_step)
  // note: B does not use VNNI format K affine
  checkAndGetPosInDomain(brgemmDims.batchDimB, batchAffinePos, operandB);
  checkAndGetPosInDomain(brgemmDims.leadingDimB, {kAffinePos[0]}, operandB);
  checkAndGetPosInDomain(brgemmDims.minorDimB, {nAffinePos}, operandB);
  // C(m, n)
  // Currently useless, no need to set
  // checkAndGetPosInDomain(brgemmDims.leadingDimC, {mAffinePos}, operandC);
  // checkAndGetPosInDomain(brgemmDims.minorDimC, kAffinePos, operandC);

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmDims] A batch dim: "
                          << brgemmDims.batchDimA
                          << ", A leading dim: " << brgemmDims.leadingDimA
                          << ", A minor dim: " << brgemmDims.minorDimA
                          << "; B batch dim: " << brgemmDims.batchDimB
                          << ", B leading dim: " << brgemmDims.leadingDimB
                          << ", B minor dim: " << brgemmDims.minorDimB << "\n");
  return brgemmDims;
}

template <typename SrcBrmmOpTy>
static FailureOr<linalg::TransposeOp> getFusibleTranspose(SrcBrmmOpTy brmmOp,
                                                          Value buffer) {
  auto defOp = buffer.getDefiningOp();
  auto transOp = dyn_cast_or_null<linalg::TransposeOp>(defOp);
  if (!transOp)
    return failure();

  using one_t = std::integral_constant<size_t, 1>;
  using two_t = std::integral_constant<size_t, 2>;
  constexpr size_t lastDimOffsetA = 1;
  constexpr size_t lastDimOffsetB = std::conditional<
      std::is_same<SrcBrmmOpTy, linalg::BatchReduceMatmulOp>::value, one_t,
      two_t>::type::value;
  size_t lastDimOffset =
      buffer == brmmOp.getInputs()[0] ? lastDimOffsetA : lastDimOffsetB;

  ArrayRef<int64_t> permutation = transOp.getPermutation();
  bool lastDimContigious = true;
  // Last dim can't not be permuted if we want to incorporate the
  // transpose, because BRGEMM requires last dim to be contigious.
  // For VNNI, it requires the last two dims to be non-permutedi
  for (size_t idx = permutation.size() - lastDimOffset;
       idx < permutation.size(); idx++)
    lastDimContigious = lastDimContigious && (permutation[idx] == long(idx));

  if (lastDimContigious)
    return transOp;
  return failure();
}

// Replace linalgOp with corresponding microkernel Op
static void replaceOpWithMicrokernelOp(PatternRewriter &rewriter,
                                       linalg::LinalgOp linalgOp,
                                       const BrgemmDims &dims,
                                       const DenseMap<Value, Value> &replaceMap,
                                       bool isInitOutput) {
  OpBuilder::InsertionGuard guard(rewriter);

  DenseI64ArrayAttr batchDims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{dims.batchDimA, dims.batchDimB});
  DenseI64ArrayAttr leadingDims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{dims.leadingDimA, dims.leadingDimB});

  SmallVector<Attribute> brgemmFlags;
  if (isInitOutput) {
    brgemmFlags.push_back(microkernel::BrgemmFlagsAttr::get(
        rewriter.getContext(), microkernel::BrgemmFlags::BETA_0));
  }
  auto flags = rewriter.getArrayAttr(brgemmFlags);

  Value operandA = linalgOp.getDpsInputOperands()[0]->get();
  Value operandB = linalgOp.getDpsInputOperands()[1]->get();
  Value operandC = linalgOp.getDpsInitsMutable()[0].get();

  SmallVector<Value> inputs{operandA, operandB};
  auto brgemmOp = rewriter.replaceOpWithNewOp<microkernel::BrgemmOp>(
      linalgOp, operandC.getType(), inputs, operandC, batchDims, leadingDims,
      flags);
  // Replace operands according to fusion
  rewriter.modifyOpInPlace(brgemmOp, [&]() {
    for (const auto &pair : replaceMap)
      brgemmOp->replaceUsesOfWith(pair.first, pair.second);
  });
}

static bool isZeroArithConstant(arith::ConstantOp op) {
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
    if (!op.hasPureTensorSemantics())
      return failure();

    auto brgemmDims = inferBrgemmDims(op);
    if (failed(brgemmDims))
      return failure();

    DenseMap<Value, Value> replaceMap;
    /*
    // Check for fusible linalg::TransposeOp on operand A & B
    Value operandA = op.getDpsInputOperands()[0]->get();
    Value operandB = op.getDpsInputOperands()[1]->get();
    auto fusibleTransA = getFusibleTranspose(op, operandA);
    auto fusibleTransB = getFusibleTranspose(op, operandB);
    // Presumably minorDims are last dims and not permutated, so no need to
    // transform them
    if (!failed(fusibleTransA)) {
      ArrayRef<int64_t> permutation = fusibleTransA->getPermutation();
      brgemmDims->batchDimA = permutation[brgemmDims->batchDimA];
      brgemmDims->leadingDimA = permutation[brgemmDims->leadingDimA];
      replaceMap[fusibleTransA->getResult()[0]] = fusibleTransA->getInput();
    }
    if (!failed(fusibleTransB)) {
      ArrayRef<int64_t> permutation = fusibleTransB->getPermutation();
      brgemmDims->batchDimB = permutation[brgemmDims->batchDimB];
      brgemmDims->leadingDimB = permutation[brgemmDims->leadingDimB];
      replaceMap[fusibleTransB->getResult()[0]] = fusibleTransB->getInput();
    }

    // Check for fusible linalg::FillOp on operand C
    bool isInitOutput = false;
    Value operandC = op.getDpsInitsMutable()[0].get();
    auto defOp = operandC.getDefiningOp();
    if (llvm::isa<linalg::FillOp>(defOp)) {
      auto fillOp = dyn_cast_or_null<linalg::FillOp>(defOp);
      auto inputCst = dyn_cast_or_null<arith::ConstantOp>(
          fillOp.getInputs()[0].getDefiningOp());
      if (isZeroArithConstant(inputCst)) {
        replaceMap[fillOp.getResultTensors()[0]] = fillOp.getOutputs()[0];
        isInitOutput = true;
      }
    }
    */

    replaceOpWithMicrokernelOp(rewriter, op, *brgemmDims, replaceMap,
                               isInitOutput);
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
