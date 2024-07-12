//===- ConvertLinalgToMicrokernel.cpp - Linalg To Microkernel -*- C++ -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Utils/StructuredOpMatcher.h"
#include "gc/Utils/ValueUtils.h"

namespace mlir::microkernel {
#define GEN_PASS_DEF_CONVERTLINALGTOMICROKERNEL
#include "gc/Transforms/Microkernel/MicrokernelPasses.h.inc"

#define DEBUG_TYPE "convert-linalg-to-microkernel"

class BrgemmFusionAnalysis {
public:
  struct BrgemmFusible {
    std::optional<linalg::TransposeOp> transposeA;
    std::optional<linalg::TransposeOp> transposeB;
    std::optional<linalg::FillOp> zeroInitC;
  };

private:
  // A map for linalg::brmm -> fusible ops
  DenseMap<Operation *, BrgemmFusible> brgemmFusible;
  // A set storing all fusible ops
  DenseSet<Operation *> fusibleSet;

  void addBrgemmFusible(Operation *brmm, BrgemmFusible fusible);
  SmallVector<Operation *, 10> getMemRefUseSequence(Value val);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BrgemmFusionAnalysis)
  explicit BrgemmFusionAnalysis(Operation *root, AnalysisManager &am);
  BrgemmFusible getBrgemmFusible(Operation *brmm) const {
    auto iter = brgemmFusible.find(brmm);
    if (iter == brgemmFusible.end()) {
      return BrgemmFusible();
    }
    return iter->second;
  }
  const DenseSet<Operation *> &getFusibleSet() const { return fusibleSet; }
};

void BrgemmFusionAnalysis::addBrgemmFusible(Operation *brmm,
                                            BrgemmFusible fusible) {
  return;
  auto iter = brgemmFusible.find(brmm);
  if (iter == brgemmFusible.end()) {
    brgemmFusible[brmm] = fusible;
    if (fusible.transposeA)
      fusibleSet.insert(*fusible.transposeA);
    if (fusible.transposeB)
      fusibleSet.insert(*fusible.transposeB);
    if (fusible.zeroInitC)
      fusibleSet.insert(*fusible.zeroInitC);
  } else {
    auto &origBrgemmFusible = iter->second;
    if (!origBrgemmFusible.transposeA && fusible.transposeA) {
      origBrgemmFusible.transposeA = fusible.transposeA;
      fusibleSet.insert(*fusible.transposeA);
    }
    if (!origBrgemmFusible.transposeB && fusible.transposeB) {
      origBrgemmFusible.transposeB = fusible.transposeB;
      fusibleSet.insert(*fusible.transposeB);
    }
    if (!origBrgemmFusible.zeroInitC && fusible.zeroInitC) {
      origBrgemmFusible.zeroInitC = fusible.zeroInitC;
      fusibleSet.insert(*fusible.zeroInitC);
    }
  }
}

template <typename BrmmOp>
static inline bool isFusibleBrmm(Value buffer, linalg::TransposeOp transOp,
                                 Operation *op) {
  BrmmOp brmm = dyn_cast<BrmmOp>(op);
  if (!brmm)
    return false;
  if (buffer != brmm.getInputs()[0] && buffer != brmm.getInputs()[1])
    return false;

  using one_t = std::integral_constant<size_t, 1>;
  using two_t = std::integral_constant<size_t, 2>;
  constexpr size_t lastDimOffsetA = 1;
  constexpr size_t lastDimOffsetB =
      std::conditional<std::is_same<BrmmOp, linalg::BatchReduceMatmulOp>::value,
                       one_t, two_t>::type::value;
  size_t lastDimOffset =
      buffer == brmm.getInputs()[0] ? lastDimOffsetA : lastDimOffsetB;

  ArrayRef<int64_t> permutation = transOp.getPermutation();
  bool lastDimContigious = true;
  // Last dim can't not be permuted if we want to incorporate the
  // transpose, because BRGEMM requires last dim to be contigious.
  // For VNNI, it requires the last two dims to be non-permutedi
  for (size_t idx = permutation.size() - lastDimOffset;
       idx < permutation.size(); idx++)
    lastDimContigious = lastDimContigious && (permutation[idx] == idx);

  return lastDimContigious;
}

static inline BrgemmFusionAnalysis::BrgemmFusible
generateBrgemmFusible(Value buffer, linalg::TransposeOp transOp,
                      Operation *op) {
  BrgemmFusionAnalysis::BrgemmFusible fusible;
  Value inputA, inputB;
  if (auto brmm = dyn_cast<linalg::BatchReduceMatmulOp>(op)) {
    inputA = brmm.getInputs()[0];
    inputB = brmm.getInputs()[1];
  } else if (auto brmm = dyn_cast<linalgx::BatchReduceMatmulVnniOp>(op)) {
    inputA = brmm.getInputs()[0];
    inputB = brmm.getInputs()[1];
  }
  if (buffer == inputA)
    fusible.transposeA = transOp;
  else if (buffer == inputB)
    fusible.transposeB = transOp;

  return fusible;
}

SmallVector<Operation *, 10>
BrgemmFusionAnalysis::getMemRefUseSequence(Value value) {
  auto valueType = value.getType();
  if (!isa<MemRefType>(valueType))
    return {};

  DenseSet<Operation *> users;
  Operation *creationOp = value.getDefiningOp();
  for (Operation *user : value.getUsers()) {
    users.insert(user);
  }

  // creationOp should be the first in sequence, and dominates all the others
  auto commonBlock = creationOp->getBlock();

  SmallVector<Operation *, 10> seq;
  commonBlock->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (users.find(op) != users.end())
      seq.push_back(op);
  });

  return seq;
}

BrgemmFusionAnalysis::BrgemmFusionAnalysis(Operation *root,
                                           AnalysisManager &am) {
  func::FuncOp func = dyn_cast_or_null<func::FuncOp>(root);
  if (!func)
    return;

  auto &domInfo = am.getAnalysis<DominanceInfo>();

  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // For each possible fusible Op (transposeOp/fillOp), determine whether it's
    // actually fusible, and record them if fusible.
    // Implementation:
    // For encountered transpose Op, do the following:
    // 1. Get all uses of transpose Op output;
    // 2. Generate ordered execution sequence of above uses Ops, by firstly
    // finding lowest common ancestor of all uses and then walk the ancestor;
    // 3. If the following conditions are met, then this transpose Op is added
    // to fusible:

    // 	  i. All uses after transpose Op in sequence are linalg brmm Ops;
    // 	  ii. This transpose Op could be fused into above all linalg brmm Ops;
    // 	  iii. This transpose Op must dominate these linalg brmm Ops.
    auto transposeOp = dyn_cast_or_null<linalg::TransposeOp>(op);
    if (!transposeOp)
      return;

    auto buffer = transposeOp.getInit();
    LLVM_DEBUG(llvm::dbgs()
               << "[BrgemmFusionAnalysis] Encounter `" << transposeOp
               << "` with output buffer `" << buffer << "`\n");
    auto useSequence = getMemRefUseSequence(buffer);

    bool afterTranspose = false;
    bool isFusible = true;
    SmallVector<Operation *, 5> fusingBrmmOps;
    for (auto use : useSequence) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[BrgemmFusionAnalysis] Check use: " << *use << "\n");
      if (use == transposeOp) {
        afterTranspose = true;
        continue;
      }
      if (!afterTranspose)
        continue;

      if (!domInfo.properlyDominates(transposeOp, use)) {
        isFusible = false;
        break;
      }

      if (isFusibleBrmm<linalg::BatchReduceMatmulOp>(buffer, transposeOp,
                                                     use) ||
          isFusibleBrmm<linalgx::BatchReduceMatmulVnniOp>(buffer, transposeOp,
                                                          use)) {
        fusingBrmmOps.push_back(use);
      } else {
        isFusible = false;
        break;
      }
    }

    if (isFusible) {
      for (auto op : fusingBrmmOps) {
        LLVM_DEBUG(llvm::dbgs() << "[BrgemmFusionAnalysis] `" << transposeOp
                                << "` could be fused into `" << *op << "`\n");
        addBrgemmFusible(op, generateBrgemmFusible(buffer, transposeOp, op));
      }
    }
  });
}

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
    return dims;
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
  if (dimPos.size() == 1) {
    if (firstDim == expr)
      return true;
    else
      return false;
  }
  // If not regular dim affine, check for VNNI format K affine
  auto secondKPosDim = getAffineDimExpr(dimPos[1], linalgOp.getContext());
  // An K affine result for VNNI should be this format:
  // d{kPos[0]} * s{kPos[1]} + d{kPos[1]} (k0 * K_vnni + k1)
  if (auto add = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (add.getKind() == AffineExprKind::Add) {
      auto lhs = add.getLHS();
      auto rhs = add.getRHS();
      if (rhs == secondKPosDim) {
        auto mul = dyn_cast<AffineBinaryOpExpr>(lhs);
        if (mul && mul.getKind() == AffineExprKind::Mul &&
            mul.getLHS() == firstDim) {
          if (auto cst_affine = dyn_cast<AffineConstantExpr>(mul.getRHS())) {
            if (cst_affine.getValue() == 2 || cst_affine.getValue() == 4) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

// Return the position of `dim` in the codomain of `operand`.
static FailureOr<unsigned> getPosInCodomain(ArrayRef<unsigned> dimPos,
                                            OpOperand *operand,
                                            linalg::LinalgOp linalgOp) {
  assert(operand->getOwner() == linalgOp);
  auto map = linalgOp.getMatchingIndexingMap(operand);
  for (unsigned i = 0, numResults = map.getNumResults(); i < numResults; i++) {
    if (isMatchingAffineResult(linalgOp, map.getResult(i), dimPos))
      return i;
  }
  return failure();
}

struct BrgemmOperand {
  OpOperand *operand;
  // dim pos in operand's codomain
  unsigned batchDim, minorDim, majorDim;
  bool isVnni;
};

struct BrgemmDimLoopPos {
  // dim pos in linalg loops
  unsigned mPos, nPos;
  unsigned batchPos;
  SmallVector<unsigned, 2> kPos;
};

struct BrgemmOperands {
  BrgemmDimLoopPos loopPos;
  BrgemmOperand operandA;
  BrgemmOperand operandB;
  BrgemmOperand operandC;
};

static FailureOr<BrgemmInfo> inferBrgemmInfo(linalg::LinalgOp linalgOp,
                                             const BrgemmOperands &operands) {

  auto checkStridesAndGetLd =
      [&](const BrgemmOperand &operand) -> FailureOr<int64_t> {
    auto stridesOnOperand =
        gcext::utils::getStaticStrides(operand.operand->get());
    if (failed(stridesOnOperand))
      return failure();
    auto minorDimLd = (*stridesOnOperand)[operand.minorDim];
    auto majorDimLd = (*stridesOnOperand)[operand.majorDim];
    if (minorDimLd != 1) {
      // VNNI format exists, special treatment to align LD with non-VNNI format,
      // as VNNI does not change input LD
      if (!operand.isVnni || (minorDimLd != 2 && minorDimLd != 4))
        return failure();
      return majorDimLd / minorDimLd;
    }
    return majorDimLd;
  };

  // A(m, k)
  auto lda = checkStridesAndGetLd(operands.operandA);
  if (failed(lda))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on A: OK\n");

  // B(k, n)
  // note: B does not use VNNI format K affine
  auto ldb = checkStridesAndGetLd(operands.operandB);
  if (failed(ldb))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLd(operands.operandC);
  if (failed(ldc))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] Strides on C: OK\n");

  int64_t strideA = 1;
  int64_t strideB = 1;
  auto stridesOnA =
      gcext::utils::getStaticStrides(operands.operandA.operand->get());
  strideA = (*stridesOnA)[operands.operandA.batchDim];

  auto stridesOnB =
      gcext::utils::getStaticStrides(operands.operandB.operand->get());
  strideB = (*stridesOnB)[operands.operandB.batchDim];

  const auto &loopPos = operands.loopPos;
  auto loops = linalgOp.computeStaticLoopSizes();
  auto kSize = loopPos.kPos.size() == 1
                   ? loops[loopPos.kPos[0]]
                   : (loops[loopPos.kPos[0]] * loops[loopPos.kPos[1]]);

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmInfo] final BrgemmInfo: m("
                          << loops[loopPos.mPos] << "), n("
                          << loops[loopPos.nPos] << "), k(" << kSize
                          << "), batch(" << loops[loopPos.batchPos] << "), lda("
                          << *lda << "), ldb(" << *ldb << "), ldc(" << *ldc
                          << "), strideA(" << strideA << "), strideB("
                          << strideB << ")\n");

  BrgemmInfo info{loops[loopPos.mPos],
                  loops[loopPos.nPos],
                  kSize,
                  loops[loopPos.batchPos],
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

static FailureOr<BrgemmOperands>
inferBrgemmOperands(const BrgemmFusionAnalysis &fusionAnalysis,
                    linalg::LinalgOp linalgOp) {
  auto contractionDims = customInferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Not a valid contraction\n");
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      // batch-reduce dim for BRGEMM would be identified as one of k dim
      // for both VNNI & non-VNNI cases
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 3) ||
      contractionDims->batch.size() != 0) {
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

  BrgemmOperands operands;
  BrgemmDimLoopPos &loopPos = operands.loopPos;

  loopPos.mPos = contractionDims->m[0];
  loopPos.nPos = contractionDims->n[0];
  // dims.k could be of 2 cases:
  //     1. dims.k.size() == 2: non-VNNI, K = dims.k[1]
  //     2. dims.k.size() == 3: VNNI, K = dims.k[1] * dims.k[2]
  loopPos.batchPos = contractionDims->k.front();
  if (contractionDims->k.size() == 2)
    loopPos.kPos = {contractionDims->k[1]};
  else if (contractionDims->k.size() == 3)
    loopPos.kPos = {contractionDims->k[1], contractionDims->k[2]};
  else
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmOperands] Candidate loop dims: \n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmOperands] m pos in affine: "
                          << loopPos.mPos << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmOperands] n pos in affine: "
                          << loopPos.nPos << "\n");
  for (auto kp : loopPos.kPos)
    LLVM_DEBUG(llvm::dbgs()
               << "[inferBrgemmOperands] k pos in affine: " << kp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[inferBrgemmOperands] batch pos in affine: "
                          << loopPos.batchPos << "\n");

  operands.operandA.operand = linalgOp.getDpsInputOperands()[0];
  operands.operandB.operand = linalgOp.getDpsInputOperands()[1];
  operands.operandC.operand = &linalgOp.getDpsInitsMutable()[0];

  auto getAllPosInCodomain =
      [&](BrgemmOperand &operand, ArrayRef<unsigned> minor,
          ArrayRef<unsigned> major,
          std::optional<unsigned> batch = std::nullopt) -> bool {
    auto minorDimPos = getPosInCodomain(minor, operand.operand, linalgOp);
    auto majorDimPos = getPosInCodomain(major, operand.operand, linalgOp);
    if (failed(minorDimPos) || failed(majorDimPos))
      return false;
    operand.minorDim = *minorDimPos;
    operand.majorDim = *majorDimPos;
    if (batch) {
      auto batchDimPos = getPosInCodomain(*batch, operand.operand, linalgOp);
      if (failed(batchDimPos))
        return false;
      operand.batchDim = *batchDimPos;
    }
    return true;
  };

  auto tryFusePrecedingTranspose =
      [&](std::optional<linalg::TransposeOp> transOp, BrgemmOperand &operand) {
        if (!transOp)
          return;
        // Try to incorporate the transposeOp
        ArrayRef<int64_t> permutation = transOp->getPermutation();
        bool lastDimContigious = true;
        // Last dim can't not be permuted if we want to incorporate the
        // transpose, because BRGEMM requires last dim to be contigious.
        // For VNNI, it requires the last two dims to be non-permutedi
        size_t lastDimOffset = operand.isVnni ? 2 : 1;
        for (size_t idx = permutation.size() - lastDimOffset;
             idx < permutation.size(); idx++)
          lastDimContigious = lastDimContigious && (permutation[idx] == idx);
        assert(lastDimContigious &&
               "Expecting contigious last dims for preceding transpose");
        LLVM_DEBUG(llvm::dbgs()
                   << "[inferBrgemmOperands] Fuse preceding transpose, dims "
                      "change from "
                   << operand.operand->get() << "(" << operand.batchDim << ", "
                   << operand.minorDim << ", " << operand.majorDim << ") \n");
        operand.operand = &(*transOp)->getOpOperand(0);
        operand.minorDim = permutation[operand.minorDim];
        operand.majorDim = permutation[operand.majorDim];
        operand.batchDim = permutation[operand.batchDim];
        LLVM_DEBUG(llvm::dbgs()
                   << "[inferBrgemmOperands] To " << operand.operand->get()
                   << " (" << operand.batchDim << ", " << operand.minorDim
                   << ", " << operand.majorDim << ") \n");
      };

  auto fusible = fusionAnalysis.getBrgemmFusible(linalgOp);

  // A(m, k)
  if (!getAllPosInCodomain(operands.operandA, loopPos.kPos, {loopPos.mPos},
                           loopPos.batchPos))
    return failure();
  operands.operandA.isVnni = false;
  tryFusePrecedingTranspose(fusible.transposeA, operands.operandA);

  // B(k, n)
  // note: B does not use VNNI format K affine
  if (!getAllPosInCodomain(operands.operandB, {loopPos.nPos}, {loopPos.kPos[0]},
                           loopPos.batchPos))
    return failure();
  operands.operandB.isVnni = loopPos.kPos.size() == 2;
  tryFusePrecedingTranspose(fusible.transposeB, operands.operandB);

  // C(m, n)
  if (!getAllPosInCodomain(operands.operandC, {loopPos.nPos}, {loopPos.mPos}))
    return failure();
  operands.operandC.isVnni = false;

  return operands;
}

static FailureOr<BrgemmInfo>
inferBrgemmInfo(const BrgemmFusionAnalysis &fusionAnalysis,
                linalg::LinalgOp linalgOp) {
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

  auto brgemmOperands = inferBrgemmOperands(fusionAnalysis, linalgOp);
  if (failed(brgemmOperands))
    return failure();

  return inferBrgemmInfo(linalgOp, *brgemmOperands);
}

// Replace linalgOp with a set of microkernel ops
static void replaceOpWithMicrokernelOpSet(PatternRewriter &rewriter,
                                          const BrgemmFusionAnalysis &ana,
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
  auto invoke = rewriter.create<microkernel::BrgemmOp>(loc, invokeOperands);
  // replace invoke op operands if preceding transpose could be fused
  auto fusible = ana.getBrgemmFusible(linalgOp);
  if (fusible.transposeA)
    rewriter.modifyOpInPlace(invoke, [&]() {
      invoke->replaceUsesOfWith(fusible.transposeA->getInit(),
                                fusible.transposeA->getInput());
    });
  if (fusible.transposeB)
    rewriter.modifyOpInPlace(invoke, [&]() {
      invoke->replaceUsesOfWith(fusible.transposeB->getInit(),
                                fusible.transposeB->getInput());
    });

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

template <typename ContractionOpType>
class ConvertContractionOpToBrgemmRewriter
    : public OpRewritePattern<ContractionOpType> {
private:
  const BrgemmFusionAnalysis &fusionAnalysis;

public:
  using OpRewritePattern<ContractionOpType>::OpRewritePattern;

  ConvertContractionOpToBrgemmRewriter(MLIRContext *context,
                                       BrgemmFusionAnalysis &ana)
      : OpRewritePattern<ContractionOpType>(context), fusionAnalysis(ana) {}

  LogicalResult matchAndRewrite(ContractionOpType op,
                                PatternRewriter &rewriter) const final {
    // All operands should be MemRef.
    for (auto value : op->getOperands()) {
      auto valueType = value.getType();
      if (!isa<MemRefType>(valueType)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cannot convert: expecting MemRef for all op operands\n");
        return failure();
      }
    }

    auto brgemmInfo = inferBrgemmInfo(fusionAnalysis, op);
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

    replaceOpWithMicrokernelOpSet(rewriter, fusionAnalysis, op, *brgemmInfo);
    return success();
  }
};

template <typename FusedOpType>
class CleanBrgemmFusedOpsRewriter : public OpRewritePattern<FusedOpType> {
private:
  const BrgemmFusionAnalysis &fusionAnalysis;

public:
  using OpRewritePattern<FusedOpType>::OpRewritePattern;

  CleanBrgemmFusedOpsRewriter(MLIRContext *context, BrgemmFusionAnalysis &ana)
      : OpRewritePattern<FusedOpType>(context), fusionAnalysis(ana) {}

  LogicalResult matchAndRewrite(FusedOpType op,
                                PatternRewriter &rewriter) const final {
    const auto &fusibleSet = fusionAnalysis.getFusibleSet();
    if (fusibleSet.find(op) == fusibleSet.end())
      return rewriter.notifyMatchFailure(op, "Op not fused.");
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgToMicrokernel
    : public impl::ConvertLinalgToMicrokernelBase<ConvertLinalgToMicrokernel> {
public:
  using impl::ConvertLinalgToMicrokernelBase<
      ConvertLinalgToMicrokernel>::ConvertLinalgToMicrokernelBase;
  void runOnOperation() final {
    auto &fusionAnalysis = getAnalysis<BrgemmFusionAnalysis>();

    RewritePatternSet patterns(&getContext());
    patterns
        .add<ConvertContractionOpToBrgemmRewriter<linalg::BatchReduceMatmulOp>>(
            &getContext(), fusionAnalysis);
    patterns.add<
        ConvertContractionOpToBrgemmRewriter<linalgx::BatchReduceMatmulVnniOp>>(
        &getContext(), fusionAnalysis);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();

    RewritePatternSet postPatterns(&getContext());
    postPatterns.add<CleanBrgemmFusedOpsRewriter<linalg::TransposeOp>>(
        &getContext(), fusionAnalysis);
    FrozenRewritePatternSet postPatternSet(std::move(postPatterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), postPatternSet)))
      signalPassFailure();
  }
};

} // namespace mlir::microkernel
