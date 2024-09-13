//===- VectorBasedFusionAnalysis.cpp - analysis vector ops ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Analysis/VectorBasedFusionAnalysis.h"
#include "gc/Dialect/Linalgx/Utils.h"

namespace mlir {
namespace gc {

#define DEBUG_TYPE "vector-operation-analysis"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define SAFE_EXPAND(X) X
#define LDBG(X) LLVM_DEBUG(DBGS() << SAFE_EXPAND(X) << "\n")

#define ARITH_CAST_OPERATIONS                                                  \
  arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::BitcastOp,             \
      arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp, arith::UIToFPOp,      \
      arith::TruncFOp, arith::TruncIOp

#define NOT_NEED_TO_PROCESS_OP                                                 \
  linalg::BatchReduceMatmulOp, linalg::MatmulOp, linalg::BatchMatmulOp,        \
      linalg::BatchMatmulTransposeAOp, linalg::BatchMatmulTransposeBOp,        \
      linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,                  \
      linalg::QuantizedBatchMatmulOp, linalg::QuantizedMatmulOp,               \
      tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ExtractSliceOp,  \
      tensor::InsertSliceOp, microkernel::BrgemmOp

static inline bool isNotNeedToProcessOp(Operation *op) {
  return isa<NOT_NEED_TO_PROCESS_OP>(op) or
         linalgx::isAnyGenericPackedMatmulOp(op);
}

static inline bool isSpecialOp(Operation *op) {
  return isa<vector::TransposeOp, vector::ReductionOp, vector::BroadcastOp,
             vector::ShapeCastOp, vector::MultiDimReductionOp, func::CallOp>(
      op);
}

static inline bool isReadOrWriteOperation(Operation *op) {
  return isa<vector::TransferReadOp, vector::TransferWriteOp>(op);
}

///  which axis do the shape cast in source shape a
void shapeCastSourceAxis(const ArrayRef<int64_t> &a, const ArrayRef<int64_t> &b,
                         SmallVector<int64_t> &res) {
  unsigned rankA = a.size();
  unsigned rankB = b.size();
  if (rankA >= rankB)
    llvm::llvm_unreachable_internal("May be invalid shape cast operation.");

  auto isOne = [](int64_t v) { return v == 1; };

  // Special-case for n-D to 0-d shape cast. 'b' must be all ones to be shape
  // casted to a 0-d vector.
  if (rankA == 0 && all_of(b, isOne)) {
    for (size_t i = 0; i < a.size(); i++) {
      res.emplace_back(i);
    }
    return;
  }

  unsigned i = 0;
  unsigned j = 0;
  while (i < rankA && j < rankB) {
    int64_t dimA = a[i];
    int64_t dimB = 1;
    int64_t bAxisBegin = j;
    while (dimB < dimA && j < rankB)
      dimB *= b[j++];
    if (dimA != dimB) {
      llvm::llvm_unreachable_internal(" Invalid shape cast operation.");
      break;
    }
    if (bAxisBegin != j) {
      res.emplace_back(i);
    }
    ++i;

    // Handle the case when trailing dimensions are of size 1.
    // Include them into the contiguous sequence.
    if (i < rankA && all_of(a.slice(i), isOne))
      i = rankA;
    if (j < rankB && all_of(b.slice(j), isOne))
      j = rankB;
  }
  if (i != rankA or j != rankB)
    llvm_unreachable("Invalid shapecast operation.");
}

bool isScalar(Type type) {
  if (not type)
    llvm_unreachable("Not a valid type");
  if (auto vecType = dyn_cast<VectorType>(type))
    return false;
  if (auto tensorType = dyn_cast<TensorType>(type))
    return false;
  return true;
}

void getSrcBroadcastDim(const ShapedType &input, const ShapedType &output,
                        SmallVector<int64_t> &bcAxis) {
  auto inputShape = input.getShape();
  auto outputShape = output.getShape();
  // following auto_broadcast semantics
  const size_t input_rank = inputShape.size();
  const size_t output_rank = outputShape.size();
  if (output_rank < input_rank)
    llvm_unreachable("Incorrect input or output shape for broadcast op.");
  const size_t offset = output_rank - input_rank;
  for (size_t i = 0; i < input_rank; ++i) {
    if (inputShape[i] == outputShape[i + offset] ||
        (ShapedType::isDynamic(inputShape[i]) &&
         ShapedType::isDynamic(outputShape[i + offset]))) {
      bcAxis.emplace_back(i);
    }
  }
  if (bcAxis.empty())
    bcAxis.emplace_back(-1);
}

void getOperationDataAxis(Operation *op, SmallVector<int64_t> &dataAxis) {
  return TypeSwitch<Operation *>(op)
      .Case<vector::MultiDimReductionOp>(
          [&](vector::MultiDimReductionOp multiReductionOp) {
            auto rdDimsRange = multiReductionOp.getReductionDims();
            dataAxis.assign(rdDimsRange.begin(), rdDimsRange.end());
            return;
          })
      .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp shapeCastOp) {
        auto srcType = shapeCastOp.getSourceVectorType();
        auto dstType = shapeCastOp.getResultVectorType();
        auto srcShape = srcType.getShape();
        auto dstShape = dstType.getShape();
        if (srcShape.size() < dstShape.size()) {
          shapeCastSourceAxis(srcShape, dstShape, dataAxis);
        } else {
          shapeCastSourceAxis(dstShape, srcShape, dataAxis);
        }
        return;
      })
      .Case<vector::BroadcastOp>([&](vector::BroadcastOp broadcastOp) {
        auto srcType = broadcastOp.getSourceType();
        auto dstType = broadcastOp.getResultVectorType();
        if (isScalar(srcType)) {
          dataAxis.emplace_back(0);
        } else {
          auto inputType = mlir::cast<ShapedType>(srcType);
          auto outputType = mlir::cast<ShapedType>(dstType);
          getSrcBroadcastDim(inputType, outputType, dataAxis);
        }
        return;
      })
      .Case<vector::TransposeOp>([&](vector::TransposeOp transposeOp) {
        auto perm = transposeOp.getPermutation();
        int start = 0;
        for (auto x : perm) {
          if (x != start) {
            dataAxis.emplace_back(x);
          }
          start++;
        }
        return;
      })
      .Default([&](Operation *op) {
        // default is last axis
        dataAxis.emplace_back(
            cast<ShapedType>(op->getResultTypes()[0]).getRank() - 1);
        return;
      });
}

static inline bool hasSameAxis(ArrayRef<int64_t> dims1,
                               ArrayRef<int64_t> dims2) {
  DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());
  return llvm::any_of(dims1,
                      [&checkSet](int64_t x) { return checkSet.contains(x); });
}

/// whether op2 use op1 result
/// Currently we just enable this function for write and read operation
template <typename T, typename = typename std::enable_if<
                          std::is_same_v<T, vector::TransferWriteOp> ||
                              std::is_same_v<T, vector::TransferReadOp>,
                          T>>
static bool isOperationsHasDefUseRelation(Operation *op1, Operation *op2) {
  return llvm::any_of(op2->getOperands(),
                      [&op1](Value opd) { return opd.getDefiningOp() == op1; });
}

/// whether two operation has data dependency
/// op1 default is previous operation, op2 default is current operation
bool hasDataDependency(Operation *op1, Operation *op2) {
  if (!isSpecialOp(op1) and !isSpecialOp(op2))
    return false;

  if (isReadOrWriteOperation(op1) or isReadOrWriteOperation(op2)) {
    // if op1 is read the value and pass it to op2, it is not data dependency
    if (isOperationsHasDefUseRelation<vector::TransferReadOp>(op1, op2))
      return false;
  }

  // broadcast only fuse with post-op
  if (isa<vector::BroadcastOp>(op2))
    return true;

  // only special operation may cause data dependency
  if (!isSpecialOp(op1))
    return hasDataDependency(op2, op1);

  auto res =
      TypeSwitch<Operation *, bool>(op1)
          .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp shapeCastOp) {
            SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            if (!isSpecialOp(op2))
              return hasSameAxis(dims1, dims2);

            return true;
          })
          .Case<vector::MultiDimReductionOp>(
              [&](vector::MultiDimReductionOp multiReductionOp) {
                SmallVector<int64_t> dims2, reductionDims, parallelDims;
                getOperationDataAxis(op1, reductionDims);
                getOperationDataAxis(op2, dims2);
                DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());
                auto op2VectorType = getOperationVectorType(op2);
                if (!isSpecialOp(op2)) {
                  // all reduction axis should be op2's data axis
                  bool reduceDependent = false;
                  for (auto x : reductionDims) {
                    if (!checkSet.contains(x)) {
                      reduceDependent = true;
                      break;
                    }
                  }
                  if (!reduceDependent)
                    return false;

                  // all parallel axis should equal to op2's axis
                  checkSet.clear();
                  checkSet.insert(reductionDims.begin(), reductionDims.end());
                  auto rdRank =
                      multiReductionOp.getSourceVectorType().getRank();
                  for (auto i = 0; i < rdRank; i++)
                    if (not checkSet.contains(i))
                      parallelDims.emplace_back(i);

                  checkSet.clear();
                  checkSet.insert(parallelDims.begin(), parallelDims.end());
                  auto rank = op2VectorType->getRank();
                  for (auto i = 0; i < rank; i++)
                    if (!checkSet.contains(i))
                      return true;

                  return false;
                }

                return true;
              })
          .Case<vector::BroadcastOp>([&](vector::BroadcastOp broadcastOp) {
            if (isSpecialOp(op2))
              return true;

            return !OpTrait::util::staticallyKnownBroadcastable(
                getOperationVectorType(op1, false)->getShape(),
                getOperationVectorType(op2)->getShape());
          })
          .Case<vector::TransposeOp>(
              [&](vector::TransposeOp transposeOp) { return true; })
          .Default([&](Operation *op) { return false; });

  return res;
}

/// Get the operation which is not a read-write in current queue
/// \param [in, out] op
Operation *getNotReadWriteOperaiton(std::queue<Operation *> &tmpQ) {
  Operation *op = nullptr;
  while (!tmpQ.empty()) {
    Operation *cur = tmpQ.front();
    tmpQ.pop();
    if (isReadOrWriteOperation(cur))
      continue;

    op = cur;
  }
  return op;
}

/// operation should not contain for loop
bool is_innermost_operation(Operation *op) {
  bool inner_most = true;
  op->walk([&inner_most](Operation *p) {
    if (isa<scf::ForOp>(p)) {
      inner_most = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return inner_most;
}

/// whether operate on last dimension
bool isLastDim(const AffineExpr &expr, const size_t rank) {
  return isa<AffineDimExpr>(expr) &&
         dyn_cast<AffineDimExpr>(expr).getPosition() == rank - 1;
}

bool isReadWriteOnLastDim(Operation *op) {
  if (isReadOrWriteOperation(op)) {
    AffineMap permutationMap =
        dyn_cast<vector::TransferReadOp>(op)
            ? cast<vector::TransferReadOp>(op).getPermutationMap()
            : cast<vector::TransferWriteOp>(op).getPermutationMap();
    int64_t rank =
        dyn_cast<vector::TransferReadOp>(op)
            ? cast<ShapedType>(op->getOperand(0).getType()).getRank()
            : cast<ShapedType>(op->getOperand(1).getType()).getRank();
    ArrayRef<AffineExpr> dimExpr = permutationMap.getResults();
    bool find = false;
    for (const auto &expr : dimExpr)
      if (isLastDim(expr, rank)) {
        find = true;
        break;
      }

    return find;
  }
  llvm::llvm_unreachable_internal(
      "The operation is not a read or write operation.");
  return false;
}

// Filter out the operations that can be vectorized. We are only interested in
// operations that do not contain any for loops(innermost IR).
[[nodiscard]] bool filterOperation(Operation *op) {
  if (!is_innermost_operation(op)) {
    return false;
  }

  // We are only interested about the operation in vector dialect
  if (failed(getOperationVectorType(op))) {
    return false;
  }

  // We don't need to vectorize the constant operation
  if (isa<arith::ConstantOp>(op)) {
    return false;
  }

  if (isReadOrWriteOperation(op) and !isReadWriteOnLastDim(op)) {
    return false;
  }

  return true;
}

VectorType TypeHelper::getVectorzedType(Operation *op, uint32_t loopStep) {
  // Check that the operation type can be broken
  // down into a loop.
  mlir::FailureOr<VectorType> baseType = getOperationVectorType(op);
  if (failed(baseType)) {
    llvm_unreachable("Failed to get vector type for operation");
    return VectorType();
  }
  auto vectorizedType = baseType.value();
  if (loopStep == 0)
    loopStep = getDataTypeValidSteps(vectorizedType);

  return VectorType::get({loopStep}, vectorizedType.getElementType());
}

int TypeHelper::generateValidSteps(int steps, VectorType type) {
  if (type.getShape().back() >= steps)
    return steps;
  int evenStep = getNearestVectorStep(type.getShape().back());
  auto typebits = type.getElementTypeBitWidth();
  return evenStep * typebits >= 128 ? evenStep : 1;
}

// Get the maximum number of current data types that a register can hold
[[nodiscard]] int TypeHelper::getDataTypeMAXSIMDLength(VectorType type) {
  auto typebits = type.getElementTypeBitWidth();
  const int favx512bits = 512;
  const int favx2bits = 256;
  if (info.favx512f)
    return favx512bits / typebits;

  if (info.favx2)
    return favx2bits / typebits;

  // invalid hardware
  llvm_unreachable("Invalid hardware.");
  return -1;
}

/// Get a appropriate for loop step for current vector type
[[nodiscard]] int TypeHelper::getDataTypeValidSteps(VectorType type) {
  return generateValidSteps(getDataTypeMAXSIMDLength(type), type);
}

/// default op1 is previous operation
bool GroupOperationFusion::isCompatibleVectorType(Operation *op1,
                                                  Operation *op2) {
  // only lower to vector pass can produce read operation. In general two read
  // operation is compatible
  if (isa<vector::TransferReadOp>(op1) and isa<vector::TransferReadOp>(op2)) {
    return true;
  }

  mlir::FailureOr<VectorType> type1 = getOperationVectorType(op1, true);
  mlir::FailureOr<VectorType> type2 = getOperationVectorType(op2, false);
  // some operation has two different operands type like multireduction, we need
  // to check whether compitable with accumulate vector
  VectorType suppleType;
  if (failed(type1) || failed(type2))
    return false;

  auto sp1 = type1.value();
  auto sp2 = type2.value();

  auto isCompatible = [](VectorType sp1, VectorType sp2) {
    bool isCompatible = true;
    auto min_rank = std::min(sp1.getRank(), sp2.getRank());
    // from front to back
    for (long i = 0; i < min_rank; i++) {
      if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
        isCompatible = false;
        break;
      }
    }
    return isCompatible;
  };

  bool result;
  result = isCompatible(sp1, sp2);
  // operand check only happen on later operation is op2
  // TODO: may need to support other similar operation like multireduction has
  // two different operands type
  if (isa<vector::MultiDimReductionOp>(op2)) {
    suppleType = cast<VectorType>(op2->getOperandTypes()[1]);
    result |= isCompatible(suppleType, sp1);
  }

  return result;
}

void GroupOperationFusion::updateGroupBigestVectorType(VectorType vectorType) {
  int64_t rank = vectorType.getRank();
  llvm::SmallDenseMap<size_t, VectorType> &groupVectorType =
      getGroupBiggestRankVectorType();

  if (groupVectorType.contains(opGroups.size() - 1)) {
    VectorType bigestType = groupVectorType[opGroups.size() - 1];
    if (bigestType.getRank() < rank)
      groupVectorType[opGroups.size() - 1] = vectorType;

    return;
  }

  groupVectorType[opGroups.size() - 1] = vectorType;
}

void GroupOperationFusion::addOperationToGroup(Operation *op) {
  if (not op)
    llvm_unreachable("Op can't be NULL.");
  VectorType vectorType = getOperationMaxVectorType(op).value();
  if (isNeedNewGroup(op))
    opGroups.emplace_back(std::queue<Operation *>());

  if (not isa<vector::TransferReadOp>(op)) {
    updateGroupBigestVectorType(vectorType);
    while (not notNeedToJudgeOps.empty()) {
      auto cur = notNeedToJudgeOps.front();
      notNeedToJudgeOps.pop();
      opGroupIndexMap[cur] = opGroups.size() - 1;
      opGroups.back().push(cur);
    }
    opGroups.back().push(op);
    opGroupIndexMap[op] = opGroups.size() - 1;
  }
  opAnchorPos[op] = getOperationMaxVectorType(op)->getRank() - 1;
}

// We classify the operations we are interested in after filtering. Operations
// of in the same group have no data dependencies. Those operations can generate
// a same outter for loop.
void GroupOperationFusion::classifyOperations() {
  // dummpy
  if (opGroups.empty())
    opGroups.emplace_back(std::queue<Operation *>());

  func::FuncOp func = getFunction();

  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (filterOperation(op)) {
      addOperationToGroup(op);
      return WalkResult::advance();
    }
    if (isNotNeedToProcessOp(op) and !opGroups.back().empty())
      opGroups.emplace_back(std::queue<Operation *>());

    return WalkResult::advance();
  });
  // init operations results and initialization args
  groupOpResults.clear();
  groupOpInitArgs.clear();
  for (size_t i = 0; i < opGroups.size(); i++) {
    groupOpResults.emplace_back(
        llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>());
    groupOpInitArgs.emplace_back(SetVector<Value>());
  }
}

void GroupOperationFusion::run() { classifyOperations(); }

bool GroupOperationFusion::isNeedNewGroup(Operation *op) {
  if (isa<vector::TransferReadOp>(op)) {
    notNeedToJudgeOps.push(op);
    return false;
  }
  // 1. check previous operation
  if (!opGroups.back().empty()) {
    // We only care about the calculation operation.
    std::queue<Operation *> tmpQ(opGroups.back());
    Operation *prevOp = nullptr;
    prevOp = getNotReadWriteOperaiton(tmpQ);
    if (!prevOp) {
      // if previous operation is not in the same block, we need to create a
      // group
      return opGroups.back().back()->getParentOp() != op->getParentOp() or
             isSpecialOp(op);
    }

    if (prevOp->getParentOp() != op->getParentOp())
      return true;

    // special operation need to check data dependency axis
    if (hasDataDependency(prevOp, op))
      return true;

    // previous operation vector type is not compatible with current operation
    if (!isCompatibleVectorType(prevOp, op))
      return true;
  }
  return false;
}

void GroupOperationAnalysis::analysisEmptyGroup() {
  SmallVector<std::queue<Operation *>, 8> &opGroups =
      fusionStrategy.getOpGroups();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = fusionStrategy.getGroupOpResults();
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    if (grp.empty())
      continue;
    if (groupOpResults[idx].empty())
      std::queue<Operation *>().swap(grp);
  }
}

void GroupOperationAnalysis::analysisGroupMaxSteps() {
  auto &opGroups = fusionStrategy.getOpGroups();

  for (auto [idx, grp] : llvm::enumerate(opGroups)) {

    uint32_t steps = std::numeric_limits<uint32_t>::max();

    llvm::SmallVector<uint32_t, 8> &grpSteps =
        fusionStrategy.getGroupMaxSteps();
    while (idx + 1 > grpSteps.size())
      grpSteps.emplace_back(steps);

    std::queue<Operation *> tmpQueue(grp);
    auto calculateOpSteps = [&](Type type) {
      auto opType = dyn_cast<VectorType>(type);
      if (opType)
        steps = std::min(steps, (uint32_t)fusionStrategy.getTypeHelper()
                                    .getDataTypeValidSteps(opType));
    };
    while (!tmpQueue.empty()) {
      auto op = tmpQueue.front();
      tmpQueue.pop();
      if (isa<ARITH_CAST_OPERATIONS>(op))
        calculateOpSteps(op->getOperandTypes()[0]);

      calculateOpSteps(getOperationVectorType(op).value());
    }
    grpSteps[idx] = steps;
  }
}
} // namespace gc
} // namespace mlir