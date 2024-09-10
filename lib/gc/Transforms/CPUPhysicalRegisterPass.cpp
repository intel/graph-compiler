//===- CPUPhysicalRegisterPass.cpp - tiling as physical vector --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TilingVector.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CPUPHYSICALREGISTERPASS
#include "gc/Transforms/Passes.h.inc"
namespace {
#define DEBUG_TYPE "lower-to-physical-register-pass"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define SAFE_EXPAND(X) X
#define LDBG(X) LLVM_DEBUG(DBGS() << SAFE_EXPAND(X) << "\n")

#define ARITH_CAST_OPERATIONS                                                  \
  arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::BitcastOp,             \
      arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp, arith::UIToFPOp,      \
      arith::TruncFOp, arith::TruncIOp

#define NOT_NEED_TO_PROCESS_OP                                                 \
  linalgx::BatchReduceMatmulVnniOp, linalgx::MultiBatchMatmulOp,               \
      linalg::BatchReduceMatmulOp, linalgx::Mm2DVnniOp, linalgx::Mm4DVnniOp,   \
      linalg::MatmulOp, linalg::BatchMatmulOp,                                 \
      linalg::BatchMatmulTransposeAOp, linalg::BatchMatmulTransposeBOp,        \
      linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,                  \
      linalg::QuantizedBatchMatmulOp, linalg::QuantizedMatmulOp,               \
      tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ExtractSliceOp,  \
      tensor::InsertSliceOp, microkernel::BrgemmOp

/// TODO: remove it in the future
bool disableSpecialOp = false;
bool disableBroadcastOp = false;
bool enableDebugPrinter = false;

void printQueue(const std::queue<Operation *> &opQueue) {
  auto tempQ(opQueue);
  while (!tempQ.empty()) {
    auto cur = tempQ.front();
    cur->dump();
    tempQ.pop();
  }
}

void printGroupOps(SmallVector<std::queue<Operation *>, 8> &opGroups) {
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    llvm::outs() << " group id: " << idx << "\n";
    if (grp.empty()) {
      continue;
    }
    llvm::outs() << "__________________ group start_____________"
                 << "\n";
    printQueue(grp);
    llvm::outs() << "__________________ group end_____________"
                 << "\n";
  }
}

static inline bool isUsedByOtherOp(Operation *op) {
  return isa<affine::AffineApplyOp>(op);
}

static inline bool isCandidateMoveOperations(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp, tensor::EmptyOp>(
      op);
}

static inline bool isNotNeedToProcessOp(Operation *op) {
  return isa<NOT_NEED_TO_PROCESS_OP>(op);
}

static inline bool isReadOrWriteOperation(Operation *op) {
  return isa<vector::TransferReadOp, vector::TransferWriteOp>(op);
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
/// Get the index position of the first element that is true
static size_t getFirstTrueIndex(ArrayRef<bool> ararys) {
  for (size_t i = 0; i < ararys.size(); i++)
    if (!ararys[i])
      return i;

  return -1;
}

static inline bool isSpecialOp(Operation *op) {
  return isa<vector::TransposeOp, vector::ReductionOp, vector::BroadcastOp,
             vector::ShapeCastOp, vector::MultiDimReductionOp, func::CallOp>(
      op);
}

static inline void moveOpBeginingOfBlock(Operation *op) {
  Block *block = op->getBlock();
  assert(not block->getOperations().empty() && "Empty block.");
  if (&block->front() == op)
    return;
  op->moveBefore(&block->front());
}

/// find the original tensor
Value findOriginalTensor(Value writeTensor, Block *block) {
  while (auto wtOp = dyn_cast_or_null<vector::TransferWriteOp>(
             writeTensor.getDefiningOp())) {
    if (block != writeTensor.getDefiningOp()->getBlock())
      break;

    writeTensor = wtOp->getOperand(1);
  }
  return writeTensor;
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

/// whether operation is a not support operation
bool isNotSupportOperation(Operation *op) {
  return isa<vector::MaskOp, vector::ConstantMaskOp, vector::MaskedLoadOp,
             vector::MaskedStoreOp, vector::CreateMaskOp>(op);
}

/// Get vector type of the operation \param op
/// \param isPrevOp whether the operation is a previous operation, if it is not
/// prev-op, may need to use result vectortype
/// default will return the opeation result type
mlir::FailureOr<VectorType> getOperationVectorType(Operation *op,
                                                   bool isPrevOp = true) {
  if (not op)
    return failure();

  auto isDynamicType = [](VectorType &type) { return !type.hasStaticShape(); };
  auto ret =
      TypeSwitch<Operation *, mlir::FailureOr<VectorType>>(op)
          .Case<vector::TransferWriteOp>(
              [&](vector::TransferWriteOp transferWriteOp)
                  -> mlir::FailureOr<VectorType> {
                if (auto retType = dyn_cast<VectorType>(
                        transferWriteOp.getOperandTypes()[0]))
                  return retType;

                LDBG("TransferWrite Operation has wrong vector to write.");
                return failure();
              })
          .Case<vector::TransferReadOp>(
              [&](vector::TransferReadOp transferReadOp)
                  -> mlir::FailureOr<VectorType> {
                return transferReadOp.getVectorType();
              })
          .Case<vector::MultiDimReductionOp>(
              [&](vector::MultiDimReductionOp multiReductionOp) {
                if (isPrevOp)
                  return cast<VectorType>(
                      multiReductionOp->getResultTypes()[0]);

                // TODO: may need to add accumulate value vectortype
                return cast<VectorType>(multiReductionOp.getSourceVectorType());
              })
          .Default([&](Operation *op) -> mlir::FailureOr<VectorType> {
            if (isPrevOp) {
              if (op->getResultTypes().empty())
                return failure();

              if (auto shapedType =
                      dyn_cast<VectorType>(op->getResultTypes()[0]))
                return shapedType;

              return failure();
            }
            if (op->getOperandTypes().empty())
              return failure();

            if (auto shapedType =
                    dyn_cast<VectorType>(op->getOperandTypes()[0])) {
              return shapedType;
            }
            return failure();
          });
  if (!failed(ret) and isDynamicType(ret.value())) {
    return failure();
  }
  return ret;
}

/// whether the vector operation is operate on dynamic shape
bool hasDynamicShape(Operation *op) {
  if (failed(getOperationVectorType(op))) {
    return false;
  }
  auto isDynamicShapedType = [](Value x) {
    if (auto type = dyn_cast<ShapedType>(x.getType()))
      if (ShapedType::isDynamicShape(type.getShape()))
        return true;

    return false;
  };
  // Check operands data type.
  if (llvm::any_of(op->getOperands(), [&isDynamicShapedType](Value x) {
        return isDynamicShapedType(x);
      })) {
    return true;
  }

  // Check results data type.
  if (llvm::any_of(op->getResults(), [&isDynamicShapedType](OpResult x) {
        return isDynamicShapedType(x);
      })) {
    return true;
  }

  return false;
}

// TODO: Need to support these operations in the future
bool hasNotSupportOperation(func::FuncOp *func) {
  auto walkRes = func->walk([](Operation *op) {
    if (isNotSupportOperation(op)) {
      LDBG("Operation do not support yet : " << *op << "\n");
      return WalkResult::interrupt();
    }
    if (hasDynamicShape(op)) {
      LDBG("Operation has dynamic shape: " << *op << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walkRes != WalkResult::advance();
}

/// select nearest even step
int getNearestVectorStep(const int step) {
  assert(step > 0);
  int nbits = 0, n = step;
  while (n) {
    n = n >> 1;
    nbits++;
  }
  assert(nbits <= 6 || (nbits == 7 && step == 64));
  return (1 << (nbits - 1)) == step ? step : (1 << nbits);
}

/// whether operate on last dimension
bool isLastDim(const AffineExpr &expr, const size_t rank) {
  return isa<AffineDimExpr>(expr) &&
         dyn_cast<AffineDimExpr>(expr).getPosition() == rank - 1;
}

void GenerateLoopHelper::setNextAnchorArgs(
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  currentLoopStateIdxMap = nextAnchorArgsIdxMap;
  loopIterArgs = nextAnchorArgs;
}

void GenerateLoopHelper::clearNextAnchorResults() {
  nextAnchorResults.clear();
  nextAnchorResultsIdxMap.clear();
  nextAnchorResultOrignalResultMap.clear();
}

void GenerateLoopHelper::setAnchorId(size_t anchorId) noexcept {
  anchorIdx = anchorId;
}

void GenerateLoopHelper::updateDataBeforePreOpMove(
    ArrayRef<Value> loopState, std::queue<Operation *> &candidateQueue,
    std::queue<Operation *> &movedQueue) {
  loopIterArgs = loopState;
  candidateOps = &candidateQueue;
  movedOps = &movedQueue;
}

void GenerateLoopHelper::updateDataAfterPreOpMove(
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  setNextAnchorArgs(nextAnchorArgsIdxMap, nextAnchorArgs);
}

void GenerateLoopHelper::updateDataBeforePostOpMove(
    ArrayRef<Value> iterArgs, DenseMap<Value, int> &currentLoopStateIdxMap,
    DenseMap<Value, Value> &currentoriginalArgsMap,
    DenseMap<Value, Value> &currentArgsOriginalMap, ValueRange forResults,
    Block *forBlock, std::queue<Operation *> &movedQueue, size_t anchorId) {
  this->originalOperandLoopArgsMap = currentoriginalArgsMap;
  this->loopArgsOriginalOperandMap = currentArgsOriginalMap;
  this->forResults = forResults;
  this->forBlock = forBlock;
  this->anchorIdx = anchorId;
  this->currentLoopStateIdxMap = currentLoopStateIdxMap;
  this->loopIterArgs = iterArgs;
  this->movedOps = &movedQueue;
}

void GenerateLoopHelper::updateDataAfterPostOpMove(
    size_t anchorId, DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  setAnchorId(anchorId);
  setNextAnchorArgs(nextAnchorArgsIdxMap, nextAnchorArgs);
}

void GenerateLoopHelper::setNextAnchorResults(
    SmallVector<Value> &currentAnchorResults,
    DenseMap<Value, Value> &currentResultMap,
    DenseMap<Value, int> &currentResultIdxMap) {
  nextAnchorResults = std::move(currentAnchorResults);
  nextAnchorResultOrignalResultMap = std::move(currentResultMap);
  nextAnchorResultsIdxMap = std::move(currentResultIdxMap);
}

void GenerateLoopHelper::updateCurrentArgsStatus(
    DenseMap<Value, int> &currentArgsIdxMap, SmallVector<Value, 4> &currentArgs,
    DenseMap<Value, Value> &originalArgsMap,
    DenseMap<Value, Value> &argsOriginalMap) {
  setNextAnchorArgs(currentArgsIdxMap, currentArgs);
  originalOperandLoopArgsMap = originalArgsMap;
  loopArgsOriginalOperandMap = argsOriginalMap;
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
  if (HWInfo.favx512f)
    return favx512bits / typebits;

  if (HWInfo.favx2)
    return favx2bits / typebits;

  // invalid hardware
  LDBG("Please check the hardware information.");
  assert(false && "Invalid hardware.");
  return -1;
}

/// Get a appropriate for loop step for current vector type
[[nodiscard]] int TypeHelper::getDataTypeValidSteps(VectorType type) {
  return generateValidSteps(getDataTypeMAXSIMDLength(type), type);
}

/// get float or integer dense attribute
/// \param [in,out] attr
template <typename T>
void getConstantDenseAttr(TypedAttr &attr, VectorType type,
                          DenseElementsAttr denseAttr) {
  using APX = std::conditional_t<std::is_same_v<T, DenseFPElementsAttr>,
                                 APFloat, APInt>;
  attr = T::get(type, denseAttr.getSplatValue<APX>());
}

/// Create a new arith constant operation according to the dense element attr
FailureOr<Value> createArithSplatConstantOp(IRRewriter &rewriter,
                                            const Location &loc,
                                            DenseElementsAttr valueType,
                                            VectorType newOperandType) {
  if (not valueType.isSplat())
    return failure();

  TypedAttr attr;
  if (isa<FloatType>(newOperandType.getElementType())) {
    getConstantDenseAttr<DenseFPElementsAttr>(attr, newOperandType, valueType);
  } else {
    getConstantDenseAttr<DenseIntElementsAttr>(attr, newOperandType, valueType);
  }
  return rewriter.create<arith::ConstantOp>(loc, attr)->getResults()[0];
}

/// get operation vector type
/// \param isPrevOp whether the operation is a previous operation, if it is not
/// prev-op, may need to use result vectortype
/// default will return the opeation result type
mlir::FailureOr<VectorType> getOperationMaxVectorType(Operation *op) {
  if (not op)
    return failure();

  auto isDynamicType = [](VectorType &type) { return !type.hasStaticShape(); };
  auto ret =
      TypeSwitch<Operation *, mlir::FailureOr<VectorType>>(op)
          .Case<vector::TransferWriteOp>(
              [&](vector::TransferWriteOp transferWriteOp)
                  -> mlir::FailureOr<VectorType> {
                if (auto retType =
                        cast<VectorType>(transferWriteOp.getOperandTypes()[0]))
                  return retType;
                return failure();
              })
          .Case<vector::TransferReadOp>(
              [&](vector::TransferReadOp transferReadOp)
                  -> mlir::FailureOr<VectorType> {
                return transferReadOp.getVectorType();
              })
          .Case<vector::MultiDimReductionOp>(
              [&](vector::MultiDimReductionOp multiReductionOp) {
                return cast<VectorType>(multiReductionOp.getSourceVectorType());
              })
          .Default([&](Operation *op) -> mlir::FailureOr<VectorType> {
            if (op->getResultTypes().empty() and op->getOperandTypes().empty())
              return failure();

            if (op->getResultTypes().empty())
              return cast<VectorType>(op->getOperandTypes()[0]);

            if (op->getOperandTypes().empty())
              return cast<VectorType>(op->getResultTypes()[0]);

            auto opdType = cast<VectorType>(op->getOperandTypes()[0]);
            auto retType = cast<VectorType>(op->getResultTypes()[0]);
            return opdType.getRank() > retType.getRank() ? opdType : retType;
          });
  if (!failed(ret) and isDynamicType(ret.value()))
    return failure();

  return ret;
}

VectorType TypeHelper::getVectorzedType(Operation *op, uint32_t loopStep) {
  // Check that the operation type can be broken
  // down into a loop.
  mlir::FailureOr<VectorType> baseType = getOperationVectorType(op);
  if (failed(baseType)) {
    LDBG("Failed to get vector type for operation: " << *op << "\n");
    assert(0 && "Failed to get vector type for operation");
    return VectorType();
  }
  auto vectorizedType = baseType.value();
  if (loopStep == 0)
    loopStep = getDataTypeValidSteps(vectorizedType);

  return VectorType::get({loopStep}, vectorizedType.getElementType());
}

/// whether the operation result need to be returned
/// \param anchorIdx resuilt produce operation anchor position
/// \param retType resuilt return type
bool needReturnResult(std::pair<ReturnTypeKind, size_t> &retType,
                      size_t anchorIdx) {
  return retType.first != ReturnTypeKind::RT_InGroup or
         retType.second < anchorIdx;
}

union Float32Bits {
  uint32_t u;
  float f;
};

const uint32_t kF32MantiBits = 23;
const uint32_t kF32HalfMantiBitDiff = 13;
const uint32_t kF32HalfBitDiff = 16;
const Float32Bits kF32Magic = {113 << kF32MantiBits};
const uint32_t kF32HalfExpAdjust = (127 - 15) << kF32MantiBits;
const uint32_t kF32BfMantiBitDiff = 16;

/// Constructs the 16 bit representation for a half precision value from a float
/// value. This implementation is adapted from Eigen.
uint16_t float2half(float floatValue) {
  const Float32Bits inf = {255 << kF32MantiBits};
  const Float32Bits f16max = {(127 + 16) << kF32MantiBits};
  const Float32Bits denormMagic = {((127 - 15) + (kF32MantiBits - 10) + 1)
                                   << kF32MantiBits};
  uint32_t signMask = 0x80000000u;
  uint16_t halfValue = static_cast<uint16_t>(0x0u);
  Float32Bits f;
  f.f = floatValue;
  uint32_t sign = f.u & signMask;
  f.u ^= sign;

  if (f.u >= f16max.u) {
    const uint32_t halfQnan = 0x7e00;
    const uint32_t halfInf = 0x7c00;
    // Inf or NaN (all exponent bits set).
    halfValue = (f.u > inf.u) ? halfQnan : halfInf; // NaN->qNaN and Inf->Inf
  } else {
    // (De)normalized number or zero.
    if (f.u < kF32Magic.u) {
      // The resulting FP16 is subnormal or zero.
      //
      // Use a magic value to align our 10 mantissa bits at the bottom of the
      // float. As long as FP addition is round-to-nearest-even this works.
      f.f += denormMagic.f;

      halfValue = static_cast<uint16_t>(f.u - denormMagic.u);
    } else {
      uint32_t mantOdd =
          (f.u >> kF32HalfMantiBitDiff) & 1; // Resulting mantissa is odd.

      // Update exponent, rounding bias part 1. The following expressions are
      // equivalent to `f.u += ((unsigned int)(15 - 127) << kF32MantiBits) +
      // 0xfff`, but without arithmetic overflow.
      f.u += 0xc8000fffU;
      // Rounding bias part 2.
      f.u += mantOdd;
      halfValue = static_cast<uint16_t>(f.u >> kF32HalfMantiBitDiff);
    }
  }

  halfValue |= static_cast<uint16_t>(sign >> kF32HalfBitDiff);
  return halfValue;
}

/// Converts the 16 bit representation of a half precision value to a float
/// value. This implementation is adapted from Eigen.
float half2float(uint16_t halfValue) {
  const uint32_t shiftedExp =
      0x7c00 << kF32HalfMantiBitDiff; // Exponent mask after shift.

  // Initialize the float representation with the exponent/mantissa bits.
  Float32Bits f = {
      static_cast<uint32_t>((halfValue & 0x7fff) << kF32HalfMantiBitDiff)};
  const uint32_t exp = shiftedExp & f.u;
  f.u += kF32HalfExpAdjust; // Adjust the exponent

  // Handle exponent special cases.
  if (exp == shiftedExp) {
    // Inf/NaN
    f.u += kF32HalfExpAdjust;
  } else if (exp == 0) {
    // Zero/Denormal?
    f.u += 1 << kF32MantiBits;
    f.f -= kF32Magic.f;
  }

  f.u |= (halfValue & 0x8000) << kF32HalfBitDiff; // Sign bit.
  return f.f;
}

// Constructs the 16 bit representation for a bfloat value from a float value.
// This implementation is adapted from Eigen.
uint16_t float2bfloat(float floatValue) {
  if (std::isnan(floatValue))
    return std::signbit(floatValue) ? 0xFFC0 : 0x7FC0;

  Float32Bits floatBits;
  floatBits.f = floatValue;
  uint16_t bfloatBits;

  // Least significant bit of resulting bfloat.
  uint32_t lsb = (floatBits.u >> kF32BfMantiBitDiff) & 1;
  uint32_t roundingBias = 0x7fff + lsb;
  floatBits.u += roundingBias;
  bfloatBits = static_cast<uint16_t>(floatBits.u >> kF32BfMantiBitDiff);
  return bfloatBits;
}

// Converts the 16 bit representation of a bfloat value to a float value. This
// implementation is adapted from Eigen.
float bfloat2float(uint16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << kF32BfMantiBitDiff;
  return floatBits.f;
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
  LDBG("The operation is not a read or write operation." << *op << "\n");
  assert(0 && "The operation is not a read or write operation.");
  return false;
}

std::variant<float, int64_t> numeric_limits_minimum(Type type) {
  Type t1 = getElementTypeOrSelf(type);
  if (t1.isF32()) {
    return -std::numeric_limits<float>::infinity();
  } else if (t1.isBF16()) {
    return bfloat2float(float2bfloat(-std::numeric_limits<float>::infinity()));
  } else if (t1.isF16()) {
    return (float)half2float(
        float2half(-std::numeric_limits<float>::infinity()));
  } else if (t1.isSignedInteger(8)) {
    return int64_t(-128);
  } else if (t1.isSignedInteger(32)) {
    return int64_t(std::numeric_limits<int32_t>::min());
  } else if (t1.isSignlessInteger(8) or t1.isSignlessInteger(32)) {
    return int64_t(0);
  } else {
    LDBG("Unsupported data type: " << t1 << "\n");
    assert(0 && "unsupported data type");
    return (int64_t)0;
  }
}

std::variant<float, int64_t> numericLimitsMaximum(Type type) {
  Type t1 = getElementTypeOrSelf(type);
  if (t1.isF32()) {
    return std::numeric_limits<float>::infinity();
  } else if (t1.isBF16()) {
    return bfloat2float(float2bfloat(std::numeric_limits<float>::infinity()));
  } else if (t1.isF16()) {
    return (float)half2float(
        float2half(std::numeric_limits<float>::infinity()));
  } else if (t1.isSignedInteger(8)) {
    return int64_t(127);
  } else if (t1.isSignedInteger(32)) {
    return int64_t(std::numeric_limits<int32_t>::max());
  } else if (t1.isSignlessInteger(8)) {
    return int64_t(255);
  } else if (t1.isSignedInteger(32)) {
    return int64_t(std::numeric_limits<uint32_t>::max());
  } else {
    LDBG("Unsupported data type: " << t1 << "\n");
    assert(0 && "unsupported data type");
    return (int64_t)0;
  }
}

template <typename T = float>
T getInitValForReduce(vector::CombiningKind kind, Type t) {
  T result;
  Type t1 = getElementTypeOrSelf(t);

  switch (kind) {
  case vector::CombiningKind::ADD:
    if (t1.isIntOrIndex())
      result = 0;
    else if (isa<FloatType>(t1))
      result = 0.0f;
    else
      llvm_unreachable("invalid value types for ADD reduction");
    break;
  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
    assert(isa<FloatType>(t1) && "expected float values");
    result = std::get<T>(numeric_limits_minimum(t));
    break;
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    assert(isa<FloatType>(t1) && "expected float values");
    result = std::get<T>(numericLimitsMaximum(t));
    break;
  case vector::CombiningKind::MAXSI:
  case vector::CombiningKind::MAXUI:
    assert(t1.isIntOrIndex() && "expected int values");
    result = std::get<T>(numeric_limits_minimum(t));
    break;
  case vector::CombiningKind::MINSI:
  case vector::CombiningKind::MINUI:
    assert(t1.isIntOrIndex() && "expected int values");
    result = std::get<T>(numericLimitsMaximum(t));
    break;
  case vector::CombiningKind::MUL:
    if (t1.isIntOrIndex())
      result = 1;
    else if (isa<FloatType>(t1))
      result = 1.f;
    else
      llvm_unreachable("invalid value types for MUL reduction");
    break;
  default:
    llvm_unreachable("unsupported reduction kind");
  };
  return result;
}

// Since we rewrite transfer_read and transfer_write, the `permutationmap` must
// be changed.
void setOpVectorizationPermutationMap(Operation *op, OpBuilder &rewriter,
                                      const ShapedType &tensorType,
                                      const AffineMap &permutationMap) {
  auto dimExpr = permutationMap.getResults();
  auto lastDim = dyn_cast<AffineDimExpr>(dimExpr.back());
  assert(isa<AffineDimExpr>(lastDim));

  SmallVector<AffineExpr, 1> affineExprs;
  affineExprs.push_back(lastDim);
  auto destAffineMap = AffineMap::get(tensorType.getRank(), 0, affineExprs,
                                      rewriter.getContext());
  SmallVector<bool> inBounds(1, true);
  if (isa<vector::TransferWriteOp>(op)) {
    auto transferWriteOp = cast<vector::TransferWriteOp>(op);
    transferWriteOp.setPermutationMap(destAffineMap);
    transferWriteOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  } else if (isa<vector::TransferReadOp>(op)) {
    auto transferReadOp = cast<vector::TransferReadOp>(op);
    transferReadOp.setPermutationMap(destAffineMap);
    transferReadOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  }
}

// scf.for yield helper function
scf::YieldOp maybeYieldValue(OpBuilder &b, Location loc,
                             const ValueRange &value) {
  bool hasRetVal = !value.empty();
  if (hasRetVal)
    return b.create<scf::YieldOp>(loc, value);
  else
    return b.create<scf::YieldOp>(loc);
}

Operation *createTensorEmptyBefore(Operation *op) {

  auto rtType = cast<ShapedType>(op->getResultTypes()[0]);
  IRRewriter reWriter(op);
  Block *block = op->getBlock();

  reWriter.setInsertionPoint(block, block->getOperations().begin());

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i))
      dynDims.push_back(
          reWriter.create<tensor::DimOp>(op->getLoc(), op->getResult(0), i));
  }
  auto emtpyOp = reWriter.create<tensor::EmptyOp>(
      op->getLoc(), rtType.getShape(), rtType.getElementType(), dynDims);
  return emtpyOp;
}

/// get the tensor that operation should write into
Value getOperationResultTensor(
    Operation *op, DenseMap<Operation *, size_t> &visitedOperation) {
  OpResult result = op->getResults()[0];
  for (Operation *x : result.getUsers()) {
    if (not isa<vector::TransferWriteOp>(x))
      continue;

    Value sourceTensor = x->getOperands()[1];
    Operation *srcOp = sourceTensor.getDefiningOp();
    if (not visitedOperation.contains(srcOp))
      continue;

    size_t pos = visitedOperation[srcOp];
    if (pos > visitedOperation[op])
      continue;

    return sourceTensor;
  }
  LDBG("Result not write back to tensor.");

  return createTensorEmptyBefore(op)->getResults()[0];
}

Operation *createTransferWriteOpAfter(Operation *op, const Value &dest) {
  auto rtType = cast<ShapedType>(op->getResultTypes()[0]);
  int64_t rank = rtType.getRank();
  auto dstType = cast<ShapedType>(dest.getType());
  IRRewriter reWriter(op);

  auto zero = reWriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

  reWriter.setInsertionPointAfter(op);
  SmallVector<bool> inBoundsVal(rank, true);

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i))
      dynDims.push_back(
          reWriter.create<tensor::DimOp>(op->getLoc(), op->getResult(0), i));
  }
  return reWriter.create<vector::TransferWriteOp>(
      op->getLoc(),
      /*vector=*/op->getResult(0),
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(dstType.getRank(), zero),
      /*inBounds=*/inBoundsVal);
}

Operation *CanonicalizerCommonUsedData::createTransferReadOpBefore(
    Operation *op, const Value &operand, vector::TransferReadOp *srcReadOp) {
  auto operandType = cast<ShapedType>(operand.getType());

  IRRewriter rewriter(op);
  auto zero =
      rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
  auto padValue = rewriter.create<arith::ConstantOp>(
      rewriter.getUnknownLoc(),
      rewriter.getZeroAttr(operandType.getElementType()));

  if (srcReadOp) {
    auto resultType = cast<ShapedType>(srcReadOp->getType());
    SmallVector<bool> inBoundsVal(resultType.getRank(), true);
    auto srcReadOpAffineMap = srcReadOp->getPermutationMap();
    // result of read operation should be same as operand
    auto t = rewriter.create<vector::TransferReadOp>(
        op->getLoc(),
        /*vectorType=*/
        VectorType::get(resultType.getShape(), resultType.getElementType()),
        /*source=*/operand,
        /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
        /**affinemap*/ srcReadOpAffineMap,
        /*inBounds=*/inBoundsVal);
    DenseMap<Operation *, AffineMap> &permutationMap = getOpPermuationMap();
    permutationMap[t] = srcReadOpAffineMap;
    getFusionStrategy().getOpAnchorPos()[t] = t.getVectorType().getRank() - 1;

    return t;
  }
  SmallVector<bool> inBoundsVal(operandType.getRank(), true);
  auto t = rewriter.create<vector::TransferReadOp>(
      op->getLoc(),
      /*vectorType=*/
      VectorType::get(operandType.getShape(), operandType.getElementType()),
      /*source=*/operand,
      /*indices=*/SmallVector<Value>(operandType.getRank(), zero),
      /**affinemap*/ padValue,
      /*inBounds=*/inBoundsVal);
  DenseMap<Operation *, AffineMap> &permutationMap = getOpPermuationMap();
  permutationMap[t] = t.getPermutationMap();
  getFusionStrategy().getOpAnchorPos()[t] = t.getVectorType().getRank() - 1;

  return t;
}

// canonicalizing operation as tensor empty and transfer write the operation
// result into the empty tensor
[[nodiscard]] std::pair<Value, Value>
canonicalizeSourceOperation(Operation *op,
                            DenseMap<Operation *, size_t> &visitedOperation) {
  auto resultTensor = getOperationResultTensor(op, visitedOperation);
  auto writeOp = createTransferWriteOpAfter(op, resultTensor);
  return std::make_pair(resultTensor, writeOp->getResults()[0]);
}

[[nodiscard]] Value CanonicalizerCommonUsedData::canonicalizeCurrentOperation(
    Operation *op, const Value &transferReadOperand, size_t operandIdx,
    vector::TransferReadOp *srcReadOp) {
  // transfer_read operation
  auto readOp = createTransferReadOpBefore(op, transferReadOperand, srcReadOp);
  op->setOperand(operandIdx, readOp->getResults()[0]);
  return readOp->getResults()[0];
}

// __________________________________
// Speical operations canonicalization
// __________________________________

//===----------------------------------------------------------------------===//
// MultiReduce Operation
//===----------------------------------------------------------------------===//

void getOpSourceOps(Operation *op, DenseSet<Operation *> &srcOps) {
  SmallVector<Value> srcOperands = op->getOperands();
  std::deque<Value> srcOperandsQueue(srcOperands.begin(), srcOperands.end());
  DenseSet<Operation *> visited;
  visited.insert(op);
  while (!srcOperandsQueue.empty()) {
    Value accOperand = srcOperandsQueue.front();
    srcOperandsQueue.pop_front();
    Operation *accOperandOp = accOperand.getDefiningOp();
    if (!accOperandOp or visited.count(accOperandOp))
      continue;

    visited.insert(accOperandOp);
    srcOps.insert(accOperandOp);
    auto accOperandOperands = accOperandOp->getOperands();
    srcOperandsQueue.insert(srcOperandsQueue.end(), accOperandOperands.begin(),
                            accOperandOperands.end());
  }
}

bool isSrcRelated(const DenseSet<Operation *> &srcOps, Operation *op) {
  return srcOps.count(op);
}

void getPrevOps(std::queue<Operation *> &prevOps,
                std::queue<Operation *> &opQueue, Operation *currentOp) {
  while (!opQueue.empty() && currentOp != opQueue.front()) {
    prevOps.push(opQueue.front());
    opQueue.pop();
  }
}

void getPostOps(std::queue<Operation *> &postOps,
                std::queue<Operation *> &opQueue, Operation *currentOp) {
  // pop multireduction op
  assert(currentOp == opQueue.front() && "Current operation is not the front "
                                         "operation of the operation queue.");
  opQueue.pop();
  while (!opQueue.empty()) {
    postOps.push(opQueue.front());
    opQueue.pop();
  }
}

void getReductionInitAttr(vector::MultiDimReductionOp &multiReductionOp,
                          Attribute &initValueAttr) {
  auto vecType = multiReductionOp.getSourceVectorType();
  auto resultElementType = vecType.getElementType();
  if (isa<FloatType>(resultElementType))
    initValueAttr = FloatAttr::get(
        resultElementType,
        getInitValForReduce(multiReductionOp.getKind(), vecType));
  else
    initValueAttr = IntegerAttr::get(
        resultElementType,
        getInitValForReduce<int64_t>(multiReductionOp.getKind(), vecType));
}

/// get multi_reduction operation accumulate value source related operations
/// \param srcOp accumulate value source operation
void classifyAccRelatedOps(std::queue<Operation *> &accRelatedOps,
                           std::queue<Operation *> &sourceRelatedOps,
                           Operation *srcOp, std::queue<Operation *> &prevOps) {
  DenseSet<Operation *> srcOpsSet;
  getOpSourceOps(srcOp, srcOpsSet);
  while (!prevOps.empty()) {
    auto op = prevOps.front();
    prevOps.pop();
    if (isSrcRelated(srcOpsSet, op) or op == srcOp)
      accRelatedOps.push(op);
    else
      sourceRelatedOps.push(op);
  }
}

Value makeIndexArithConstantOp(OpBuilder &opBuilder, const Location &loc,
                               int64_t x) {
  return opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(), x));
}

void ForLoopGenerator::moveOperationsToCurrentForBody(
    const OpBuilder &b, std::queue<Operation *> &opQueue,
    GenerateLoopHelper &loopHelperParam) {
  auto &opPermuationMap = getOpPermuationMap();
  auto tmpQ(opQueue);
  while (!tmpQ.empty()) {
    auto x = tmpQ.front();
    tmpQ.pop();
    x->moveBefore(b.getBlock(), b.getBlock()->end());
    // check operation type to set correct operand
    setOperationCorrectOperand(x, opPermuationMap, loopHelperParam);
  }
}

void ForLoopGenerator::getResultInCurrentOps(
    const size_t anchorIdx, const size_t groupId,
    const std::queue<Operation *> &ops, SmallVector<Value, 4> &results,
    DenseMap<Value, int> &nextAnchorResultsIdxMap,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  auto tmpQ(ops);
  llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &groupResults =
      getGroupOpResults()[groupId];
  while (!tmpQ.empty()) {
    Operation *cur = tmpQ.front();
    tmpQ.pop();
    auto curResult = cur->getResults()[0];
    if (groupResults.contains(curResult)) {
      std::pair<ReturnTypeKind, size_t> retType = groupResults[curResult];
      if (needReturnResult(retType, anchorIdx)) {
        results.emplace_back(curResult);
        nextAnchorResultsIdxMap[curResult] = results.size() - 1;
        forResultOrignalResultMap[curResult] = curResult;
      }
    }
  }
}

/// update loop args related status
/// \param nextAnchorArgsIdxMap anchor args index map
/// \param nextOriginalOperandMap original value to next loop args map
/// \param nextOperandOriginalMap next loop args to original value map
void updateCurrentArgsStatus(ValueRange loopState, const size_t loopStateIdx,
                             SmallVector<Value, 4> &nextAnchorArgs,
                             Value originalValue,
                             DenseMap<Value, int> &nextAnchorArgsIdxMap,
                             DenseMap<Value, Value> &nextOriginalOperandMap,
                             DenseMap<Value, Value> &nextOperandOriginalMap) {
  Value currentArgs = loopState[loopStateIdx];
  if (currentArgs.getType() != originalValue.getType()) {
    llvm::outs() << loopStateIdx << ","
                 << "\n";
    currentArgs.dump();
    llvm::llvm_unreachable_internal("Type not equal.");
  }
  nextAnchorArgs.emplace_back(currentArgs);
  nextAnchorArgsIdxMap[currentArgs] = nextAnchorArgs.size() - 1;
  nextOriginalOperandMap[originalValue] = currentArgs;
  nextOperandOriginalMap[currentArgs] = originalValue;
}

void ForLoopGenerator::getInitArgsToNextAnchor(
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs,
    GenerateLoopHelper &loopHelperParam) {
  DenseMap<Operation *, size_t> &opAnchorPos =
      getFusionStrategy().getOpAnchorPos();
  SetVector<Value> &opInitArgs = getGroupOpInitArgs()[loopHelperParam.groupIdx];

  DenseSet<Value> visited;
  // find the next anchor arguments
  std::queue<Operation *> tmpQ(*loopHelperParam.candidateOps);
  DenseMap<Value, Value> nextOriginalOperandMap, nextOperandOriginalMap;

  while (!tmpQ.empty()) {
    Operation *cur = tmpQ.front();
    tmpQ.pop();
    auto curOperands = cur->getOperands();
    for (auto x : curOperands) {
      if (!visited.contains(x) and opInitArgs.contains(x) and
          opAnchorPos[cur] > loopHelperParam.anchorIdx) {
        assert(loopHelperParam.originalOperandLoopArgsMap.contains(x));
        int loopStateIdx = loopHelperParam.currentLoopStateIdxMap
                               [loopHelperParam.originalOperandLoopArgsMap[x]];
        updateCurrentArgsStatus(loopHelperParam.loopIterArgs, loopStateIdx,
                                nextAnchorArgs, x, nextAnchorArgsIdxMap,
                                nextOriginalOperandMap, nextOperandOriginalMap);
        visited.insert(x);
      }
    }
  }
  loopHelperParam.originalOperandLoopArgsMap =
      std::move(nextOriginalOperandMap);
  loopHelperParam.loopArgsOriginalOperandMap =
      std::move(nextOperandOriginalMap);
}

void ForLoopGenerator::getOperationInCurrentAnchor(
    const size_t anchorIdx, std::queue<Operation *> &fromQueue,
    std::queue<Operation *> &toQueue) {
  while (!fromQueue.empty()) {
    Operation *curOp = fromQueue.front();
    if (anchorIdx == getFusionStrategy().getOpAnchorPos()[curOp]) {
      toQueue.push(curOp);
      fromQueue.pop();
      continue;
    }
    break;
  }
}

void ForLoopGenerator::replaceOperationsWithForLoopResult(
    IRRewriter &rewrite, const std::queue<Operation *> &movingOperations,
    GenerateLoopHelper &loopHelperParam) {
  auto tmpQ(movingOperations);
  DenseSet<Value> operationOperands;
  while (!tmpQ.empty()) {
    auto curOp = tmpQ.front();
    tmpQ.pop();
    for (auto x : curOp->getOperands())
      operationOperands.insert(x);
  }
  auto replaceIfFn = [&](OpOperand &use) {
    return operationOperands.contains(use.get());
  };
  for (auto [nxtForResult, nextLoopResult] :
       zip(loopHelperParam.forResults, loopHelperParam.nextAnchorResults)) {
    Value originalResult =
        loopHelperParam.nextAnchorResultOrignalResultMap[nextLoopResult];

    rewrite.replaceOpUsesWithIf(originalResult.getDefiningOp(), nxtForResult,
                                replaceIfFn);
  }
}

/// \param [in,out] nextLoopStateIdxMap
/// \param [in,out] nextAnchorArgs
void ForLoopGenerator::movePreOpToCurrentAnchor(
    OpBuilder &b, DenseMap<Value, int> &nextLoopStateIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs,
    GenerateLoopHelper &loopHelperParam) {

  // 1. get operations in current anchor position
  std::queue<Operation *> movingOperation;
  getOperationInCurrentAnchor(loopHelperParam.anchorIdx,
                              *loopHelperParam.candidateOps, movingOperation);

  // 2. rewrite operation as vectorize IR
  rewriteOperationAsVectorize(b, loopHelperParam.groupIdx, &movingOperation);

  // 3. move opeartions to current for block
  moveOperationsToCurrentForBody(b, movingOperation, loopHelperParam);

  // 4. get next anchor args
  getInitArgsToNextAnchor(nextLoopStateIdxMap, nextAnchorArgs, loopHelperParam);

  // 5. move operations to moved queue
  while (!movingOperation.empty()) {
    loopHelperParam.movedOps->push(movingOperation.front());
    movingOperation.pop();
  }
}

void ForLoopGenerator::movePostOpToCurrentAnchor(
    OpBuilder &b, GenerateLoopHelper &loopHelperParam) {

  std::queue<Operation *> movingOperations;
  // 1. get post-op to current loop bod
  getOperationInCurrentAnchor(loopHelperParam.anchorIdx,
                              *loopHelperParam.candidateOps, movingOperations);
  // 2. rewrite operation as vectorize IR
  rewriteOperationAsVectorize(b, loopHelperParam.groupIdx, &movingOperations);

  // 3. move opeartions to current for block
  moveOperationsToCurrentForBody(b, movingOperations, loopHelperParam);

  // 4. replace correct for loop result to post-op
  IRRewriter rewriter(b);
  replaceOperationsWithForLoopResult(rewriter, movingOperations,
                                     loopHelperParam);

  // 5. move operations to moved queue
  while (!movingOperations.empty()) {
    loopHelperParam.movedOps->push(movingOperations.front());
    movingOperations.pop();
  }
}

void ForLoopGenerator::generateLoopResults(
    OpBuilder &b, const Location &loc, GenerateLoopHelper &loopHelperParam,
    DenseMap<Value, int> &nextOperandIdxMap) {
  SmallVector<Value, 4> results;
  DenseMap<Value, Value> currentResultMap;
  getResultInCurrentOps(loopHelperParam.anchorIdx, loopHelperParam.groupIdx,
                        *loopHelperParam.movedOps, results,
                        loopHelperParam.nextAnchorResultsIdxMap,
                        currentResultMap);

  llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &groupResults =
      getGroupOpResults()[loopHelperParam.groupIdx];
  // check for yield results whether need to return to next anchor
  for (auto [idx, forResult] :
       llvm::enumerate(loopHelperParam.nextAnchorResults)) {
    Value originalResult =
        loopHelperParam.nextAnchorResultOrignalResultMap[forResult];

    if (groupResults.contains(originalResult)) {
      std::pair<ReturnTypeKind, size_t> resultType =
          groupResults[originalResult];
      if (needReturnResult(resultType, loopHelperParam.anchorIdx)) {
        results.emplace_back(loopHelperParam.forResults[idx]);
        currentResultMap[loopHelperParam.forResults[idx]] = originalResult;
      }
    }
  }

  loopHelperParam.nextAnchorResults.clear();
  loopHelperParam.nextAnchorResultsIdxMap.clear();
  // reduction operation due to special process results size will be zero
  if (not results.empty())
    for (Value x : loopHelperParam.loopIterArgs) {
      loopHelperParam.nextAnchorResults.emplace_back(
          results[nextOperandIdxMap[x]]);
      loopHelperParam.nextAnchorResultsIdxMap[results[nextOperandIdxMap[x]]] =
          loopHelperParam.nextAnchorResults.size() - 1;
    }

  loopHelperParam.nextAnchorResultOrignalResultMap =
      std::move(currentResultMap);
}

void updateLoopArgsData(Value val, Value originalVal,
                        SmallVector<Value, 4> &argsArray,
                        DenseMap<Value, int> &anchorArgsIdxMap,
                        DenseMap<Value, Value> &originalOperandLoopArgsMap,
                        DenseMap<Value, Value> &loopArgsOriginalOperandMap) {
  argsArray.emplace_back(val);
  anchorArgsIdxMap[val] = argsArray.size() - 1;
  loopArgsOriginalOperandMap[val] = originalVal;
  originalOperandLoopArgsMap[originalVal] = val;
}

scf::ForOp ForLoopGenerator::reductionAxisGenerateForLoop(
    OpBuilder &opBuilder, const size_t reductionIdx,
    GenerateLoopHelper &loopHelperParam) {

  MultiReductionCanonicalizer rdCanonicalizer =
      getMultiRdCanonicalizers()[loopHelperParam.groupIdx];
  auto &multireductionOp = rdCanonicalizer.getCandidateOps()[0];
  VectorFusionStrategy &fusionStrategy = getFusionStrategy();

  SmallVector<std::queue<Operation *>, 8> &opGroups =
      fusionStrategy.getOpGroups();
  std::queue<Operation *> &opQueue = opGroups[loopHelperParam.groupIdx];

  const auto loc = multireductionOp->getLoc();
  SmallVector<int64_t, 4> &reductionAxis = rdCanonicalizer.getReductionAxis();
  bool lastDimReduction = rdCanonicalizer.hasLastDimReduction();
  VectorType vectorType = rdCanonicalizer.getSourceType();
  const int loopStep =
      getFusionStrategy().getGroupMaxSteps()[loopHelperParam.groupIdx];

  IRRewriter rewriterOfFunc(func);

  Value zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  Value forSteps = makeIndexArithConstantOp(
      opBuilder, loc,
      (reductionIdx == reductionAxis.size() - 1 && lastDimReduction) ? loopStep
                                                                     : 1);
  Value numIter = makeIndexArithConstantOp(
      opBuilder, loc, vectorType.getShape()[reductionAxis[reductionIdx]]);
  scf::ForOp forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);
        size_t currentAnchorId = loopHelperParam.anchorIdx;
        SmallVector<Value> tmpArgs(loopState);
        Value originalRetVal = multireductionOp->getResults()[0];

        if (reductionIdx < reductionAxis.size() - 1) {

          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          std::queue<Operation *> movedOperation;
          DenseMap<Value, Value> currentoriginalArgsMap =
              loopHelperParam.originalOperandLoopArgsMap;
          DenseMap<Value, Value> currentArgsOriginalMap =
              loopHelperParam.loopArgsOriginalOperandMap;
          DenseMap<Value, int> currentArgsIdxMap =
              loopHelperParam.currentLoopStateIdxMap;
          DenseMap<Value, Value> originalArgsMap, argsOriginalMap;
          loopHelperParam.updateDataBeforePreOpMove(tmpArgs, opQueue,
                                                    movedOperation);
          movePreOpToCurrentAnchor(b, nextAnchorArgsIdxMap, nextAnchorArgs,
                                   loopHelperParam);
          loopHelperParam.updateDataAfterPreOpMove(nextAnchorArgsIdxMap,
                                                   nextAnchorArgs);

          // replace reduction init args
          if (loopHelperParam.originalOperandLoopArgsMap.contains(
                  multireductionOp.getAcc())) {
            size_t accValIdx =
                loopHelperParam.currentLoopStateIdxMap
                    [loopHelperParam.originalOperandLoopArgsMap[multireductionOp
                                                                    .getAcc()]];
            updateCurrentArgsStatus(
                loopState, accValIdx, nextAnchorArgs, multireductionOp.getAcc(),
                nextAnchorArgsIdxMap, originalArgsMap, argsOriginalMap);
            loopHelperParam.updateCurrentArgsStatus(
                nextAnchorArgsIdxMap, nextAnchorArgs, originalArgsMap,
                argsOriginalMap);
          }

          loopHelperParam.anchorIdx += 1;
          // 2. generate next for loop
          scf::ForOp nxtFor = reductionAxisGenerateForLoop(b, reductionIdx + 1,
                                                           loopHelperParam);
          loopHelperParam.anchorIdx -= 1;

          loopHelperParam.updateDataBeforePostOpMove(
              tmpArgs, currentArgsIdxMap, currentoriginalArgsMap,
              currentArgsOriginalMap, nxtFor->getResults(), b.getBlock(),
              movedOperation, currentAnchorId);
          // 3. move postOp to current body
          movePostOpToCurrentAnchor(b, loopHelperParam);

          // 4. generate loop results
          generateLoopResults(b, loc, loopHelperParam, nextAnchorArgsIdxMap);

          // reduction must return accumulate
          if (loopHelperParam.orignalResultNextAnchorResultMap.contains(
                  originalRetVal)) {
            Value lastForResult =
                loopHelperParam
                    .orignalResultNextAnchorResultMap[originalRetVal];
            size_t retIdx = nextAnchorArgsIdxMap
                [loopHelperParam
                     .nextAnchorResultOrignalResultMap[lastForResult]];
            Value forRes = nxtFor->getResults()[retIdx];
            // accumulate for loop iter args must be last, so we just put the
            // reduction result as the last result
            updateLoopArgsData(
                forRes, originalRetVal, loopHelperParam.nextAnchorResults,
                loopHelperParam.nextAnchorResultsIdxMap,
                loopHelperParam.orignalResultNextAnchorResultMap,
                loopHelperParam.nextAnchorResultOrignalResultMap);
          }

          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);

        } else if (reductionIdx == reductionAxis.size() - 1) {
          std::queue<Operation *> movingOperation;

          while (!opQueue.empty()) {
            Operation *curOp = opQueue.front();
            opQueue.pop();
            if (isa<vector::MultiDimReductionOp>(curOp))
              break;

            movingOperation.push(curOp);
          }
          // remove all the multi_reduction operation
          while (!opQueue.empty()) {
            Operation *curOp = opQueue.front();
            if (isa<vector::MultiDimReductionOp>(curOp)) {
              opQueue.pop();
              continue;
            }
            break;
          }

          rewriteOperationAsVectorize(b, loopHelperParam.groupIdx,
                                      &movingOperation);
          loopHelperParam.loopIterArgs = loopState;
          moveOperationsToCurrentForBody(b, movingOperation, loopHelperParam);
          loopHelperParam.movedOps = &movingOperation;
          loopHelperParam.candidateOps = &opQueue;

          int accValIdx =
              loopHelperParam.currentLoopStateIdxMap
                  [loopHelperParam
                       .originalOperandLoopArgsMap[multireductionOp.getAcc()]];

          Value reductionResult = makeArithReduction(
              b, loc, multireductionOp.getKind(), multireductionOp.getSource(),
              loopState[accValIdx]);

          loopHelperParam.updateDataBeforePostOpMove(
              tmpArgs, loopHelperParam.currentLoopStateIdxMap,
              loopHelperParam.originalOperandLoopArgsMap,
              loopHelperParam.loopArgsOriginalOperandMap, ValueRange(),
              b.getBlock(), movingOperation, currentAnchorId);

          movePostOpToCurrentAnchor(b, loopHelperParam);

          loopHelperParam.nextAnchorResults.clear();
          updateLoopArgsData(reductionResult, originalRetVal,
                             loopHelperParam.nextAnchorResults,
                             loopHelperParam.nextAnchorResultsIdxMap,
                             loopHelperParam.orignalResultNextAnchorResultMap,
                             loopHelperParam.nextAnchorResultOrignalResultMap);
          getResultInCurrentOps(
              loopHelperParam.anchorIdx, loopHelperParam.groupIdx,
              movingOperation, loopHelperParam.nextAnchorResults,
              loopHelperParam.nextAnchorResultsIdxMap,
              loopHelperParam.nextAnchorResultOrignalResultMap);
          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);
        }
      });

  return forOp;
}

void ForLoopGenerator::ensureAccInParallelLoop(
    GenerateLoopHelper &loopHelperParam, ArrayRef<int64_t> parallelAxis,
    Value multiReductionAcc, DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs) {
  if (loopHelperParam.anchorIdx == parallelAxis.size() - 1) {
    // Ensure accumalate expression appear in this parallel anchor
    // position. If it not appear in current anchor, we must move it in
    // here.
    //   1. delete it in operation queue
    //   2. move it in current movedqueue
    DenseSet<Value> argsSet(nextAnchorArgs.begin(), nextAnchorArgs.end());
    std::queue<Operation *> checkAccQueue(*loopHelperParam.movedOps);
    Value accInitVal;
    while (!checkAccQueue.empty()) {
      Operation *cur = checkAccQueue.front();
      checkAccQueue.pop();
      bool ok = false;
      for (auto x : cur->getResults()) {
        if (x == multiReductionAcc) {
          accInitVal = x;
          ok = true;
          break;
        }
      }
      if (ok)
        break;
    }
    if (accInitVal) {
      // we put initVal at last for loop args
      if (!argsSet.contains(accInitVal)) {
        nextAnchorArgs.emplace_back(accInitVal);
        nextAnchorArgsIdxMap[accInitVal] = nextAnchorArgs.size() - 1;
        loopHelperParam.loopArgsOriginalOperandMap[accInitVal] =
            multiReductionAcc;
        loopHelperParam.originalOperandLoopArgsMap[multiReductionAcc] =
            accInitVal;
      }
      loopHelperParam.loopIterArgs = nextAnchorArgs;
      loopHelperParam.nextAnchorResultsIdxMap = nextAnchorArgsIdxMap;
    } else {
      llvm::llvm_unreachable_internal("Wrong accumualte source value. Because "
                                      "acc value must appear in here.");
    }
  }
}

/// Generate for loop for parallel axis of `vector.multi_reduction`.
/// This function also call reduction axis for loop
scf::ForOp ForLoopGenerator::parallelAxisGenerateForLoop(
    OpBuilder &opBuilder, GenerateLoopHelper &loopHelperParam) {
  MultiReductionCanonicalizer &rdCanonicalizer =
      getMultiRdCanonicalizers()[loopHelperParam.groupIdx];
  vector::MultiDimReductionOp &multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  VectorType vectorType = rdCanonicalizer.getSourceType();
  IRRewriter rewriterOfFunc(func);

  SmallVector<int64_t, 4> &parallelAxis = rdCanonicalizer.getParallelAxis();
  const Location &loc = multiReductionOp.getLoc();
  Value zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  size_t grpMaxStep =
      getFusionStrategy().getGroupMaxSteps()[loopHelperParam.groupIdx];
  size_t actualStep =
      (loopHelperParam.anchorIdx == parallelAxis.size() - 1 ? grpMaxStep : 1);
  Value forSteps = makeIndexArithConstantOp(opBuilder, loc, actualStep);

  // last dim reduction need to a generate dim=16 loop for fused with pre-op
  int dimSize = 0;
  if (loopHelperParam.anchorIdx == parallelAxis.size())
    dimSize = getFusionStrategy().getGroupMaxSteps()[loopHelperParam.groupIdx];
  else
    dimSize = vectorType.getShape()[parallelAxis[loopHelperParam.anchorIdx]];

  Value numIter = makeIndexArithConstantOp(opBuilder, loc, dimSize);
  // Create a loop and move vectorized operation into loops.
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);
        VectorFusionStrategy &fusionStrategy = getFusionStrategy();
        DenseMap<Operation *, size_t> &opIndexMap =
            fusionStrategy.getOpGroupIndexMap();

        assert(opIndexMap.contains(multiReductionOp) &&
               " Must constains multireduction operation.");

        size_t opIndex = opIndexMap[multiReductionOp];
        SmallVector<std::queue<Operation *>, 8> &opGroups =
            fusionStrategy.getOpGroups();
        std::queue<Operation *> &opQueue = opGroups[opIndex];
        Value multiReductionAcc = multiReductionOp.getAcc();

        if (loopHelperParam.anchorIdx < parallelAxis.size()) {
          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          std::queue<Operation *> movedQueue;
          DenseMap<Value, Value> currentOriginalOperandMap =
              loopHelperParam.originalOperandLoopArgsMap;
          DenseMap<Value, Value> currentOperandOriginalMap =
              loopHelperParam.loopArgsOriginalOperandMap;
          DenseMap<Value, int> currentLoopStateIdxMap =
              loopHelperParam.currentLoopStateIdxMap;
          SmallVector<Value> tmpArgs(loopState);
          loopHelperParam.updateDataBeforePreOpMove(tmpArgs, opQueue,
                                                    movedQueue);
          movePreOpToCurrentAnchor(b, nextAnchorArgsIdxMap, nextAnchorArgs,
                                   loopHelperParam);
          loopHelperParam.updateDataAfterPreOpMove(nextAnchorArgsIdxMap,
                                                   nextAnchorArgs);
          ensureAccInParallelLoop(loopHelperParam, parallelAxis,
                                  multiReductionAcc, nextAnchorArgsIdxMap,
                                  nextAnchorArgs);
          scf::ForOp nxtFor;
          // 2. generate next for loop
          bool useParallelLoop =
              rdCanonicalizer.hasLastDimReduction() or
              loopHelperParam.anchorIdx < parallelAxis.size() - 1;
          loopHelperParam.anchorIdx += 1;
          if (useParallelLoop) {
            nxtFor = parallelAxisGenerateForLoop(b, loopHelperParam);
          } else {
            nxtFor = reductionAxisGenerateForLoop(b, 0, loopHelperParam);
          }
          loopHelperParam.anchorIdx -= 1;

          // 3. move postOp to current body
          loopHelperParam.updateDataBeforePostOpMove(
              tmpArgs, currentLoopStateIdxMap, currentOriginalOperandMap,
              currentOperandOriginalMap, nxtFor->getResults(),
              nxtFor->getBlock(), movedQueue, loopHelperParam.anchorIdx);
          movePostOpToCurrentAnchor(b, loopHelperParam);
          // 4. generate loop results
          generateLoopResults(b, loc, loopHelperParam, nextAnchorArgsIdxMap);
          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);

        } else if (loopHelperParam.anchorIdx == parallelAxis.size()) {

          DenseMap<Value, Value> tmpOriginOperandLoopArgsMap =
              loopHelperParam.originalOperandLoopArgsMap;
          DenseMap<Value, Value> tmpLoopArgsOriginalOperandMap =
              loopHelperParam.loopArgsOriginalOperandMap;

          // get accumualte value
          Attribute initValueAttr;
          getReductionInitAttr(multiReductionOp, initValueAttr);

          auto accVal = b.create<arith::ConstantOp>(
              loc, DenseElementsAttr::get(
                       getVectorzedType(multiReductionOp, dimSize),
                       {initValueAttr}));

          // put accumulte val at first for loop args
          DenseMap<Value, int> localAnchorArgsIdxMap;
          DenseMap<Value, Value> localOriginalOperandLoopArgsMap,
              localLoopArgsOriginalOperandMap;
          SmallVector<Value, 4> argsArray;
          updateLoopArgsData(
              accVal, multiReductionAcc, argsArray, localAnchorArgsIdxMap,
              localOriginalOperandLoopArgsMap, localLoopArgsOriginalOperandMap);

          size_t accLoopStateIdx =
              loopHelperParam.currentLoopStateIdxMap
                  [loopHelperParam
                       .originalOperandLoopArgsMap[multiReductionAcc]];
          for (auto [idx, x] : llvm::enumerate(loopState)) {
            if (idx == accLoopStateIdx)
              continue;
            updateLoopArgsData(x,
                               loopHelperParam.loopArgsOriginalOperandMap
                                   [loopHelperParam.loopIterArgs[idx]],
                               argsArray, localAnchorArgsIdxMap,
                               localOriginalOperandLoopArgsMap,
                               localLoopArgsOriginalOperandMap);
          }
          loopHelperParam.updateCurrentArgsStatus(
              localAnchorArgsIdxMap, argsArray, localOriginalOperandLoopArgsMap,
              localLoopArgsOriginalOperandMap);
          DenseMap<Value, Value> originalResultForResultMap;
          auto nxtFor = reductionAxisGenerateForLoop(b, 0, loopHelperParam);

          // insert accumulate value to original vector
          Value nxtForAccVal =
              originalResultForResultMap[multiReductionOp->getResults()[0]];
          size_t accIdx = loopHelperParam.nextAnchorResultsIdxMap[nxtForAccVal];
          auto accRes = nxtFor->getResults()[accIdx];

          Operation *reductionOp = b.create<vector::ReductionOp>(
              loc, multiReductionOp.getKind(), accRes);
          auto insertOp = b.create<vector::InsertOp>(
              loc, reductionOp->getResult(0), loopState[accLoopStateIdx], iv);

          // generate loop result
          SmallVector<Value> currentAnchorResults(loopState.size());
          DenseMap<Value, Value> currentResultMap;
          DenseMap<Value, int> currentResultIdxMap;

          currentAnchorResults[accLoopStateIdx] = insertOp->getResults()[0];
          // reduce axis for loop first result we has already processed above
          currentResultMap[insertOp->getResults()[0]] =
              multiReductionOp->getResults()[0];
          currentResultIdxMap[insertOp->getResults()[0]] = accLoopStateIdx;
          for (auto [idx, x] :
               llvm::enumerate(loopHelperParam.nextAnchorResults)) {
            if (loopHelperParam.nextAnchorResultOrignalResultMap[x] ==
                multiReductionOp->getResults()[0])
              continue;

            Value originalResult =
                loopHelperParam.nextAnchorResultOrignalResultMap[x];
            size_t itrIdx = loopHelperParam.currentLoopStateIdxMap
                                [tmpOriginOperandLoopArgsMap[originalResult]];
            currentAnchorResults[itrIdx] = nxtFor->getResults()[idx];
            currentResultIdxMap[nxtFor->getResults()[idx]] = itrIdx;
            currentResultMap[nxtFor->getResults()[idx]] = originalResult;
          }
          loopHelperParam.clearNextAnchorResults();
          loopHelperParam.setNextAnchorResults(
              currentAnchorResults, currentResultMap, currentResultIdxMap);
          maybeYieldValue(b, loc, loopHelperParam.nextAnchorResults);
        }
      });
}

scf::ForOp ForLoopGenerator::generateTransposeForLoopWithLastDim(
    OpBuilder &opBuilder, const int tpSteps, const Location &loc,
    Operation *successorWriteOp, GenerateLoopHelper &loopHelperParam) {
  auto &tpCanonicalizer =
      getTransposeCanonicalizers()[loopHelperParam.groupIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getVector().getType();
  size_t rank = vtType.getRank();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  bool isTransposeDim =
      loopHelperParam.anchorIdx == tpCanonicalizer.getFirstTpIdx() or
      loopHelperParam.anchorIdx == tpCanonicalizer.getSecondTpIdx();
  auto forSteps =
      makeIndexArithConstantOp(opBuilder, loc, isTransposeDim ? tpSteps : 1);
  auto numIter = makeIndexArithConstantOp(
      opBuilder, loc, vtType.getShape()[loopHelperParam.anchorIdx]);
  VectorType kernelType =
      VectorType::get({tpSteps, tpSteps}, vtType.getElementType());
  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (loopHelperParam.anchorIdx == rank - 1) {
          // transfer read from source tensor
          Value source = tpOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(vtType.getElementType()));
          SmallVector<bool> inBoundsVal(2, true);
          inBoundsVal[0] = !ShapedType::isDynamic(
              vtType.getShape()[tpCanonicalizer.getFirstTpIdx()]);
          inBoundsVal[1] = !ShapedType::isDynamic(
              vtType.getShape()[tpCanonicalizer.getSecondTpIdx()]);

          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp.getSource(),
              /*indices=*/loopHelperParam.inductionVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);
          SmallVector<int64_t> perm{1, 0};
          auto transposeOp = b.create<vector::TransposeOp>(
              loc, transferReadOp->getResults()[0], perm);
          SmallVector<Value> writeVars(loopHelperParam.inductionVars.begin(),
                                       loopHelperParam.inductionVars.end());
          writeVars[tpCanonicalizer.getSecondTpIdx()] =
              loopHelperParam.inductionVars[tpCanonicalizer.getFirstTpIdx()];
          writeVars[tpCanonicalizer.getFirstTpIdx()] =
              loopHelperParam.inductionVars[tpCanonicalizer.getSecondTpIdx()];
          auto writeOp = b.create<vector::TransferWriteOp>(
              loc, transposeOp->getResults()[0], loopState[0], writeVars,
              inBoundsVal);
          maybeYieldValue(b, loc, writeOp->getResults());
        } else {
          // outter loop
          loopHelperParam.anchorIdx += 1;
          loopHelperParam.loopIterArgs = loopState;
          auto nxtFor = generateTransposeForLoopWithLastDim(
              b, tpSteps, loc, successorWriteOp, loopHelperParam);
          loopHelperParam.anchorIdx -= 1;
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

void ForLoopGenerator::prepareForLoopArgs(const size_t grpIdx,
                                          GenerateLoopHelper &loopHelper) {
  SetVector<Value> &grpArgs = getGroupOpInitArgs()[grpIdx];
  loopHelper.loopIterArgs = grpArgs.getArrayRef();
  for (auto [idx, val] : llvm::enumerate(grpArgs)) {
    loopHelper.currentLoopStateIdxMap[val] = idx;
    loopHelper.originalOperandLoopArgsMap[val] = val;
    loopHelper.loopArgsOriginalOperandMap[val] = val;
  }
}

void ForLoopGenerator::rearrageMultiReductionIR(
    const size_t grpIdx,
    DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap) {
  MultiReductionCanonicalizer &rdCanonicalizer =
      getMultiRdCanonicalizers()[grpIdx];
  vector::MultiDimReductionOp multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  SmallVector<int64_t, 4> &parallelAxis = rdCanonicalizer.getParallelAxis();
  SmallVector<int64_t, 4> &reductionAxis = rdCanonicalizer.getReductionAxis();
  std::queue<Operation *> &prevOps = rdCanonicalizer.getPrevOps();
  std::queue<Operation *> &postOps = rdCanonicalizer.getPostOps();
  std::queue<Operation *> &accRelatedOps = rdCanonicalizer.getAccRelatedOps();
  std::queue<Operation *> &sourceRelatedOps =
      rdCanonicalizer.getSourceRelatedOps();
  std::queue<Operation *> &opQueue = getFusionStrategy().getOpGroups()[grpIdx];
  auto copyOpQueue(opQueue);
  getPrevOps(prevOps, copyOpQueue, multiReductionOp);
  getPostOps(postOps, copyOpQueue, multiReductionOp);
  classifyAccRelatedOps(accRelatedOps, sourceRelatedOps,
                        multiReductionOp.getAcc().getDefiningOp(), prevOps);

  // mark source read operation need to set correct for loop var idx
  std::queue<Operation *> tmpSourceQ(sourceRelatedOps);
  DenseMap<size_t, size_t> varLoopIdxMap;
  VectorType groupVector =
      getFusionStrategy().getGroupBiggestRankVectorType()[grpIdx];
  for (size_t i = 0; i < parallelAxis.size(); i++) {
    varLoopIdxMap[parallelAxis[i]] = i;
  }
  size_t offset = rdCanonicalizer.hasLastDimReduction() ? 1 : 0;
  for (size_t i = parallelAxis.size() + offset;
       i < groupVector.getRank() + offset; i++) {
    varLoopIdxMap[reductionAxis[i - parallelAxis.size() - offset]] = i;
  }
  while (!tmpSourceQ.empty()) {
    auto *curOp = tmpSourceQ.front();
    tmpSourceQ.pop();
    if (isa<vector::TransferReadOp>(curOp))
      getCurrentGroupIndiceLoopMap(indiceLoopMap, grpIdx, curOp, varLoopIdxMap);
  }

  // move accumulate related operation to operation first
  std::queue<Operation *> rectifyQueue;
  DenseSet<Operation *> pushedSet;
  auto moveOperation = [&](std::queue<Operation *> &from,
                           std::queue<Operation *> &to) {
    while (!from.empty()) {
      auto cur = from.front();
      from.pop();
      if (pushedSet.contains(cur))
        continue;

      to.push(cur);
      pushedSet.insert(cur);
    }
  };
  moveOperation(accRelatedOps, rectifyQueue);
  moveOperation(opQueue, rectifyQueue);
  opQueue = rectifyQueue;
}

void ForLoopGenerator::replaceOpUsersWithForLoopResult(
    scf::ForOp forOp, int grpIdx, SmallVector<Value, 4> &nextAnchorResults,
    DenseMap<Value, int> &nextAnchorResultsIdxMap,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  IRRewriter rewriter(func);
  DenseSet<Operation *> forOpChildOps;
  forOp->walk([&](Operation *op) { forOpChildOps.insert(op); });
  auto replaceIfFn = [&](OpOperand &use) {
    return not forOpChildOps.contains(use.getOwner());
  };
  for (auto x : nextAnchorResults) {
    auto originalResult = forResultOrignalResultMap[x];
    Value forResult = forOp->getResults()[nextAnchorResultsIdxMap[x]];
    rewriter.replaceOpUsesWithIf(originalResult.getDefiningOp(), forResult,
                                 replaceIfFn);
    // subsequent group must use the replaced result as operand
    rectifyGroupOperands(grpIdx, originalResult, forResult);
  }
}
scf::ForOp
ForLoopGenerator::generateMultiReductionForLoop(const size_t grpIdx) {

  DenseMap<Operation *, DenseMap<size_t, size_t>> indiceLoopMap;
  rearrageMultiReductionIR(grpIdx, indiceLoopMap);
  // get current loop init args
  DenseMap<Value, int> currentLoopStateIdxMap, nextAnchorResultsIdxMap;
  GenerateLoopHelper loopHelper(grpIdx, 0);
  prepareForLoopArgs(grpIdx, loopHelper);

  MultiReductionCanonicalizer &rdCanonicalizer =
      getMultiRdCanonicalizers()[grpIdx];

  OpBuilder opBuilder(rdCanonicalizer.getCandidateOps()[0]);
  loopHelper.indiceLoopMap = indiceLoopMap;

  scf::ForOp forOp = parallelAxisGenerateForLoop(opBuilder, loopHelper);
  replaceOpUsersWithForLoopResult(forOp, grpIdx, loopHelper.nextAnchorResults,
                                  loopHelper.nextAnchorResultsIdxMap,
                                  loopHelper.nextAnchorResultOrignalResultMap);

  IRRewriter rewriter(func);
  vector::MultiDimReductionOp multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  rewriter.eraseOp(multiReductionOp);

  return forOp;
}

// generate simple data movement for loop
scf::ForOp ForLoopGenerator::generateTransposeScalarDataMovement(
    OpBuilder &opBuilder, const Location &loc,
    DenseMap<size_t, size_t> &tpAxisMap, GenerateLoopHelper &loopHelperParam) {
  auto &tpCanonicalizer =
      getTransposeCanonicalizers()[loopHelperParam.groupIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getSourceVectorType();
  size_t rank = vtType.getRank();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  size_t vecStep = tpCanonicalizer.transposeOnLastDim()
                       ? tpCanonicalizer.getVectorStep()
                       : 1;
  auto forSteps = makeIndexArithConstantOp(
      opBuilder, loc, loopHelperParam.anchorIdx == rank - 1 ? (vecStep) : 1);
  auto numIter = makeIndexArithConstantOp(
      opBuilder, loc, vtType.getShape()[loopHelperParam.anchorIdx]);

  SmallVector<int64_t> vecShapes(1, vecStep);
  VectorType kernelType = VectorType::get(vecShapes, vtType.getElementType());
  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelperParam.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelperParam.inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (loopHelperParam.anchorIdx == rank - 1) {
          // transfer read from source tensor
          Value source = tpOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          vector::TransferWriteOp successorWriteOp;
          for (Operation *x : tpOp->getUsers()) {
            if (isa<vector::TransferWriteOp>(x)) {
              successorWriteOp = cast<vector::TransferWriteOp>(x);
              break;
            }
          }
          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(vtType.getElementType()));
          SmallVector<bool> inBoundsVal(1, true);
          SmallVector<Value> writeVars;
          size_t itrIdx = 0;
          while (itrIdx < rank) {
            writeVars.emplace_back(
                loopHelperParam.inductionVars[tpAxisMap[itrIdx]]);
            itrIdx++;
          }
          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp.getSource(),
              /*indices=*/loopHelperParam.inductionVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);

          rectifyWriteOperationIndice(&successorWriteOp, writeVars);

          auto writeOp = b.create<vector::TransferWriteOp>(
              loc, transferReadOp->getResults()[0], loopState[0], writeVars,
              inBoundsVal);
          maybeYieldValue(b, loc, writeOp->getResults());
        } else {
          // outter loop
          loopHelperParam.anchorIdx += 1;
          loopHelperParam.loopIterArgs = loopState;
          auto nxtFor = generateTransposeScalarDataMovement(b, loc, tpAxisMap,
                                                            loopHelperParam);
          loopHelperParam.anchorIdx -= 1;
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

scf::ForOp ForLoopGenerator::generateShapeCastReadWriteLoop(
    OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
    const size_t steps, const Location &loc, SmallVector<Value> &inductionVars,
    ValueRange iterArgs) {
  auto &scCanonicalizer = getShapeCastCanonicalizers()[grpIdx];
  vector::ShapeCastOp &scOp = scCanonicalizer.getCandidateOps()[0];
  VectorType sourceType = scOp.getSourceVectorType();
  VectorType destType = scOp.getResultVectorType();
  VectorType loopType =
      sourceType.getRank() > destType.getRank() ? sourceType : destType;
  size_t rank = loopType.getRank();
  DenseMap<Operation *, size_t> &opIndexMap =
      getFusionStrategy().getOpGroupIndexMap();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  bool isLastDim = loopType.getRank() - 1 == (int64_t)forDimIdx;
  auto forSteps =
      makeIndexArithConstantOp(opBuilder, loc, isLastDim ? steps : 1);
  auto numIter =
      makeIndexArithConstantOp(opBuilder, loc, loopType.getShape()[forDimIdx]);
  VectorType kernelType =
      VectorType::get({(int64_t)steps}, loopType.getElementType());

  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (forDimIdx == rank - 1) {
          // transfer read from source tensor
          Value source = scOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          SmallVector<vector::TransferWriteOp> successorWriteOps;
          for (Operation *x : scOp->getUsers()) {
            if (isa<vector::TransferWriteOp>(x) and opIndexMap.contains(x) and
                opIndexMap[x] == opIndexMap[scOp]) {
              successorWriteOps.emplace_back(cast<vector::TransferWriteOp>(x));
            }
          }
          SmallVector<AffineExpr> exprs(loopType.getRank(), AffineExpr());
          bindSymbolsList<AffineExpr>(b.getContext(), exprs);
          SmallVector<Value> operands{inductionVars.begin(),
                                      inductionVars.end()};
          SmallVector<Value> smallRankShapeVars;

          auto getSmallRankShapeVars = [&](VectorType smallType) {
            size_t itrIdx = 0;
            SmallVector<bool> visitedAxis(rank, false);
            while ((int64_t)itrIdx < smallType.getRank()) {

              size_t endShape = getFirstTrueIndex(visitedAxis), dimSize = 1;
              assert(endShape < rank and endShape >= 0 && "Invalid endShape");
              // skip non corresponding axis
              // e.g.: vector<32x16x1x32xbf16> -> vector<1x512x32xbf16>
              while (loopType.getShape()[endShape] >
                     smallType.getShape()[itrIdx]) {
                endShape++;
              }
              const size_t expandIdx = endShape;
              while (endShape < rank) {
                visitedAxis[endShape] = true;
                dimSize *= loopType.getShape()[endShape];
                if ((int64_t)dimSize == smallType.getShape()[itrIdx]) {
                  break;
                }
                endShape += 1;
              }
              const size_t expandSize = endShape - expandIdx + 1;
              AffineExpr calculateOffset;
              SmallVector<Value> offsetVars;

              for (size_t i = 0; i < expandSize; i++) {
                size_t startIdx = i + 1;
                size_t otherDimsSize = 1;
                while (startIdx < expandSize) {
                  otherDimsSize *= (loopType.getShape()[startIdx + expandIdx]);
                  startIdx++;
                }
                AffineExpr dimSize =
                    getAffineConstantExpr(otherDimsSize, b.getContext());
                if (i == 0) {
                  calculateOffset = exprs[i] * dimSize;
                } else {
                  calculateOffset = calculateOffset + exprs[i] * dimSize;
                }

                offsetVars.emplace_back(inductionVars[i + expandIdx]);
              }
              AffineMap map = AffineMap::get(0, expandSize, calculateOffset);

              Value offset =
                  b.createOrFold<affine::AffineApplyOp>(loc, map, offsetVars);
              smallRankShapeVars.emplace_back(offset);
              itrIdx++;
            }
          };

          if (loopType == sourceType) {
            getSmallRankShapeVars(destType);
          } else {
            getSmallRankShapeVars(sourceType);
          }

          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(loopType.getElementType()));

          SmallVector<bool> inBoundsVal(1, true);

          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp->getOperands()[0],
              /*indices=*/loopType == sourceType ? inductionVars
                                                 : smallRankShapeVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);

          SmallVector<Value> writeVars =
              loopType == sourceType ? smallRankShapeVars : inductionVars;
          SmallVector<Value> writeResults;
          for (auto successorWriteOp : successorWriteOps) {
            rectifyWriteOperationIndice(&successorWriteOp, writeVars);
            auto writeOp = b.create<vector::TransferWriteOp>(
                loc, transferReadOp->getResults()[0], loopState[0], writeVars,
                inBoundsVal);
            writeResults.emplace_back(writeOp->getResults()[0]);
          }
          maybeYieldValue(b, loc, writeResults);
        } else {
          // outter loop
          auto nxtFor = generateShapeCastReadWriteLoop(
              b, grpIdx, forDimIdx + 1, steps, loc, inductionVars, loopState);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

void ForLoopGenerator::rectifyWriteOperationIndice(
    vector::TransferWriteOp *originalWriteOp,
    SmallVectorImpl<Value> &writeVars) {
  VectorType sucessWriteVectorType = originalWriteOp->getVectorType();
  ShapedType successWriteTensorType =
      cast<ShapedType>(originalWriteOp->getResultTypes()[0]);
  size_t inMutableIdx =
      successWriteTensorType.getRank() - sucessWriteVectorType.getRank();
  Operation::operand_range writeIndices = originalWriteOp->getIndices();

  for (size_t i = 0; i < inMutableIdx; i++)
    writeVars[i] = writeIndices[i];
}

void ForLoopGenerator::rectifyReadOperationIndice(
    vector::TransferReadOp *originalReadOp, VectorType loopType,
    ArrayRef<Value> inductionVars, SmallVectorImpl<Value> &readVars) {
  ShapedType readTensorType =
      cast<ShapedType>(originalReadOp->getSource().getType());
  // currently only broadcast (fuse as transfer_read) will move into more inner
  // loop
  if (readTensorType.getRank() - 1 >=
      (int64_t)getFusionStrategy().getOpAnchorPos()[*originalReadOp])
    return;

  int64_t itrIdx = loopType.getRank() - 1;
  int64_t readIdx = readTensorType.getRank() - 1;
  while (itrIdx >= 0 and readIdx >= 0) {
    if (readTensorType.getShape()[readIdx] == loopType.getShape()[itrIdx]) {
      readVars[readIdx] = inductionVars[itrIdx];
      readIdx--;
    }
    itrIdx--;
  }
}

/// generate transpose for loop
scf::ForOp ForLoopGenerator::generateShapeCastForLoop(const size_t grpIdx) {

  ShapeCastCanonicalizer &scCanonicalizer =
      getShapeCastCanonicalizers()[grpIdx];
  vector::ShapeCastOp &scOp = scCanonicalizer.getCandidateOps()[0];

  VectorType sourceType = scOp.getSourceVectorType();
  VectorType destType = scOp.getResultVectorType();
  DenseMap<Operation *, size_t> &opIndexMap =
      getFusionStrategy().getOpGroupIndexMap();

  OpBuilder b(scOp);
  SmallVector<Value> iterArgs;
  SmallVector<vector::TransferWriteOp> successorWriteOps;
  for (Operation *x : scOp->getUsers())
    if (isa<vector::TransferWriteOp>(x) and opIndexMap.contains(x) and
        opIndexMap[x] == opIndexMap[scOp])
      successorWriteOps.emplace_back(cast<vector::TransferWriteOp>(x));

  for (auto successorWriteOp : successorWriteOps)
    iterArgs.emplace_back(successorWriteOp->getOperands()[1]);

  SmallVector<Value> inductionVars;
  IRRewriter rewriter(func);
  const size_t groupStep = getFusionStrategy().getGroupMaxSteps()[grpIdx];

  bool isSourceMultiple =
      sourceType.getShape()[sourceType.getRank() - 1] % groupStep == 0;
  bool isDestMultiple =
      destType.getShape()[destType.getRank() - 1] % groupStep == 0;

  scf::ForOp forOp;
  bool canVectorizedLoadStore = isDestMultiple and isSourceMultiple and
                                scCanonicalizer.isReadWriteOnLastDim();
  if (canVectorizedLoadStore) {
    forOp = generateShapeCastReadWriteLoop(
        b, grpIdx, 0, groupStep, scOp.getLoc(), inductionVars, iterArgs);
  } else {
    // scalar data movement
    forOp = generateShapeCastReadWriteLoop(b, grpIdx, 0, 1, scOp.getLoc(),
                                           inductionVars, iterArgs);
  }
  for (auto [idx, successorWriteOp] : enumerate(successorWriteOps))
    rewriter.replaceOp(successorWriteOp, forOp->getResults()[idx]);

  rewriter.eraseOp(scOp);
  clearCurrentOperationGroup(grpIdx);
  return forOp;
}

/// mark which operation need to set correct for loop var idx
/// due to sometimes we need to chage for loop order like reduce operation.
void ForLoopGenerator::getCurrentGroupIndiceLoopMap(
    DenseMap<Operation *, DenseMap<size_t, size_t>> &indiceLoopMap,
    const size_t groupId, Operation *op,
    const DenseMap<size_t, size_t> &setIdxMap) {
  if (setIdxMap.empty()) {
    DenseMap<size_t, size_t> forIdxMap;
    VectorType groupVector =
        getFusionStrategy().getGroupBiggestRankVectorType()[groupId];
    for (size_t i = 0; (int64_t)i < groupVector.getRank(); i++) {
      forIdxMap[i] = i;
    }
    indiceLoopMap[op] = forIdxMap;
    return;
  }
  indiceLoopMap[op] = setIdxMap;
}

void ForLoopGenerator::clearCurrentOperationGroup(size_t grpIdx) {
  std::queue<Operation *>().swap(getFusionStrategy().getOpGroups()[grpIdx]);
};

scf::ForOp ForLoopGenerator::generateTransposeForLoop(const size_t grpIdx) {

  // transpose rank must bigger than 2
  TransposeCanonicalizer &tpCanonicalizer =
      getTransposeCanonicalizers()[grpIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  IRRewriter rewriter(func);

  VectorType vtType = tpOp.getResultVectorType();
  size_t rank = vtType.getRank();
  if (rank < 2) {
    llvm::llvm_unreachable_internal(
        "Wrong transpose operation appear. It's rank must bigger than 2.");
    return nullptr;
  }

  // permutation contains last dim can use optimizing algorithm
  ArrayRef<int64_t> permutation = tpOp.getPermutation();
  DenseSet<int64_t> permuteSet(permutation.begin(), permutation.end());
  bool isTwoDTranspose = tpCanonicalizer.isTwoDTranspose();

  Operation *successorWriteOp =
      getNextTargetOperationInCurrentGroup<vector::TransferWriteOp>(tpOp,
                                                                    grpIdx);

  DenseMap<Value, int> operandIdxMap;
  DenseMap<Value, Value> originalOperandMap, operandOriginalMap, resultIdxMap,
      forResultOrignalResultMap;
  SmallVector<Value> iterArgs;
  GenerateLoopHelper loopHelper(grpIdx, 0);
  prepareForLoopArgs(grpIdx, loopHelper);

  OpBuilder b(tpOp);
  int tpStep = TransposeCanonicalizer::TRANSPOSE_KERNEL::KERNEL_16X16;
  // only contains last dim can use fast transpose algorithm
  if ((tpCanonicalizer.getFirstTpIdx() == (rank - 1) or
       tpCanonicalizer.getSecondTpIdx() == (rank - 1)) and
      isTwoDTranspose) {
    scf::ForOp forOp = generateTransposeForLoopWithLastDim(
        b, tpStep, tpOp.getLoc(), successorWriteOp, loopHelper);

    rewriter.replaceOp(successorWriteOp, forOp);
    // clear current group operation
    clearCurrentOperationGroup(grpIdx);
    return forOp;
  }
  DenseMap<size_t, size_t> tpAxisMap;
  size_t itrIdx = 0;
  while (itrIdx < rank) {
    tpAxisMap[itrIdx] = permutation[itrIdx];
    itrIdx++;
  }
  // scalar data movement
  scf::ForOp forOp = generateTransposeScalarDataMovement(b, tpOp.getLoc(),
                                                         tpAxisMap, loopHelper);

  rewriter.replaceOp(successorWriteOp, forOp);
  clearCurrentOperationGroup(grpIdx);
  return forOp;
}

template <class T>
SmallVector<T, 4> &SpecialOperationCanonicalizer<T>::getCandidateOps() {
  return candidateRdOps;
};

void MultiReductionCanonicalizer::initReductionAxis() {
  auto reductionAxisRange = getCandidateOps()[0].getReductionDims();
  reductionAxis.assign(reductionAxisRange.begin(), reductionAxisRange.end());
  llvm::sort(reductionAxis);
}

void MultiReductionCanonicalizer::initParallelAxis() {
  llvm::SmallDenseSet<int64_t, 4> reductionAxisSet(reductionAxis.begin(),
                                                   reductionAxis.end());
  for (int64_t i = 0; i < typeRank; ++i)
    if (!reductionAxisSet.contains(i))
      parallelAxis.push_back(i);

  llvm::sort(parallelAxis);
}

int64_t MultiReductionCanonicalizer::getTypeRank() {
  auto srcRank = sourceType.getRank();
  typeRank = srcRank;
  return srcRank;
}

void MultiReductionCanonicalizer::getReductionAxisAndParallelAxis() {
  initReductionAxis();
  initParallelAxis();
}

bool MultiReductionCanonicalizer::hasLastDimReduction() {
  llvm::SmallDenseSet<int64_t, 4> reductionAxisSet(reductionAxis.begin(),
                                                   reductionAxis.end());
  bool res = false;
  if (reductionAxisSet.contains(typeRank - 1))
    res = true;

  haslastDimReduction = res;
  return res;
}

void MultiReductionCanonicalizer::prepareSpecialOperationInfo() {
  if (getCandidateOps().empty())
    return;

  sourceType = getCandidateOps()[0].getSourceVectorType();
  accType = cast<VectorType>(getCandidateOps()[0].getAcc().getType());
  getTypeRank();
  getReductionAxisAndParallelAxis();
  hasLastDimReduction();

  // whether all the reduction axis is 1
  for (auto axis : reductionAxis) {
    if (sourceType.getShape()[axis] != 1) {
      isEmptyReduction = false;
      break;
    }
  }
};

bool TransposeCanonicalizer::isTransposeOnAllOneDim() {
  vector::TransposeOp tpOp = getCandidateOps()[0];
  ArrayRef<int64_t> permutation = tpOp.getPermutation();
  VectorType tpVectorType = tpOp.getResultVectorType();
  int64_t itrIdx = 0;
  while (itrIdx < tpVectorType.getRank()) {
    if (itrIdx == permutation[itrIdx]) {
      itrIdx++;
      continue;
    }
    if (tpVectorType.getShape()[itrIdx] != 1)
      return false;

    itrIdx++;
  }
  return true;
}

bool TransposeCanonicalizer::isTwoDTranspose() {
  ArrayRef<int64_t> permutation = getCandidateOps()[0].getPermutation();

  size_t rank = permutation.size();
  int diffCount = 0;
  // get the first transpose axis
  size_t itrIdx = 0;
  while (itrIdx < rank) {
    if ((int64_t)itrIdx != permutation[itrIdx])
      diffCount += 1;

    itrIdx += 1;
  }

  itrIdx = 0;
  while (itrIdx < rank) {
    if (permutation[itrIdx] != (int64_t)itrIdx) {
      firstTpIdx = itrIdx;
      break;
    }
    itrIdx++;
  }

  itrIdx = 0;
  // get the second transpose axis
  while (itrIdx < rank) {
    if (permutation[itrIdx] == (int64_t)firstTpIdx) {
      secondTpIdx = itrIdx;
      break;
    }
    itrIdx++;
  }

  const int tpStep = TRANSPOSE_KERNEL::KERNEL_16X16;
  VectorType vtType = getCandidateOps()[0].getResultVectorType();
  // currently we only support shape that is an integer multiple of tpStep
  if (vtType.getShape()[getFirstTpIdx()] % tpStep != 0 or
      vtType.getShape()[getSecondTpIdx()] % tpStep != 0)
    return false;

  return diffCount == 2;
}

bool TransposeCanonicalizer::transposeOnLastDim() {
  ArrayRef<int64_t> permutation = getCandidateOps()[0].getPermutation();
  size_t rank = permutation.size();
  if (permutation[rank - 1] != (int64_t)rank - 1)
    return false;

  VectorType vtType = getCandidateOps()[0].getResultVectorType();
  return vtType.getShape()[rank - 1] % getVectorStep() == 0;
}

bool ShapeCastCanonicalizer::isReadWriteOnLastDim() {
  vector::ShapeCastOp &shapeCastOp = getCandidateOps()[0];
  VectorType sourceType = shapeCastOp.getSourceVectorType();
  VectorType destType = shapeCastOp.getResultVectorType();
  VectorType smallRankType =
      sourceType.getRank() > destType.getRank() ? destType : sourceType;
  VectorType largeRankType =
      sourceType.getRank() < destType.getRank() ? destType : sourceType;
  SmallVector<bool> visitedAxis(largeRankType.getRank(), false);
  // Map the index of the larger rank shape to the index of the smaller rank
  // shape.
  DenseMap<size_t, SmallVector<size_t>> shapeIdxMap;
  for (size_t i = 0; (int64_t)i < smallRankType.getRank(); i++)
    shapeIdxMap[i] = SmallVector<size_t>();

  int64_t itrIdx = 0;
  while (itrIdx < smallRankType.getRank()) {
    int64_t endShape = getFirstTrueIndex(visitedAxis), dimSize = 1;
    assert(endShape < largeRankType.getRank() and endShape >= 0 &&
           "Invalid endShape");
    // skip non corresponding axis
    // e.g.: vector<32x16x1x32xbf16> -> vector<1x512x32xbf16>
    while (largeRankType.getShape()[endShape] >
           smallRankType.getShape()[itrIdx])
      endShape++;

    while (endShape < largeRankType.getRank()) {
      visitedAxis[endShape] = true;
      shapeIdxMap[itrIdx].emplace_back(endShape);
      dimSize *= largeRankType.getShape()[endShape];
      if ((int64_t)dimSize == smallRankType.getShape()[itrIdx])
        break;

      endShape++;
    }
    itrIdx++;
  }
  // check if the last dim is read write
  SmallVector<size_t> lastDims = shapeIdxMap[smallRankType.getRank() - 1];
  DenseSet<size_t> set(lastDims.begin(), lastDims.end());
  return set.contains(largeRankType.getRank() - 1);
}

template <class T>
void addDummyInit(SmallVector<T, 8> &canonicalizer, size_t steps = 1) {
  canonicalizer.emplace_back(T({}, steps));
};

void CanonicalizerVectorOperation::clearSpecialOperationCanonicalizers() {
  getMultiRdCanonicalizers().clear();
  getBroadcastCanonicalizers().clear();
  getTransposeCanonicalizers().clear();
  getShapeCastCanonicalizers().clear();
}

void CanonicalizerVectorOperation::dummyInitSpecialOperation(size_t steps) {
  addDummyInit<MultiReductionCanonicalizer>(getMultiRdCanonicalizers(), steps);
  addDummyInit<BroadcastCanonicalizer>(getBroadcastCanonicalizers(), steps);
  addDummyInit<TransposeCanonicalizer>(getTransposeCanonicalizers(), steps);
  addDummyInit<ShapeCastCanonicalizer>(getShapeCastCanonicalizers(), steps);
}

void CanonicalizerVectorOperation::initSpeicalOperationCanonicalizers() {
  clearSpecialOperationCanonicalizers();
  SmallVector<std::queue<Operation *>, 8> &opGroups =
      getFusionStrategy().getOpGroups();
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    dummyInitSpecialOperation(getFusionStrategy().getGroupMaxSteps()[idx]);
    if (grp.empty())
      continue;

    std::queue<Operation *> tempQ(grp);
    while (!tempQ.empty()) {
      auto op = tempQ.front();
      tempQ.pop();
      TypeSwitch<Operation *>(op)
          .Case<vector::MultiDimReductionOp>([&](vector::MultiDimReductionOp
                                                     multiReductionOp) {
            getMultiRdCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::MultiDimReductionOp>(op));
            getMultiRdCanonicalizers().back().prepareSpecialOperationInfo();
          })
          .Case<vector::BroadcastOp>([&](vector::BroadcastOp broadCastOp) {
            getBroadcastCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::BroadcastOp>(op));
          })
          .Case<vector::TransposeOp>([&](vector::TransposeOp tpOp) {
            getTransposeCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::TransposeOp>(op));
          })
          .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp spOp) {
            getShapeCastCanonicalizers().back().getCandidateOps().emplace_back(
                cast<vector::ShapeCastOp>(op));
          })
          .Default([&](Operation *op) {});
    }
  }
}

template <class T, class U>
void CanonicalizerVectorOperation::processSpecialOperation(
    T &canonicalizers, const std::function<void(const size_t)> &generateFunc) {
  for (auto [groupId, canonicalizer] : llvm::enumerate(canonicalizers)) {
    SmallVector<U, 4> &ops = canonicalizer.getCandidateOps();
    if (!ops.empty())
      // generate MultiReduction for loops
      generateFunc(groupId);
  }
}

void CanonicalizerVectorOperation::canonicalizeSpecialOperation() {
  OpBuilder::InsertionGuard guard(rewriter);

  initSpeicalOperationCanonicalizers();
  // traverse all groups
  llvm::SmallVector<MultiReductionCanonicalizer, 8> &multiRdCanonicalizers =
      getMultiRdCanonicalizers();
  processSpecialOperation<SmallVector<MultiReductionCanonicalizer, 8>,
                          vector::MultiDimReductionOp>(
      multiRdCanonicalizers, [this](const size_t grpIdx) {
        (void)generateMultiReductionForLoop(grpIdx);
      });
  // generate loop for transpose operation
  SmallVector<TransposeCanonicalizer, 8> &transposeCanonicalizers =
      getTransposeCanonicalizers();
  processSpecialOperation<SmallVector<TransposeCanonicalizer, 8>,
                          vector::TransposeOp>(
      transposeCanonicalizers,
      [this](const size_t grpIdx) { (void)generateTransposeForLoop(grpIdx); });
  // generate loop for shapecast opearation
  SmallVector<ShapeCastCanonicalizer, 8> &shapeCastCanonicalizers =
      getShapeCastCanonicalizers();
  processSpecialOperation<SmallVector<ShapeCastCanonicalizer, 8>,
                          vector::ShapeCastOp>(
      shapeCastCanonicalizers,
      [this](const size_t grpIdx) { (void)generateShapeCastForLoop(grpIdx); });
}

void CanonicalizerVectorOperation::run() {
  auto &fusionStrategy = getFusionStrategy();
  if (kind == CanonicalizerKind::OperationsGroup) {
    // 1. Analysis the operation's operands and results
    // We need to analyze which operation's result is needed by other
    // operations, and we need to pass these results correctly. Mapping the
    // operation result value with the forloop yeild result value. We can
    // replace the operation operand as: map(operand, forloop yield result) ->
    // operand = loop yield result We put all the operation result into this
    // map.

    // 1.a. Find results which should be generated by current group for
    // using as operands to other operations?

    // Traverse all operations. If the operand of operations in other groups
    // or outside the group is the result of the operation in current group,
    // then the current operation needs to generate a result. We use `setvector`
    // to save the results that need to be generated by the current group.

    //  1.b. What operands are needed to find in the current group, and where
    //  can they be obtained ?

    //  Thanks to 1.a, we get the result generated by the operations of
    //  each group, and this result will use `scf.yield` to generate a
    //  new result. Since the scope of the parent block of mlir is covered
    //  the current operation, the current operation does not need to pass
    //  these `for loop result` to the `iterArgs` of the required `for loop`.
    //  It only needs to replace the operand of the current operation with the
    //  corresponding `for loop yield result`.

    // However, for some operations that are not DPS, we need to canonicalize
    // them. Canonicalization means that the operand of this operation is a
    // vector but we can't get this vector due to it locates in another block
    // which has a different scope. Therefore, it is necessary to write the
    // vector results into a temporary tensor to save it. Then the vector
    // needs to be read from the tensor before the current operation operate
    // on it. Therefore,  `empty tensor`, `transfer_write` and `transfer_read`
    // need to be inserted at target place.
    if (enableDebugPrinter) {
      printGroupOps(getFusionStrategy().getOpGroups());
      llvm::outs() << "___________ before analysis ________________"
                   << "\n";
    }
    analysisGroupOperaion();
    if (enableDebugPrinter) {
      llvm::outs() << "___________ after analysis ________________"
                   << "\n";
      printGroupOps(getFusionStrategy().getOpGroups());
    }

    // Speical Operation Canonicalization
    canonicalizeSpecialOperation();

    // 2.Generate vectorized IR for each operation group
    for (size_t idx = 0; idx < fusionStrategy.getOpGroups().size(); ++idx)
      generateGroupOpVectorizedIR(idx);

    // 3. Some IR cleanup work
    DominanceInfo domInfo;
    eliminateCommonSubExpressions(rewriter, domInfo, func);
  } else {
    // TODO: need to add directly canonicalize operations logic
    // generateGroupOpVectorizedIR(idx, grp, fusionStrategy.opGroupIndexMap);
  }
}

// Filter out the operations that can be vectorized. We are only interested in
// operations that do not contain any for loops(innermost IR).
[[nodiscard]] bool filterOperation(Operation *op) {
  if (!is_innermost_operation(op)) {
    LDBG("Operation is not innermost" << *op << "\n");
    return false;
  }

  // We are only interested about the operation in vector dialect
  if (failed(getOperationVectorType(op))) {
    LDBG("Operation is not in vector dialect" << *op << "\n");
    return false;
  }

  // We don't need to vectorize the constant operation
  if (isa<arith::ConstantOp>(op)) {
    LDBG("Operation is constantOp" << *op << "\n");
    return false;
  }

  if (isReadOrWriteOperation(op) and !isReadWriteOnLastDim(op)) {
    LDBG("Operation is not last dim read/write" << *op << "\n");
    return false;
  }

  return true;
}

///
void ForLoopGenerator::setOperationCorrectOperand(
    Operation *op, const DenseMap<Operation *, AffineMap> &opPermuationMap,
    GenerateLoopHelper &loopHelperParam) {
  for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
    if (not loopHelperParam.originalOperandLoopArgsMap.contains(opd))
      continue;

    Value loopArg = loopHelperParam.originalOperandLoopArgsMap[opd];
    if (not loopHelperParam.currentLoopStateIdxMap.contains(loopArg))
      continue;

    op->setOperand(
        idx,
        loopHelperParam
            .loopIterArgs[loopHelperParam.currentLoopStateIdxMap.at(loopArg)]);
  }
  int offset = isa<vector::TransferWriteOp>(op) ? 2 : 1;
  if (dyn_cast<vector::TransferWriteOp>(op) ||
      dyn_cast<vector::TransferReadOp>(op)) {
    assert(opPermuationMap.contains(op));
    auto permutationMap = opPermuationMap.at(op);

    auto dimExpr = permutationMap.getResults();
    for (auto [idx, x] : llvm::enumerate(dimExpr)) {

      if (not isa<AffineDimExpr, AffineConstantExpr>(x))
        llvm::llvm_unreachable_internal(
            "Permuatation map must contains dim expr.");

      int64_t dim = 0;
      if (auto d = dyn_cast<AffineDimExpr>(x)) {
        dim = d.getPosition();
      } else if (auto d = dyn_cast<AffineConstantExpr>(x)) {
        dim = d.getValue();
      }

      ShapedType tensorType =
          cast<ShapedType>(op->getOperandTypes()[offset - 1]);
      int64_t varIdx = dim;
      if (tensorType.getRank() >
          (int64_t)loopHelperParam.inductionVars.size()) {
        int64_t tensorOffset =
            tensorType.getRank() - loopHelperParam.inductionVars.size();
        if (dim < tensorOffset)
          continue;

        varIdx = dim - tensorOffset;
      }
      if (loopHelperParam.indiceLoopMap.contains(op))
        op->setOperand(
            dim + offset,
            loopHelperParam
                .inductionVars[loopHelperParam.indiceLoopMap[op][varIdx]]);
      else
        op->setOperand(dim + offset, loopHelperParam.inductionVars[varIdx]);
    }
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      size_t grpIdx = getFusionStrategy().getOpGroupIndexMap()[op];
      VectorType loopType =
          getFusionStrategy().getGroupBiggestRankVectorType()[grpIdx];
      SmallVector<Value> readIndices(readOp.getIndices().begin(),
                                     readOp.getIndices().end());
      rectifyReadOperationIndice(&readOp, loopType,
                                 loopHelperParam.inductionVars, readIndices);
      readOp.getIndicesMutable().assign(readIndices);
    }
  }
}

scf::ForOp ForLoopGenerator::constructNestedForOp(
    const size_t groupIdx, OpBuilder &b, const Location &loc,
    ArrayRef<int64_t> dims, GenerateLoopHelper &loopHelper) {
  const int loop_step = getFusionStrategy().getGroupMaxSteps()[groupIdx];
  // loop initialization variable
  auto zero = makeIndexArithConstantOp(b, loc, 0);
  auto forSteps = makeIndexArithConstantOp(
      b, loc, loopHelper.anchorIdx == dims.size() - 1 ? loop_step : 1);
  auto numIter = makeIndexArithConstantOp(b, loc, dims[loopHelper.anchorIdx]);

  // Create a loop and move vectorized operation into loops.
  auto forOp = b.create<scf::ForOp>(
      loc, zero, numIter, forSteps, loopHelper.loopIterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        loopHelper.inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (loopHelper.anchorIdx == dims.size() - 1) {
          std::queue<Operation *> &opQueue =
              getFusionStrategy().getOpGroups()[groupIdx];
          loopHelper.loopIterArgs = loopState;
          // 1. get operations in current anchor position
          std::queue<Operation *> movingOperation;
          getOperationInCurrentAnchor(loopHelper.anchorIdx, opQueue,
                                      movingOperation);

          // 2. rewrite operation as vectorize IR
          rewriteOperationAsVectorize(b, groupIdx, &movingOperation);

          // 3. move opeartions to current for block
          moveOperationsToCurrentForBody(b, movingOperation, loopHelper);

          getResultInCurrentOps(loopHelper.anchorIdx, groupIdx, movingOperation,
                                loopHelper.nextAnchorResults,
                                loopHelper.nextAnchorResultsIdxMap,
                                loopHelper.nextAnchorResultOrignalResultMap);
          maybeYieldValue(b, loc, loopHelper.nextAnchorResults);
        } else {
          // outter loop

          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          DenseMap<Value, Value> currentOriginalOperandMap =
              loopHelper.originalOperandLoopArgsMap;
          DenseMap<Value, Value> currentOperandOriginalMap =
              loopHelper.loopArgsOriginalOperandMap;
          DenseMap<Value, int> currentArgsIdxMap =
              loopHelper.currentLoopStateIdxMap;

          std::queue<Operation *> movedQueue;
          std::queue<Operation *> &opQueue =
              getFusionStrategy().getOpGroups()[groupIdx];
          SmallVector<Value> tmpArgs(loopState);
          loopHelper.updateDataBeforePreOpMove(tmpArgs, opQueue, movedQueue);
          movePreOpToCurrentAnchor(b, nextAnchorArgsIdxMap, nextAnchorArgs,
                                   loopHelper);
          loopHelper.updateDataAfterPreOpMove(nextAnchorArgsIdxMap,
                                              nextAnchorArgs);
          loopHelper.anchorIdx += 1;
          auto nxtFor =
              constructNestedForOp(groupIdx, b, loc, dims, loopHelper);
          loopHelper.anchorIdx -= 1;
          SmallVector<Value, 4> currentArgs(loopState);

          loopHelper.updateCurrentArgsStatus(currentArgsIdxMap, currentArgs,
                                             currentOriginalOperandMap,
                                             currentOperandOriginalMap);

          loopHelper.updateDataBeforePostOpMove(
              tmpArgs, currentArgsIdxMap, currentOriginalOperandMap,
              currentOperandOriginalMap, nxtFor->getResults(), b.getBlock(),
              movedQueue, loopHelper.anchorIdx);
          movePostOpToCurrentAnchor(b, loopHelper);

          generateLoopResults(b, loc, loopHelper, nextAnchorArgsIdxMap);

          maybeYieldValue(b, loc, loopHelper.nextAnchorResults);
        }
      });
  return forOp;
}

/// default op1 is previous operation
bool VectorFusionStrategy::isCompatibleVectorType(Operation *op1,
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

///  which axis do the shape cast in source shape a
void shapeCastSourceAxis(const ArrayRef<int64_t> &a, const ArrayRef<int64_t> &b,
                         SmallVector<int64_t> &res) {
  unsigned rankA = a.size();
  unsigned rankB = b.size();
  assert(rankA < rankB && "May be invalid shape cast operation.");

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
      assert(false && " Invalid shape cast operation.");
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

  assert(i == rankA && j == rankB && "Invalid shapecast operation.");
}

bool isScalar(Type type) {
  assert(type && "Not a valid type");
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
  assert(output_rank >= input_rank &&
         "Incorrect input or output shape for broadcast op.");
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

/// whether two operation has data dependency
/// op1 default is previous operation, op2 default is current operation
bool hasDataDependency(Operation *op1, Operation *op2) {
  if (!isSpecialOp(op1) and !isSpecialOp(op2))
    return false;

  // TODO: Remove this condition to support special operation fusion in the
  // future
  if (disableSpecialOp)
    return true;

  if (isReadOrWriteOperation(op1) or isReadOrWriteOperation(op2)) {
    // if op1 is read the value and pass it to op2, it is not data dependency
    if (isOperationsHasDefUseRelation<vector::TransferReadOp>(op1, op2))
      return false;
  }

  // broadcast only fuse with post-op
  if (isa<vector::BroadcastOp>(op2))
    return true;

  if (isa<vector::BroadcastOp>(op1) and disableBroadcastOp)
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

bool VectorFusionStrategy::isNeedNewGroup(Operation *op) {
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

void VectorFusionStrategy::updateGroupBigestVectorType(VectorType vectorType) {
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

void VectorFusionStrategy::addOperationToGroup(Operation *op) {
  assert(op);
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
void VectorFusionStrategy::classifyOperations() {
  // dummpy
  if (opGroups.empty())
    opGroups.emplace_back(std::queue<Operation *>());

  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (filterOperation(op)) {
      addOperationToGroup(op);
      return WalkResult::advance();
    }
    if (isNotNeedToProcessOp(op) and !opGroups.back().empty())
      opGroups.emplace_back(std::queue<Operation *>());

    return WalkResult::advance();
  });
}

Value setOutGroupOperationOperandResult(Operation *op,
                                        const VectorType &newOperandType) {
  auto ret =
      TypeSwitch<Operation *, Value>(op)
          .Case<arith::ConstantOp>([&](arith::ConstantOp constantOp) {
            IRRewriter rewriter(op);
            rewriter.setInsertionPointAfter(op);
            Type resultElementType = newOperandType.getElementType();
            auto value = constantOp.getValue();
            Attribute initValueAttr;

            if (isa<ElementsAttr>(value)) {
              auto valueType = mlir::dyn_cast<ElementsAttr>(value);
              if (valueType.isSplat()) {
                if (isa<FloatType>(valueType.getElementType()))
                  initValueAttr = FloatAttr::get(
                      resultElementType,
                      valueType.getSplatValue<APFloat>().convertToDouble());
                else
                  initValueAttr = IntegerAttr::get(
                      resultElementType,
                      valueType.getSplatValue<APInt>().getSExtValue());
              } else {
                // write original vector into tensor
                // then we transfer_read from the tensor
                assert(0 && "Not support non-splat constant value.");
              }
            } else if (isa<FloatType>(resultElementType)) {
              initValueAttr = FloatAttr::get(
                  resultElementType, cast<FloatAttr>(value).getValueAsDouble());
            } else {
              initValueAttr = IntegerAttr::get(
                  resultElementType, cast<IntegerAttr>(value).getInt());
            }

            auto cntOp = rewriter.create<arith::ConstantOp>(
                rewriter.getUnknownLoc(),
                DenseElementsAttr::get(newOperandType, {initValueAttr}));
            return cntOp->getResults()[0];
          })
          .Default([&](Operation *op) { return Value(); });
  return ret;
}

void setOperationOperandResult(Operation *op, const VectorType &newOperandType,
                               const DenseMap<Operation *, size_t> &opMap) {
  for (auto [idx, x] : llvm::enumerate(op->getOperands())) {
    if (dyn_cast<VectorType>(x.getType())) {
      if (!opMap.contains(x.getDefiningOp())) {
        auto result = setOutGroupOperationOperandResult(x.getDefiningOp(),
                                                        newOperandType);
        op->setOperand(idx, result);
      } else {
        x.setType(newOperandType);
      }
    }
  }
  for (auto x : op->getResults())
    if (dyn_cast<VectorType>(x.getType()))
      x.setType(newOperandType);
};

void ForLoopGenerator::createNewConstantOp(
    Operation *srcOp, vector::TransferWriteOp *transferWriteOp,
    size_t groupSteps) {
  DenseMap<Operation *, AffineMap> &opPermuationMap = getOpPermuationMap();

  IRRewriter srcWriter(srcOp);
  VectorType newOperandType =
      getVectorzedType(cast<Operation *>(srcOp), groupSteps);
  auto srcConstantOp = dyn_cast<arith::ConstantOp>(srcOp);
  Operation *newConstantOp;
  if (isa<DenseElementsAttr>(srcConstantOp.getValue())) {
    auto valueType = dyn_cast<DenseElementsAttr>(srcConstantOp.getValue());
    if (valueType.isSplat()) {
      FailureOr<Value> res = createArithSplatConstantOp(
          srcWriter, srcOp->getLoc(), valueType, newOperandType);
      if (failed(res)) {
        llvm::llvm_unreachable_internal("Wrong to create constant op.");
      }
      newConstantOp = res.value().getDefiningOp();
    } else {
      // TODO: need to test not splat value
      llvm::llvm_unreachable_internal(
          "Can't support not splat constant value.");
    }

    newConstantOp->getResult(0).setType(newOperandType);
    transferWriteOp->setOperand(0, newConstantOp->getResult(0));
    opPermuationMap.insert(
        {*transferWriteOp, transferWriteOp->getPermutationMap()});
    setOpVectorizationPermutationMap(
        *transferWriteOp, srcWriter,
        cast<ShapedType>(transferWriteOp->getResults()[0].getType()),
        transferWriteOp->getPermutationMap());
    return;
  }
  llvm::llvm_unreachable_internal(
      "Can't support not DenseElementsAttr constant.");
}

/// Rewrite the operations in the group to vectorized form.
void ForLoopGenerator::rewriteOperationAsVectorize(
    OpBuilder &rewriter, size_t groupId, const std::queue<Operation *> *queue) {
  const std::queue<Operation *> groupOps =
      !queue ? getFusionStrategy().getOpGroups()[groupId] : *queue;

  const DenseMap<Operation *, size_t> &opMap =
      getFusionStrategy().getOpGroupIndexMap();
  DenseMap<Operation *, AffineMap> &opPermuationMap = getOpPermuationMap();
  std::queue<Operation *> transformQueue(groupOps);
  size_t groupSteps = getFusionStrategy().getGroupMaxSteps()[groupId];

  while (!transformQueue.empty()) {
    Operation *op = transformQueue.front();
    transformQueue.pop();
    VectorType newOperandType = getVectorzedType(op, groupSteps);
    auto lowerResult =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<vector::TransferWriteOp>(
                [&](vector::TransferWriteOp transferWriteOp) {
                  IRRewriter rewriter(transferWriteOp);

                  Operation *srcOp =
                      transferWriteOp->getOperand(0).getDefiningOp();
                  if (isa<arith::ConstantOp>(srcOp)) {
                    createNewConstantOp(srcOp, &transferWriteOp, groupSteps);
                  } else {
                    opPermuationMap.insert(
                        {transferWriteOp, transferWriteOp.getPermutationMap()});
                    transferWriteOp->getOperand(0).setType(newOperandType);

                    setOpVectorizationPermutationMap(
                        transferWriteOp, rewriter,
                        cast<ShapedType>(
                            transferWriteOp->getResult(0).getType()),
                        transferWriteOp.getPermutationMap());
                  }

                  return success();
                })
            .Case<vector::TransferReadOp>(
                [&](vector::TransferReadOp transferReadOp) {
                  opPermuationMap.insert(
                      {transferReadOp, transferReadOp.getPermutationMap()});
                  transferReadOp->getResult(0).setType(newOperandType);
                  setOpVectorizationPermutationMap(
                      transferReadOp, rewriter,
                      cast<ShapedType>(transferReadOp.getSource().getType()),
                      transferReadOp.getPermutationMap());

                  return success();
                })
            .Case<vector::MultiDimReductionOp>(
                [&](vector::MultiDimReductionOp multiReductionOp) {
                  multiReductionOp.dump();
                  llvm::llvm_unreachable_internal(
                      "It should not appear this operation.");
                  return failure();
                })
            .Case<ARITH_CAST_OPERATIONS>([&](Operation *extfOp) {
              extfOp->getResult(0).setType(newOperandType);
              return success();
            })
            .Default([&](Operation *op) {
              if (isSpecialOp(op)) {
                op->dump();
                llvm::llvm_unreachable_internal(
                    "It should not appear this operation.");
                return failure();
              }
              setOperationOperandResult(op, newOperandType, opMap);
              return success();
            });
    if (failed(lowerResult)) {
      LDBG("Failed to rewrite operation: " << *op << "\n");
      assert(false && "Failed to rewrite operation");
    }
  }
}

mlir::FailureOr<Value> getOperationOperateTensor(Operation *op) {
  return TypeSwitch<Operation *, mlir::FailureOr<Value>>(op)
      .Case<vector::TransferWriteOp>(
          [&](vector::TransferWriteOp transferWriteOp) {
            // find original tensor.empty operation
            auto writeTensor = transferWriteOp->getOperand(1);
            writeTensor =
                findOriginalTensor(writeTensor, transferWriteOp->getBlock());
            return writeTensor;
          })
      .Case<vector::TransferReadOp>([&](vector::TransferReadOp transferReadOp) {
        return transferReadOp->getOperand(0);
      })
      .Default([&](Operation *op) {
        LDBG("Try to get not DPS operation inits: " << *op << "\n");
        return failure();
      });
}

void CanonicalizerCommonUsedData::removeOpInCurrentGroups(
    size_t grpIdx, Operation *op, Operation *replacedOp) {
  std::queue<Operation *> tmpOpQueue(getFusionStrategy().getOpGroups()[grpIdx]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    if (curOp != op) {
      newOpQueue.push(curOp);
      continue;
    }
    getFusionStrategy().getOpGroupIndexMap().erase(curOp);
    getFusionStrategy().getOpAnchorPos().erase(curOp);
  }
  getFusionStrategy().getOpGroups()[grpIdx] = newOpQueue;

  // erase and replace the operation
  SmallVector<Operation *> usesOp(op->getUsers().begin(), op->getUsers().end());
  IRRewriter rewriter(op);
  rewriter.replaceOp(op, replacedOp);
  // update removed operation related operation anchor position
  getFusionStrategy().getOpAnchorPos()[replacedOp] =
      getOperationMaxVectorType(replacedOp)->getRank() - 1;
  for (Operation *x : usesOp)
    getFusionStrategy().getOpAnchorPos()[x] =
        getOperationMaxVectorType(x)->getRank() - 1;

  updateOpGroupInfo(grpIdx);
}

void CanonicalizerCommonUsedData::updateOpGroupInfo(size_t grpIdx) {
  std::queue<Operation *> tmpOpQueue(getFusionStrategy().getOpGroups()[grpIdx]);
  // dummy init
  VectorType currentMaxRankType =
      getOperationMaxVectorType(tmpOpQueue.front()).value();
  getFusionStrategy().getGroupBiggestRankVectorType()[grpIdx] =
      currentMaxRankType;

  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    VectorType type = getOperationMaxVectorType(curOp).value();
    if (type.getRank() > currentMaxRankType.getRank())
      getFusionStrategy().getGroupBiggestRankVectorType()[grpIdx] = type;
  }
}

void CanonicalizerCommonUsedData::updateOpOperandResultInGroups(
    size_t opGid, Operation *op, const Value &init, const Value &result) {
  std::queue<Operation *> tmpOpQueue(getFusionStrategy().getOpGroups()[opGid]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();

    if (curOp != op) {
      newOpQueue.push(curOp);
      continue;
    }

    if (!failed(getOperationVectorType(init.getDefiningOp()))) {
      newOpQueue.push(init.getDefiningOp());
      getFusionStrategy().getOpGroupIndexMap()[init.getDefiningOp()] = opGid;
      getFusionStrategy().getOpAnchorPos()[init.getDefiningOp()] =
          getFusionStrategy().getOpAnchorPos()[op];
    }
    newOpQueue.push(op);

    if (result && !failed(getOperationVectorType(result.getDefiningOp()))) {
      newOpQueue.push(result.getDefiningOp());
      getFusionStrategy().getOpGroupIndexMap()[result.getDefiningOp()] = opGid;
      getFusionStrategy().getOpAnchorPos()[result.getDefiningOp()] =
          getFusionStrategy().getOpGroupIndexMap()[op];
    }
  }
  getFusionStrategy().getOpGroups()[opGid] = newOpQueue;
}

void VectorFusionStrategy::run() { classifyOperations(); }

void CanonicalizerCommonUsedData::generateEmptyTensorAndWrite(
    Operation *sourceOp,
    DenseMap<Operation *, std::pair<Value, Value>> &srcOpCanoniclizedMap,
    size_t anchorPos, ReturnTypeKind retKind,
    DenseMap<Operation *, size_t> &visitedOperation) {
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getFusionStrategy().getOpGroupIndexMap();
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs = getGroupOpInitArgs();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOpResults();
  size_t sourceOpGid = opGroupIndexMap[sourceOp];

  auto [tsr, writeOpresult] =
      canonicalizeSourceOperation(sourceOp, visitedOperation);
  auto writeOp = writeOpresult.getDefiningOp<vector::TransferWriteOp>();
  srcOpCanoniclizedMap.insert({sourceOp, {tsr, writeOpresult}});
  updateOpOperandResultInGroups(sourceOpGid, sourceOp, tsr, writeOpresult);
  groupOpInitArgs[sourceOpGid].insert(tsr);
  groupOpResults[sourceOpGid].insert({writeOpresult, {retKind, anchorPos}});
  // write opeartion anchor pos is same with current operation
  getFusionStrategy().getOpAnchorPos()[writeOp] =
      writeOp.getVectorType().getRank() - 1;
  getOpPermuationMap()[writeOp] = writeOp.getPermutationMap();
}

template <class Target>
Operation *CanonicalizerCommonUsedData::getNextTargetOperationInCurrentGroup(
    Operation *curOp, const size_t grpIdx) {
  std::queue<Operation *> tmpOpQueue(getFusionStrategy().getOpGroups()[grpIdx]);
  if (isa<Target>(curOp))
    return curOp;

  while (!tmpOpQueue.empty()) {
    auto frontOp = tmpOpQueue.front();
    if (isa<Target>(frontOp)) {
      for (auto x : frontOp->getOperands())
        if (x.getDefiningOp() == curOp)
          return frontOp;
    }
    tmpOpQueue.pop();
  }
  return nullptr;
}

void VectorOperationAnalyzer::analysisEmptyGroup() {
  SmallVector<std::queue<Operation *>, 8> &opGroups =
      getFusionStrategy().getOpGroups();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOpResults();
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    if (grp.empty())
      continue;
    if (groupOpResults[idx].empty())
      std::queue<Operation *>().swap(grp);
  }
}

void VectorOperationAnalyzer::analysisGroupMaxSteps() {
  auto &opGroups = getFusionStrategy().getOpGroups();

  for (auto [idx, grp] : llvm::enumerate(opGroups)) {

    uint32_t steps = std::numeric_limits<uint32_t>::max();

    llvm::SmallVector<uint32_t, 8> &grpSteps =
        getFusionStrategy().getGroupMaxSteps();
    while (idx + 1 > grpSteps.size())
      grpSteps.emplace_back(steps);

    std::queue<Operation *> tmpQueue(grp);
    auto calculateOpSteps = [&](Type type) {
      auto opType = dyn_cast<VectorType>(type);
      if (opType)
        steps = std::min(steps, (uint32_t)getDataTypeValidSteps(opType));
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

void VectorOperationAnalyzer::specialOperationRectify(
    DenseMap<Operation *, size_t> &visitedOperation) {
  auto &opGroups = getFusionStrategy().getOpGroups();
  IRRewriter rewriter(func);

  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    std::queue<Operation *> tmpQueue(grp);
    std::queue<Operation *> newQueue;
    while (!tmpQueue.empty()) {
      auto op = tmpQueue.front();
      tmpQueue.pop();
      //  remain transfer read operation to do the broadcast fusion
      if (isa<vector::BroadcastOp>(op) and not disableBroadcastOp) {
        auto srcOp = op->getOperand(0).getDefiningOp();
        assert(isa<vector::TransferReadOp>(srcOp));
        // only have write operation, otherwise the group size will bigger
        // than 1. Because the last operation is always a write operation in
        // each group
        getFusionStrategy().getOpAnchorPos()[srcOp] =
            getFusionStrategy().getOpAnchorPos()[op];

        rewriter.replaceOp(op, srcOp);
        continue;
      }
      // anchor of multidim reduction rectify
      if (isa<vector::MultiDimReductionOp>(op)) {
        auto accSourceOp = op->getOperand(1).getDefiningOp();
        getFusionStrategy().getOpAnchorPos()[accSourceOp] =
            getOperationVectorType(accSourceOp)->getRank() - 1;
      }
      newQueue.push(op);
    }
    getFusionStrategy().getOpGroups()[idx] = newQueue;
  }
}

void VectorOperationAnalyzer::updateReturnResultKind(Operation *sourceOp,
                                                     size_t sourceOpGid,
                                                     ReturnTypeKind rtKind) {
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOpResults();
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getFusionStrategy().getOpAnchorPos();

  Value sourceResult = sourceOp->getResults()[0];
  if (srcOpCanoniclizedMap.contains(sourceOp))
    sourceResult = srcOpCanoniclizedMap[sourceOp].second;

  size_t srcOpAnchor = groupOpResults[sourceOpGid][sourceResult].second;
  ReturnTypeKind prevRtKind = groupOpResults[sourceOpGid][sourceResult].first;
  srcOpAnchor = std::min(srcOpAnchor, OpAnchorPos[sourceOp]);
  if (prevRtKind != rtKind) {
    groupOpResults[sourceOpGid][sourceResult] =
        std::make_pair(ReturnTypeKind::RT_Both, srcOpAnchor);
    return;
  }
  if (rtKind == ReturnTypeKind::RT_InGroup)
    groupOpResults[sourceOpGid][sourceResult] =
        std::make_pair(rtKind, srcOpAnchor);
}

void VectorOperationAnalyzer::replaceConstantOpAsNewOp(Operation *op,
                                                       Operation *sourceOp,
                                                       size_t operandIdx) {
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getFusionStrategy().getOpGroupIndexMap();
  if (!opGroupIndexMap.contains(op)) {
    return;
  }
  // TODO: add more operation to this case, write a constant value need
  // to do this
  if (isa<vector::TransferWriteOp>(op) and operandIdx == 0)
    return;

  if (isa<vector::MultiDimReductionOp>(op)) {
    if (operandIdx == 1) {
      // accumulate value, just empty tensor is okay
      auto resultTensor = getOperationResultTensor(sourceOp, visitedOperation);
      auto opInit = canonicalizeCurrentOperation(op, resultTensor, operandIdx);
      updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
      return;
    }
    // source operation is the value
    llvm::llvm_unreachable_internal(
        "Need to add reduce constant operation optimization.");
  }

  auto constantOp = cast<arith::ConstantOp>(sourceOp);
  IRRewriter rewriter(constantOp);
  size_t groupSteps =
      getFusionStrategy().getGroupMaxSteps()[opGroupIndexMap[op]];

  if (isa<DenseElementsAttr>(constantOp.getValue())) {
    VectorType newOperandType = getVectorzedType(op, groupSteps);
    auto valueType = cast<DenseElementsAttr>(constantOp.getValue());
    if (valueType.isSplat()) {
      FailureOr<Value> res = createArithSplatConstantOp(
          rewriter, constantOp->getLoc(), valueType, newOperandType);
      if (failed(res))
        llvm::llvm_unreachable_internal("Wrong to create constant op.");

      op->setOperand(operandIdx, res.value());
      // transfer read operation just use the constant value to do
      // calculation, don't need to read.
      if (isa<vector::TransferReadOp>(op) and operandIdx == 0)
        removeOpInCurrentGroups(opGroupIndexMap[op], op,
                                op->getOperand(0).getDefiningOp());
      return;
    }
    llvm::llvm_unreachable_internal("Can't support not splat constant value.");
  }
}

void VectorOperationAnalyzer::makeSourceOpWriteResultToTensor(
    Operation *sourceOp, size_t sourceOpGid, ReturnTypeKind rtKind) {
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getFusionStrategy().getOpAnchorPos();
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs = getGroupOpInitArgs();

  if (!srcOpCanoniclizedMap.contains(sourceOp)) {
    // get write operation
    if (Operation *writeOp =
            getNextTargetOperationInCurrentGroup<vector::TransferWriteOp>(
                sourceOp, sourceOpGid)) {
      auto writeOpresult = writeOp->getResults()[0];
      auto writeTensor = writeOp->getOperands()[1];
      // find original tensor.empty operation
      writeTensor = findOriginalTensor(writeTensor, sourceOp->getBlock());
      srcOpCanoniclizedMap.insert({sourceOp, {writeTensor, writeOpresult}});
      groupOpInitArgs[sourceOpGid].insert(writeTensor);
      updateReturnResultKind(writeOp, sourceOpGid, rtKind);
      return;
    }
    generateEmptyTensorAndWrite(sourceOp, srcOpCanoniclizedMap,
                                OpAnchorPos[sourceOp], rtKind,
                                visitedOperation);
    return;
  }
  // udpate result return type
  updateReturnResultKind(srcOpCanoniclizedMap[sourceOp].second.getDefiningOp(),
                         sourceOpGid, rtKind);
}

void VectorOperationAnalyzer::groupOperationNeedReturnResult(
    size_t sourceOpGid, Operation *sourceOp, Operation *op, size_t operandIdx,
    bool inSameGroupNeedReturn) {
  ReturnTypeKind rtKind = inSameGroupNeedReturn ? ReturnTypeKind::RT_InGroup
                                                : ReturnTypeKind::RT_OutGroup;
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs = getGroupOpInitArgs();

  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getFusionStrategy().getOpGroupIndexMap();
  // update init iterargs
  auto dstRet = getOperationOperateTensor(sourceOp);
  // need to generate tensor.emtpy and vector.transfer_write, write
  // operand to tensor and read operand from the tensor, generate
  // vector.transfer_read
  if (failed(dstRet)) {
    // already generate result tensor, special operation do the
    // transformation by itself
    if (isSpecialOp(sourceOp) and inSameGroupNeedReturn) {
      return;
    }
    makeSourceOpWriteResultToTensor(sourceOp, sourceOpGid, rtKind);
    auto opInit = canonicalizeCurrentOperation(
        op, srcOpCanoniclizedMap[sourceOp].second, operandIdx);
    updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
    return;
  }
  // if source operation is transfer_read, we need to generate a
  // same transfer_read operation like source operation.
  if (isa<vector::TransferReadOp>(sourceOp)) {
    auto transferReadOp = cast<vector::TransferReadOp>(sourceOp);
    auto opInit = canonicalizeCurrentOperation(op, dstRet.value(), operandIdx,
                                               &transferReadOp);
    updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
    return;
  }
  // transfer write operation
  groupOpInitArgs[sourceOpGid].insert(dstRet.value());
  updateReturnResultKind(sourceOp, sourceOpGid, rtKind);
}

void VectorOperationAnalyzer::analysisGroupOperaion() {
  // record the operation which has been moved
  DenseSet<Operation *> movedOperationSet;
  //  record the operation's visited order, inorder to ensure set
  //  correct operand
  size_t opCounter = 0;
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getFusionStrategy().getOpGroupIndexMap();
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getFusionStrategy().getOpAnchorPos();
  IRRewriter rewriter(func);

  analysisGroupMaxSteps();

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    visitedOperation.insert({op, opCounter++});

    for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
      Operation *sourceOp = opd.getDefiningOp();
      if (opGroupIndexMap.contains(sourceOp)) {
        auto sourceOpGid = opGroupIndexMap[sourceOp];
        bool notInSameGroup =
            opGroupIndexMap.contains(op) && sourceOpGid != opGroupIndexMap[op];
        bool outOfGroup = !opGroupIndexMap.contains(op);
        // Different anchor in same group and source operation is in inner
        // loop, we need to get source operation's result
        bool inSameGroupNeedReturn =
            !notInSameGroup and OpAnchorPos[sourceOp] > OpAnchorPos[op];

        if (notInSameGroup or outOfGroup or inSameGroupNeedReturn)
          groupOperationNeedReturnResult(sourceOpGid, sourceOp, op, idx,
                                         inSameGroupNeedReturn);

        continue;
      }
      if (isa_and_nonnull<arith::ConstantOp>(sourceOp))
        replaceConstantOpAsNewOp(op, sourceOp, idx);
    }
  });
  analysisEmptyGroup();
  specialOperationRectify(visitedOperation);
  LDBG("Complete analysis group operation results\n");
}

void ForLoopGenerator::rectifyGroupOperands(size_t currentGroupId,
                                            Value originalResult,
                                            Value forResult) {
  size_t totalGroupSize = getFusionStrategy().getOpGroups().size();
  size_t startGroup = currentGroupId;
  while (startGroup < totalGroupSize) {
    SetVector<Value> &operandVector = getGroupOpInitArgs()[startGroup++];
    if (not operandVector.contains(originalResult))
      continue;
    SetVector<Value> replacedVector;

    for (auto v : operandVector) {
      if (v == originalResult) {
        replacedVector.insert(forResult);
        continue;
      }
      replacedVector.insert(v);
    }
    getGroupOpInitArgs()[startGroup - 1] = replacedVector;
  }
}

mlir::FailureOr<scf::ForOp> ForLoopGenerator::generateVectorizedForLoop(
    const size_t groupId, IRRewriter &rewriter, VectorType vectorType) {
  // prepare for loop iterargs
  GenerateLoopHelper loopHelper(groupId, 0);
  prepareForLoopArgs(groupId, loopHelper);

  ArrayRef<int64_t> shapes = vectorType.getShape();
  // generate for loop
  auto forOp = constructNestedForOp(groupId, rewriter, rewriter.getUnknownLoc(),
                                    shapes, loopHelper);
  replaceOpUsersWithForLoopResult(forOp, groupId, loopHelper.nextAnchorResults,
                                  loopHelper.nextAnchorResultsIdxMap,
                                  loopHelper.nextAnchorResultOrignalResultMap);

  return forOp;
}

bool CanonicalizerCommonUsedData::isGroupHasSpecialOperation(
    const size_t grpIdx) {
  auto &rdCanonicalizer = getMultiRdCanonicalizers()[grpIdx];
  auto &bcCanonicalizer = getBroadcastCanonicalizers()[grpIdx];
  auto &tpCanonicalizer = getTransposeCanonicalizers()[grpIdx];
  auto &shapeCastCanonicalizer = getShapeCastCanonicalizers()[grpIdx];
  return !rdCanonicalizer.getCandidateOps().empty() or
         !bcCanonicalizer.getCandidateOps().empty() or
         !tpCanonicalizer.getCandidateOps().empty() or
         !shapeCastCanonicalizer.getCandidateOps().empty();
}

void ForLoopGenerator::generateGroupOpVectorizedIR(const int idx) {
  auto &grp = getFusionStrategy().getOpGroups()[idx];
  if (grp.empty()) {
    LDBG("Current operation Group is empty.");
    return;
  }
  // TODO: special operation better fusion
  if (isGroupHasSpecialOperation(idx)) {
    return;
  }

  VectorType groupType =
      getFusionStrategy().getGroupBiggestRankVectorType()[idx];
  IRRewriter rewriter(grp.back());
  rewriter.setInsertionPointAfter(grp.back());
  // 1. Rewrite operation as vectorized form
  // 2. Generate loop
  // rewriteOperationAsVectorize(rewriter, idx);
  auto forOp = generateVectorizedForLoop(idx, rewriter, groupType);
  // special operation do not need to change anything
  if (failed(forOp)) {
    return;
  }
  moveLoopInvariantCode(forOp.value());
}

LogicalResult
moveFront(Operation *op,
          llvm::DenseMap<Operation *, size_t> &operationPosition) {
  IRRewriter rewriter(op);
  Operation *backOperation;
  size_t pos = 0;
  // check all the operand is block argument
  bool allBlockArgs = true;
  for (auto operand : op->getOperands()) {
    if (!isa<BlockArgument>(operand)) {
      allBlockArgs = false;
      break;
    }
  }
  if (allBlockArgs) {
    moveOpBeginingOfBlock(op);
    return success();
  }
  for (auto operand : op->getOperands()) {
    if (isa<BlockArgument>(operand))
      continue;

    Operation *sourceOp = operand.getDefiningOp();
    if (operationPosition[sourceOp] > pos and
        sourceOp->getBlock() == op->getBlock()) {
      backOperation = sourceOp;
      pos = operationPosition[sourceOp];
    }
  }
  if (pos == 0) {
    // extract operand operation all in previous block
    moveOpBeginingOfBlock(op);
    return success();
  }
  if (backOperation) {
    rewriter.moveOpAfter(op, backOperation);
    return success();
  }
  return failure();
}

LogicalResult moveBack(Operation *op,
                       llvm::DenseMap<Operation *, size_t> &operationPosition) {
  IRRewriter rewriter(op);
  Operation *firstOperation;
  size_t pos = std::numeric_limits<size_t>::max();
  for (auto user : op->getUsers()) {
    if (operationPosition[user] < pos and user->getBlock() == op->getBlock()) {
      firstOperation = user;
      pos = operationPosition[user];
    }
  }
  if (pos == std::numeric_limits<size_t>::max()) {
    // Don't move.
    // TODO: need to consider move before the block which use it.
    return success();
  }
  if (firstOperation) {
    rewriter.moveOpBefore(op, firstOperation);
    return success();
  }
  return failure();
}

void moveCandidateOperation(
    llvm::DenseMap<Operation *, size_t> &operationPosition,
    ArrayRef<Operation *> candidateOps) {

  for (Operation *op : candidateOps) {
    auto ret =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<affine::AffineApplyOp>([&](affine::AffineApplyOp affineOp) {
              return moveFront(op, operationPosition);
            })
            .Case<tensor::ExtractSliceOp>(
                [&](tensor::ExtractSliceOp extractOp) {
                  return moveFront(op, operationPosition);
                })
            .Case<tensor::EmptyOp>([&](tensor::EmptyOp emptyOp) {
              return moveFront(op, operationPosition);
            })
            .Case<tensor::InsertSliceOp>([&](tensor::InsertSliceOp insertOp) {
              return moveBack(op, operationPosition);
            })
            .Case<vector::TransferReadOp>([&](vector::TransferReadOp readOp) {
              return moveFront(op, operationPosition);
            })
            .Case<vector::TransferWriteOp>(
                [&](vector::TransferWriteOp writeOp) {
                  return moveBack(op, operationPosition);
                })
            .Default([&](Operation *op) { return success(); });
    if (failed(ret)) {
      LDBG("Wrong to move operation:" << *op << "\n");
      return;
    }
  }
}

// Need to move some operations like extract_slice or insert_slice.
// Because those operation may interpret our analysis result. e.g.:
// ```
// clang-format off
  // %21 = vector.transfer_read %18[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<16x16xf32>, vector<16x16xf32>
  // %22 = arith.addf %21, %20 : vector<16x16xf32>
  // %23 = vector.transfer_write %22, %extracted_slice_12[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, tensor<16x16xf32>
  // %inserted_slice_13 = tensor.insert_slice %18 into %arg14[%arg13, 0] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<32x16xf32>
  // %extracted_slice_14 = tensor.extract_slice %arg16[%arg13, 0] [16, 16] [1, 1] : tensor<32x16xf32> to tensor<16x16xf32>
  // %24 = vector.transfer_read %cst_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<16x16xf32>, vector<16x16xf32>
  // %25 = arith.maximumf %22, %24 : vector<16x16xf32>
  // %26 = vector.transfer_write %25, %extracted_slice_14[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, tensor<16x16xf32>
  // %inserted_slice_15 = tensor.insert_slice %23 into %arg15[%arg13, 0] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<32x16xf32>
  // %inserted_slice_16 = tensor.insert_slice %26 into %arg16[%arg13, 0] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<32x16xf32>
// clang-format on
// ```
// The maximumf and addf operation can be a same group, but the extract_slice
// operation interpret us.
// The move operation(extra_slice) will check its parameters. In order to
// ensure that it does not affect the correctness of the result, we will only
// move the moved op after the op to which the parameters belong to. If it's
// operand is all the block argument, we will move it to the begining of the
// block.
// insert_slice just move them to the privious of the first operation which
// use it.
void moveSomeInterferenceOperation(
    func::FuncOp *func, MLIRContext *ctx,
    std::function<bool(Operation *)> &conditionalFunc) {
  // Pre-order traversal of each op
  // Record each operation position. Inorder to we can kown current operation
  // should move after which operation.
  DenseMap<Operation *, size_t> operationPosition;
  SmallVector<Operation *, 8> candidateOps;
  size_t opCounter = 0;

  // get the position of each operation
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    operationPosition[op] = opCounter++;
    if (conditionalFunc(op))
      candidateOps.emplace_back(op);
  });
  moveCandidateOperation(operationPosition, candidateOps);
  // eliminate some useless operation
  RewritePatternSet patterns(ctx);
  (void)applyPatternsAndFoldGreedily(*func, std::move(patterns));
}

/// Pass that lower to physical vector.
struct CPUPhysicalRegisterPass
    : public impl::CPUPhysicalRegisterPassBase<CPUPhysicalRegisterPass> {

  void runOnOperation() final {
    //
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    if (hasNotSupportOperation(&func)) {
      LDBG("Not support operation appears in current function.");
      return;
    }
    // affineApply operation is always used by other operations.
    std::function<bool(Operation *)> candidateFunc = isUsedByOtherOp;
    moveSomeInterferenceOperation(&func, ctx, candidateFunc);
    candidateFunc = isCandidateMoveOperations;
    moveSomeInterferenceOperation(&func, ctx, candidateFunc);
    // canonicalize vector operation, default use vector-based fusion
    // strategy.
    HardWareInfo hwInfo;
    CPUTargetDescriptionAnalysis sysDesc =
        getAnalysis<CPUTargetDescriptionAnalysis>();
    hwInfo.favx512f = sysDesc.getMaxVectorWidth() == 512;
    hwInfo.favx2 = sysDesc.getMaxVectorWidth() >= 256;
    CanonicalizerVectorOperation canonicalizer(
        func, CanonicalizerKind::OperationsGroup, hwInfo);
    canonicalizer.run();

    candidateFunc = isReadOrWriteOperation;
    moveSomeInterferenceOperation(&func, ctx, candidateFunc);

    // transpose kernel
    vector::VectorTransformsOptions transposeOptions =
        vector::VectorTransformsOptions();
    transposeOptions.vectorTransposeLowering =
        vector::VectorTransposeLowering::Shuffle16x16;
    vector::populateVectorTransposeLoweringPatterns(patterns, transposeOptions);

    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

} // namespace gc
} // namespace mlir