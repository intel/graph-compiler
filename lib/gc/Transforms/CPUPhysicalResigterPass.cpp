//===- CPUPhysicalResigterPass.cpp.cpp - OneDNNGraph To Linalg
// Lowering -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Transforms/TilingVector.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_CPUPHYSICALREGISTERPASS
#include "gc/Transforms/Passes.h.inc"
namespace {
#define DEBUG_TYPE "lower-to-physical-register-pass"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define ARITH_CAST_OPERATIONS                                                  \
  arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::BitcastOp,             \
      arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp, arith::UIToFPOp

// TODO: remove it in the future
bool disableSpecialOp = false;

void printGroupOps(SmallVector<std::queue<Operation *>, 8> &opGroups) {
  for (auto grp : opGroups) {
    if (grp.empty()) {
      continue;
    }
    std::cout << "__________________ group start_____________" << std::endl;
    std::queue<Operation *> tmpQ(grp);
    while (!tmpQ.empty()) {
      auto cur = tmpQ.front();
      tmpQ.pop();
      cur->dump();
    }
    std::cout << "__________________ group end_____________" << std::endl;
    std::cout << std::endl;
  }
}

void printQueue(const std::queue<Operation *> &opQueue) {
  std::cout << "________________________________ op Queue "
               "__________________"
            << std::endl;
  auto tempQ(opQueue);
  while (!tempQ.empty()) {
    auto cur = tempQ.front();
    cur->dump();
    tempQ.pop();
  }
  std::cout << "________________________________ op queue end "
               "__________________"
            << std::endl;
}

bool isSpecialOp(Operation *op) {
  return isa<vector::TransposeOp>(op) || isa<vector::BroadcastOp>(op) ||
         isa<vector::ReductionOp>(op) || isa<vector::ShapeCastOp>(op) ||
         isa<vector::MultiDimReductionOp>(op) || isa<func::CallOp>(op);
}

bool is_innermost_operation(Operation *op) {
  bool inner_most = true;
  op->walk([&inner_most](Operation *p) {
    if (mlir::isa<scf::ForOp>(p)) {
      inner_most = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return inner_most;
}

bool isNotSupportOperation(Operation *op) {
  return isa<vector::MaskOp>(op) || isa<vector::ConstantMaskOp>(op) ||
         isa<vector::MaskedLoadOp>(op) || isa<vector::MaskedStoreOp>(op) ||
         isa<vector::CreateMaskOp>(op);
}

bool isReadOrWriteOperation(Operation *op) {
  return isa<vector::TransferReadOp>(op) || isa<vector::TransferWriteOp>(op);
}

// TODO: Need to support these operations in the future
bool hasNotSupportOperation(func::FuncOp *func) {
  auto walkRes = func->walk([](Operation *op) {
    if (isNotSupportOperation(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walkRes != WalkResult::advance();
}

// select nearest even step
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

int TypeHelper::generateValidSteps(int steps, VectorType type) {
  return type.getShape().back() >= steps
             ? (steps > 16 ? 16 : steps)
             : getNearestVectorStep(type.getShape().back());
}

// expr equals `vector rank` - 1
bool isLastDim(const AffineExpr &expr, const size_t rank) {
  return mlir::isa<AffineDimExpr>(expr) &&
         mlir::dyn_cast<AffineDimExpr>(expr).getPosition() == rank - 1;
}

[[nodiscard]] int TypeHelper::getDataTypeValidSteps(VectorType type) {
  auto typebits = type.getElementTypeBitWidth();
  const int favx512bits = 512;
  const int favx2bits = 256;
  if (HWInfo.favx512f) {
    return generateValidSteps(favx512bits / typebits, type);
  } else if (HWInfo.favx2) {
    return generateValidSteps(favx2bits / typebits, type);
  } else {
    // invalid
    LDBG("Please check the hardware information.");
    assert(false && "Invalid hardware.");
    return -1;
  }
}

// Get the maximum number of current data types that a register can hold
[[nodiscard]] int TypeHelper::getDataTypeMAXSIMDLength(VectorType type) {
  auto typebits = type.getElementTypeBitWidth();
  const int favx512bits = 512;
  const int favx2bits = 256;
  if (HWInfo.favx512f) {
    return favx512bits / typebits;
  } else if (HWInfo.favx2) {
    return favx2bits / typebits;
  } else {
    // invalid
    LDBG("Please check the hardware information.");
    assert(false && "Invalid hardware.");
    return -1;
  }
}

FailureOr<Value> createArithSplatConstantOp(IRRewriter &rewriter,
                                            const Location &loc,
                                            const ElementsAttr &valueType,
                                            VectorType &newOperandType) {

  if (valueType.isSplat()) {
    Value res;
    if (mlir::isa<FloatType>(valueType.getElementType())) {
      res = rewriter.create<arith::ConstantOp>(
          loc,
          FloatAttr::get(newOperandType, valueType.getSplatValue<APFloat>()));
    } else {
      res = rewriter.create<arith::ConstantOp>(
          loc,
          IntegerAttr::get(newOperandType, valueType.getSplatValue<APInt>()));
    }
    return res;
  }

  return failure();
}

mlir::FailureOr<VectorType> getOperationVectorType(Operation *op) {
  if (!op) {
    return failure();
  }
  auto isDynamicType = [](VectorType &type) { return !type.hasStaticShape(); };
  auto ret =
      TypeSwitch<Operation *, mlir::FailureOr<VectorType>>(op)
          .Case<vector::TransferWriteOp>(
              [&](vector::TransferWriteOp transferWriteOp)
                  -> mlir::FailureOr<VectorType> {
                auto retType = mlir::dyn_cast<VectorType>(
                    transferWriteOp->getOperand(0).getType());
                if (retType) {
                  return retType;
                }
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
                return multiReductionOp.getSourceVectorType();
              })
          .Case<arith::ConstantOp>(
              [&](arith::ConstantOp constantOp) -> mlir::FailureOr<VectorType> {
                return failure();
              })
          .Default([&](Operation *op) -> mlir::FailureOr<VectorType> {
            if (!op->getResults().empty()) {
              auto t = dyn_cast<VectorType>(op->getResultTypes().front());
              if (t) {
                if (isDynamicType(t)) {
                  return failure();
                }
                return t;
              }
            }
            return failure();
          });
  if (!failed(ret) and isDynamicType(ret.value())) {
    return failure();
  }
  return ret;
}

VectorType TypeHelper::getVectorzedType(Operation *op, uint32_t loop_step) {
  // Check that the operation type can be broken
  // down into a loop.
  auto baseType = getOperationVectorType(op);
  if (failed(baseType)) {
    LDBG("Failed to get vector type for operation: " << *op << "\n");
    assert(false && "Failed to get vector type for operation");
    return VectorType();
  }
  auto vectorizedType = baseType.value();
  if (loop_step == 0) {
    loop_step = getDataTypeValidSteps(vectorizedType);
  }
  return VectorType::get({loop_step}, vectorizedType.getElementType());
}

/// whether the operation result need to be returned
/// \param anchorIdx resuilt produce operation anchor position
/// \param retType resuilt return type
bool needReturnResult(std::pair<ReturnTypeKind, size_t> &retType,
                      size_t anchorIdx) {
  return !(retType.first == ReturnTypeKind::RT_InGroup and
           retType.second >= anchorIdx);
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
  if (mlir::isa<vector::TransferReadOp>(op) ||
      mlir::isa<vector::TransferWriteOp>(op)) {
    auto permutationMap =
        mlir::dyn_cast<vector::TransferReadOp>(op)
            ? mlir::dyn_cast<vector::TransferReadOp>(op).getPermutationMap()
            : mlir::dyn_cast<vector::TransferWriteOp>(op).getPermutationMap();
    auto rank =
        mlir::dyn_cast<vector::TransferReadOp>(op)
            ? mlir::dyn_cast<ShapedType>(op->getOperand(0).getType()).getRank()
            : mlir::dyn_cast<ShapedType>(op->getOperand(1).getType()).getRank();
    auto dimExpr = permutationMap.getResults();
    bool find = false;
    for (auto &expr : dimExpr) {
      if (isLastDim(expr, rank)) {
        find = true;
      }
    }
    return find;
  }
  LDBG("The operation is not a read or write operation." << *op << "\n");
  assert(0 && "The operation is not a read or write operation.");
  return false;
}

// std::variant<float, int64_t> numeric_zero(Type type) {
//   Type t1 = getElementTypeOrSelf(type);
//   if (t1.isF32()) {
//     return 0.f;
//   } else if (t1.isBF16()) {
//     return bfloat2float(float2bfloat(0.f));
//   } else if (t1.isF16()) {
//     return half2float(float2half(0.f));
//   } else if (t1.isSignedInteger(8)) {
//     return int64_t(0);
//   } else if (t1.isSignedInteger(32)) {
//     return int64_t(0);
//   } else if (t1.isSignlessInteger(8) or t1.isSignlessInteger(32)) {
//     return int64_t(0);
//   } else {
//     LDBG("Unsupported data type: " << t1 << "\n");
//     assert(0 && "unsupported data type");
//     return (int64_t)0;
//   }
// }

// std::variant<float, int64_t> numeric_one(Type type) {
//   Type t1 = getElementTypeOrSelf(type);
//   if (t1.isF32()) {
//     return 1.f;
//   } else if (t1.isBF16()) {
//     return bfloat2float(float2bfloat(1.f));
//   } else if (t1.isF16()) {
//     return half2float(float2half(1.f));
//   } else if (t1.isSignedInteger(8)) {
//     return int64_t(1);
//   } else if (t1.isSignedInteger(32)) {
//     return int64_t(1);
//   } else if (t1.isSignlessInteger(8) or t1.isSignlessInteger(32)) {
//     return int64_t(1);
//   } else {
//     LDBG("Unsupported data type: " << t1 << "\n");
//     assert(0 && "unsupported data type");
//     return (int64_t)1;
//   }
// }

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

template <class T = float>
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
                                      const RankedTensorType &tensorType,
                                      const AffineMap &permutationMap) {

  auto dimExpr = permutationMap.getResults();
  auto lastDim = mlir::dyn_cast<AffineDimExpr>(dimExpr.back());
  assert(mlir::isa<AffineDimExpr>(lastDim));

  SmallVector<AffineExpr, 1> affineExprs;
  affineExprs.push_back(lastDim);
  auto destAffineMap = AffineMap::get(tensorType.getRank(), 0, affineExprs,
                                      rewriter.getContext());
  SmallVector<bool> inBounds(1, true);
  if (mlir::isa<vector::TransferWriteOp>(op)) {
    auto transferWriteOp = mlir::dyn_cast<vector::TransferWriteOp>(op);
    transferWriteOp.setPermutationMap(destAffineMap);
    transferWriteOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  } else if (mlir::isa<vector::TransferReadOp>(op)) {
    auto transferReadOp = mlir::dyn_cast<vector::TransferReadOp>(op);
    transferReadOp.setPermutationMap(destAffineMap);
    transferReadOp.setInBoundsAttr(rewriter.getBoolArrayAttr(inBounds));
  }
}

// scf.for yield helper function
scf::YieldOp maybeYieldValue(OpBuilder &b, Location loc,
                             const ValueRange &value) {
  bool hasRetVal = !value.empty();
  if (hasRetVal) {
    return b.create<scf::YieldOp>(loc, value);
  } else {
    return b.create<scf::YieldOp>(loc);
  }
}

Type getScalarType(Operation *op) {
  // Check that the operation type can be broken
  // down into a loop.
  auto baseType = getOperationVectorType(op);
  if (failed(baseType)) {
    LDBG("Failed to get vector type for operation: " << *op << "\n");
    assert(false && "Failed to get vector type for operation");
    return VectorType();
  }
  auto vectorizedType = baseType.value();
  return VectorType::get({1}, vectorizedType.getElementType());
}

Operation *createTensorEmptyBefore(Operation *op) {
  auto rtType = mlir::dyn_cast<ShapedType>(op->getResultTypes()[0]);
  IRRewriter reWriter(op);

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i)) {
      dynDims.push_back(reWriter.create<tensor::DimOp>(reWriter.getUnknownLoc(),
                                                       op->getResult(0), i));
    }
  }
  return reWriter.create<tensor::EmptyOp>(op->getLoc(), rtType.getShape(),
                                          rtType.getElementType(), dynDims);
}

Value getOperationResultTensor(Operation *op) {
  auto result = op->getResults()[0];
  for (auto x : result.getUsers()) {
    if (mlir::isa<vector::TransferWriteOp>(x)) {
      return x->getOperand(1);
    }
  }
  LDBG("Result not write back to tensor.");

  return createTensorEmptyBefore(op)->getResults()[0];
}

Operation *createTransferWriteOpAfter(Operation *op, const Value &dest) {
  auto rtType = mlir::dyn_cast<ShapedType>(op->getResultTypes()[0]);
  auto rank = rtType.getRank();
  auto dstType = mlir::dyn_cast<ShapedType>(dest.getType());
  IRRewriter reWriter(op);

  auto zero =
      reWriter.create<arith::ConstantIndexOp>(reWriter.getUnknownLoc(), 0);

  reWriter.setInsertionPointAfter(op);
  SmallVector<bool> inBoundsVal(rank, true);

  SmallVector<int64_t> shapes;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < rtType.getRank(); i++) {
    shapes.push_back(rtType.getDimSize(i));
    if (rtType.isDynamicDim(i)) {
      dynDims.push_back(reWriter.create<tensor::DimOp>(reWriter.getUnknownLoc(),
                                                       op->getResult(0), i));
    }
  }
  return reWriter.create<vector::TransferWriteOp>(
      reWriter.getUnknownLoc(),
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
  } else {
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
}

// canonicalizing operation as tensor empty and transfer write the operation
// result into the empty tensor
[[nodiscard]] std::pair<Value, Value>
canonicalizeSourceOperation(Operation *op) {
  auto resultTensor = getOperationResultTensor(op);
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
    auto accOperand = srcOperandsQueue.front();
    srcOperandsQueue.pop_front();
    auto accOperandOp = accOperand.getDefiningOp();
    if (!accOperandOp or visited.count(accOperandOp)) {
      continue;
    }
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
  if (isa<FloatType>(resultElementType)) {
    initValueAttr = FloatAttr::get(
        resultElementType,
        getInitValForReduce(multiReductionOp.getKind(), vecType));
  } else {
    initValueAttr = IntegerAttr::get(
        resultElementType,
        getInitValForReduce<int64_t>(multiReductionOp.getKind(), vecType));
  }
}

void classifySourceRelatedOps(std::queue<Operation *> &accRelatedOps,
                              std::queue<Operation *> &sourceRelatedOps,
                              Operation *srcOp,
                              std::queue<Operation *> &prevOps) {
  DenseSet<Operation *> srcOps;
  getOpSourceOps(srcOp, srcOps);
  while (!prevOps.empty()) {
    auto op = prevOps.front();
    prevOps.pop();
    if (isSrcRelated(srcOps, op) or op == srcOp) {
      sourceRelatedOps.push(op);
    } else {
      accRelatedOps.push(op);
    }
  }
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
    if (isSrcRelated(srcOpsSet, op) or op == srcOp) {
      accRelatedOps.push(op);
    } else {
      sourceRelatedOps.push(op);
    }
  }
}

void updateReduceReadWriteOperationOperand(
    const SmallVector<Value, 5> &inductionVars,
    const SmallVector<int64_t, 4> &parallelAxis, Operation *op,
    MultiReduceOpAxisKind rdKind = MultiReduceOpAxisKind::Parallel) {
  int indiceOffset = mlir::isa<vector::TransferReadOp>(op) ? 1 : 2;
  for (auto [idx, inductionVar] : llvm::enumerate(inductionVars)) {
    if (rdKind == MultiReduceOpAxisKind::Parallel &&
        idx >= parallelAxis.size()) {
      break;
    }
    op->setOperand(idx + indiceOffset, inductionVar);
  }
}

vector::TransferReadOp ForLoopGenerator::cloneReductionTransferRead(
    Value &source, OpBuilder &b, IRMapping &readMap,
    const SmallVector<int64_t, 4> &parallelAxis,
    SmallVector<Value, 5> &inductionVars, bool lastDimReduction,
    MultiReduceOpAxisKind rdKind) {
  IRRewriter rewriter(b);
  auto readOp = mlir::dyn_cast<vector::TransferReadOp>(source.getDefiningOp());
  assert(readOp && " Not transfer_read operation. Current multireduction "
                   "operation may have wrong analysis IR.");

  auto clonedOp = b.clone(*readOp, readMap);
  auto newReadOp = mlir::dyn_cast<vector::TransferReadOp>(clonedOp);
  updateReduceReadWriteOperationOperand(inductionVars, parallelAxis, newReadOp,
                                        rdKind);

  // modify the type of the new read operation
  auto newOperandType =
      (lastDimReduction && rdKind == MultiReduceOpAxisKind::Reduction)
          ? getVectorzedType(newReadOp)
          : getScalarType(newReadOp);
  newReadOp->getResult(0).setType(newOperandType);
  setOpVectorizationPermutationMap(
      newReadOp, b,
      mlir::dyn_cast<RankedTensorType>(newReadOp.getSource().getType()),
      newReadOp.getPermutationMap());

  rewriter.replaceOp(readOp, newReadOp);
  return newReadOp;
}

vector::TransferWriteOp
makeNewTransferWriteOp(Value source, IRMapping &writeMap, OpBuilder &b,
                       const SmallVector<int64_t, 4> &parallelAxis,
                       SmallVector<Value, 5> &inductionVars) {
  IRRewriter bodyRewriter(b);
  auto writeOp = source.getDefiningOp();
  auto newWriteOp =
      mlir::dyn_cast<vector::TransferWriteOp>(b.clone(*writeOp, writeMap));
  updateReduceReadWriteOperationOperand(inductionVars, parallelAxis, newWriteOp,
                                        MultiReduceOpAxisKind::Parallel);
  setOpVectorizationPermutationMap(
      newWriteOp, b,
      mlir::dyn_cast<RankedTensorType>(newWriteOp->getResult(0).getType()),
      newWriteOp.getPermutationMap());
  bodyRewriter.replaceOp(writeOp, newWriteOp);
  return newWriteOp;
}

Value makeIndexArithConstantOp(OpBuilder &opBuilder, const Location &loc,
                               int64_t x) {
  return opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(), x));
}

void ForLoopGenerator::moveOperationsToCurrentForBody(
    const size_t groupIdx, const OpBuilder &b, ArrayRef<Value> inductionVars,
    const DenseMap<Value, int> &operandIdxMap, const ValueRange &loopState,
    std::queue<Operation *> &opQueue) {
  auto &opPermuationMap = getOpPermuationMap();
  auto tmpQ(opQueue);
  while (!tmpQ.empty()) {
    auto x = tmpQ.front();
    tmpQ.pop();
    x->moveBefore(b.getBlock(), b.getBlock()->end());
    // check operation type to set correct operand
    setOperationCorrectOperand(x, loopState, operandIdxMap, inductionVars,
                               opPermuationMap);
  }
}

bool hasOtherOperations(const std::queue<Operation *> &opQ,
                        const Operation *multiReductionOp) {
  bool res = false;
  if (!opQ.empty()) {
    std::queue<Operation *> tempQ(opQ);
    while (!tempQ.empty()) {
      auto cur = tempQ.front();
      tempQ.pop();
      if (!isReadOrWriteOperation(cur) and cur != multiReductionOp) {
        res = true;
        break;
      }
    }
  }
  return res;
};

void ForLoopGenerator::getResultInCurrentOps(
    const size_t anchorIdx, const size_t groupId,
    const std::queue<Operation *> ops, SmallVector<Value, 4> &results,
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
        forResultOrignalResultMap[curResult] = curResult;
      }
    }
  }
}

/// update loop args related status
/// \param nextAnchorArgsIdxMap anchor args index map
/// \param nextOperandArgsMap original value to next loop args map
/// \param nextArgsOperandMap next loop args to original value map
void updateCurrentArgsStatus(const ValueRange &loopState,
                             const size_t loopStateIdx,
                             SmallVector<Value, 4> &nextAnchorArgs,
                             Value originalValue,
                             DenseMap<Value, int> &nextAnchorArgsIdxMap,
                             DenseMap<Value, Value> &nextOperandArgsMap,
                             DenseMap<Value, Value> &nextArgsOperandMap) {
  Value currentArgs = loopState[loopStateIdx];
  nextAnchorArgs.emplace_back(currentArgs);
  nextAnchorArgsIdxMap[currentArgs] = nextAnchorArgs.size() - 1;
  nextOperandArgsMap[originalValue] = currentArgs;
  nextArgsOperandMap[currentArgs] = originalValue;
}

void ForLoopGenerator::getInitArgsToNextAnchor(
    const size_t anchorIdx, const size_t groupId,
    const std::queue<Operation *> &nextOperations, const ValueRange &loopState,
    DenseMap<Value, int> &currentLoopStateIdxMap,
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs,
    DenseMap<Value, Value> &originalOperandLoopArgsMap,
    DenseMap<Value, Value> &loopArgsOriginalOperandMap) {
  DenseMap<Operation *, size_t> &opAnchorPos =
      getFusionStrategy().getOpAnchorPos();
  SetVector<Value> &opInitArgs = getGroupOpInitArgs()[groupId];

  DenseSet<Value> visited;
  // find the next anchor arguments
  std::queue<Operation *> tmpQ(nextOperations);
  DenseMap<Value, Value> nextOperandArgsMap, nextArgsOperandMap;

  while (!tmpQ.empty()) {
    Operation *cur = tmpQ.front();
    tmpQ.pop();
    auto curOperands = cur->getOperands();
    for (auto x : curOperands) {
      if (!visited.contains(x) and opInitArgs.contains(x) and
          opAnchorPos[cur] > anchorIdx) {
        int loopStateIdx =
            currentLoopStateIdxMap[originalOperandLoopArgsMap[x]];
        updateCurrentArgsStatus(loopState, loopStateIdx, nextAnchorArgs, x,
                                nextAnchorArgsIdxMap, nextOperandArgsMap,
                                nextArgsOperandMap);
        visited.insert(x);
      }
    }
  }
  originalOperandLoopArgsMap = nextOperandArgsMap;
  loopArgsOriginalOperandMap = nextArgsOperandMap;
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
    IRRewriter &rewrite, const ValueRange &forResults, const Block *forBlock,
    const llvm::SmallVector<Value, 4> &nextAnchorResults,
    const std::queue<Operation *> &movingOperations,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  auto tmpQ(movingOperations);
  DenseSet<Value> operationOperands;
  while (!tmpQ.empty()) {
    auto curOp = tmpQ.front();
    tmpQ.pop();
    for (auto x : curOp->getOperands()) {
      operationOperands.insert(x);
    }
  }
  auto replaceIfFn = [&](OpOperand &use) {
    return operationOperands.contains(use.get());
  };
  for (auto [nxtForResult, nextLoopResult] :
       zip(forResults, nextAnchorResults)) {
    Value originalResult = forResultOrignalResultMap[nextLoopResult];
    rewrite.replaceOpUsesWithIf(originalResult.getDefiningOp(), nxtForResult,
                                replaceIfFn);
  }
}

/// \param [out] nextLoopStateidxMap
/// \param [out] nextAnchorArgs
/// \param [out] movingQueue
void ForLoopGenerator::movePreOpToCurrentAnchor(
    const size_t anchorIdx, const size_t groupIdx, OpBuilder &b,
    ArrayRef<Value> inductionVars, const ValueRange &loopState,
    DenseMap<Value, int> &currentLoopStateIdxMap,
    DenseMap<Value, int> &nextAnchorArgsIdxMap,
    SmallVector<Value, 4> &nextAnchorArgs,
    std::queue<Operation *> &candidateQueue,
    std::queue<Operation *> &movedQueue,
    DenseMap<Value, Value> &originalOperandLoopArgsMap,
    DenseMap<Value, Value> &LoopArgsoriginalOperandMap) {

  // 1. get operations in current anchor position
  std::queue<Operation *> movingOperation;
  getOperationInCurrentAnchor(anchorIdx, candidateQueue, movingOperation);

  // 2. get next anchor args
  getInitArgsToNextAnchor(anchorIdx, groupIdx, candidateQueue, loopState,
                          currentLoopStateIdxMap, nextAnchorArgsIdxMap,
                          nextAnchorArgs, originalOperandLoopArgsMap,
                          LoopArgsoriginalOperandMap);

  // 3. rewrite operation as vectorize IR
  rewriteOperationAsVectorize(b, groupIdx, &movingOperation);

  // 4. move opeartions to current for block
  moveOperationsToCurrentForBody(groupIdx, b, inductionVars,
                                 currentLoopStateIdxMap, loopState,
                                 movingOperation);

  // 5. move operations to moved queue
  while (!movingOperation.empty()) {
    movedQueue.push(movingOperation.front());
    movingOperation.pop();
  }
}

void ForLoopGenerator::movePostOpToCurrentAnchor(
    OpBuilder &b, const int anchorIdx, const int groupIdx,
    const ValueRange &forResults, const Block *forBlock,
    std::queue<Operation *> &candidateOps, std::queue<Operation *> &movedOps,
    ArrayRef<Value> inductionVars, const DenseMap<Value, int> &operandIdxMap,
    const ValueRange &loopState, const SmallVector<Value, 4> &nextAnchorResults,
    DenseMap<Value, Value> &forResultOrignalResultMap) {

  // 1. move post-op to current loop body
  std::queue<Operation *> movingOperations;
  getOperationInCurrentAnchor(anchorIdx, candidateOps, movingOperations);

  rewriteOperationAsVectorize(b, groupIdx, &movingOperations);

  moveOperationsToCurrentForBody(anchorIdx, b, inductionVars, operandIdxMap,
                                 loopState, movingOperations);

  // 2. replace correct for loop result to post-op
  IRRewriter rewriter(b);
  replaceOperationsWithForLoopResult(rewriter, forResults, forBlock,
                                     nextAnchorResults, movingOperations,
                                     forResultOrignalResultMap);

  // 3. move operations to moved queue
  while (!movingOperations.empty()) {
    movedOps.push(movingOperations.front());
    movingOperations.pop();
  }
}

void ForLoopGenerator::generateLoopResults(
    OpBuilder &b, const Location &loc, const size_t anchorIdx,
    const size_t groupIdx, SmallVector<Value, 4> &nextAnchorResults,
    DenseMap<Value, int> &nextAnchorResultsIdxMap, const ValueRange &forResults,
    const std::queue<Operation *> &movedOperation,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  SmallVector<Value, 4> results;
  DenseMap<Value, Value> currentResultMap;
  getResultInCurrentOps(anchorIdx, groupIdx, movedOperation, results,
                        currentResultMap);

  llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &groupResults =
      getGroupOpResults()[groupIdx];
  // check for yield results whether need to return to next anchor
  for (auto [idx, forResult] : llvm::enumerate(nextAnchorResults)) {
    Value originalResult = forResultOrignalResultMap[forResult];

    if (groupResults.contains(originalResult)) {
      std::pair<ReturnTypeKind, size_t> resultType =
          groupResults[originalResult];
      if (needReturnResult(resultType, anchorIdx)) {
        results.emplace_back(forResults[idx]);
        currentResultMap[forResults[idx]] = originalResult;
      }
    }
  }

  nextAnchorResults.clear();
  nextAnchorResultsIdxMap.clear();
  for (Value &result : results) {
    nextAnchorResults.emplace_back(result);
    nextAnchorResultsIdxMap[result] = nextAnchorResults.size() - 1;
  }
  forResultOrignalResultMap = std::move(currentResultMap);
}

scf::ForOp ForLoopGenerator::reductionAxisGenerateForLoop(
    OpBuilder &opBuilder, const int groupIdx, const size_t reductionIdx,
    const int anchorIdx, llvm::DenseMap<Value, int> &currentLoopStateIdxMap,
    const ValueRange &initArgs,
    DenseMap<Value, Value> &originalOperandLoopArgsMap,
    DenseMap<Value, Value> &loopArgsOriginalOperandMap,
    llvm::SmallVector<Value, 4> &nextAnchorResults,
    llvm::DenseMap<Value, int> &nextAnchorResultsIdxMap,
    llvm::SmallVector<Value, 5> &inductionVars,
    DenseMap<Value, Value> &forResultOrignalResultMap,
    DenseMap<Value, Value> &originalResultForResultMap) {

  MultiReductionCanonicalizer rdCanonicalizer =
      getMultiRdCanonicalizers()[groupIdx];
  auto &multireductionOp = rdCanonicalizer.getCandidateOps()[0];
  VectorFusionStrategy &fusionStrategy = getFusionStrategy();

  SmallVector<std::queue<Operation *>, 8> &opGroups =
      fusionStrategy.getOpGroups();
  std::queue<Operation *> &opQueue = opGroups[groupIdx];

  const auto loc = multireductionOp->getLoc();
  SmallVector<int64_t, 4> &reductionAxis = rdCanonicalizer.getReductionAxis();
  bool lastDimReduction = rdCanonicalizer.hasLastDimReduction();
  VectorType vectorType = rdCanonicalizer.getSourceType();
  const int loopStep = getFusionStrategy().getGroupMaxSteps()[groupIdx];

  IRRewriter rewriterOfFunc(func);

  Value zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  Value forSteps = makeIndexArithConstantOp(
      opBuilder, loc,
      (reductionIdx == reductionAxis.size() - 1 && lastDimReduction) ? loopStep
                                                                     : 1);
  Value numIter = makeIndexArithConstantOp(
      opBuilder, loc, vectorType.getShape()[reductionAxis[reductionIdx]]);
  scf::ForOp forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, initArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        if (reductionIdx < reductionAxis.size() - 1) {

          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          std::queue<Operation *> movedOperation;
          DenseMap<Value, Value> originalArgsMap, argsOriginalMap;
          movePreOpToCurrentAnchor(
              anchorIdx, groupIdx, b, inductionVars, loopState,
              currentLoopStateIdxMap, nextAnchorArgsIdxMap, nextAnchorArgs,
              opQueue, movedOperation, originalArgsMap, argsOriginalMap);

          // replace reduction init args
          if (originalOperandLoopArgsMap.contains(multireductionOp.getAcc())) {
            size_t accValIdx = currentLoopStateIdxMap
                [originalOperandLoopArgsMap[multireductionOp.getAcc()]];
            updateCurrentArgsStatus(
                loopState, accValIdx, nextAnchorArgs, multireductionOp.getAcc(),
                nextAnchorArgsIdxMap, originalArgsMap, argsOriginalMap);
          }

          // 2. generate next for loop
          scf::ForOp nxtFor = reductionAxisGenerateForLoop(
              b, groupIdx, reductionIdx + 1, anchorIdx + 1,
              nextAnchorArgsIdxMap, nextAnchorArgs, originalArgsMap,
              argsOriginalMap, nextAnchorResults, nextAnchorResultsIdxMap,
              inductionVars, forResultOrignalResultMap,
              originalResultForResultMap);

          // 3. move postOp to current body
          movePostOpToCurrentAnchor(
              b, anchorIdx, groupIdx, nxtFor->getResults(), b.getBlock(),
              opQueue, movedOperation, inductionVars, currentLoopStateIdxMap,
              loopState, nextAnchorResults, forResultOrignalResultMap);

          // 4. generate loop results
          generateLoopResults(b, loc, anchorIdx, groupIdx, nextAnchorResults,
                              nextAnchorResultsIdxMap, nxtFor->getResults(),
                              movedOperation, forResultOrignalResultMap);

          // reduction must return acc
          if (originalResultForResultMap.contains(
                  multireductionOp->getResults()[0])) {
            Value originalValue =
                originalResultForResultMap[multireductionOp->getResults()[0]];
            size_t retIdx =
                nextAnchorArgsIdxMap[forResultOrignalResultMap[originalValue]];
            Value forRes = nxtFor->getResults()[retIdx];

            nextAnchorResults.emplace_back(forRes);
            nextAnchorResultsIdxMap[forRes] = nextAnchorResults.size() - 1;
            forResultOrignalResultMap[forRes] = originalValue;
            originalResultForResultMap[originalValue] = forRes;
          }

          maybeYieldValue(b, loc, nextAnchorResults);

        } else if (reductionIdx == reductionAxis.size() - 1) {
          std::queue<Operation *> movingOperation;

          while (!opQueue.empty()) {
            Operation *curOp = opQueue.front();
            opQueue.pop();
            if (isa<vector::MultiDimReductionOp>(curOp)) {
              break;
            }
            movingOperation.push(curOp);
          }
          while (!opQueue.empty()) {
            Operation *curOp = opQueue.front();
            if (isa<vector::MultiDimReductionOp>(curOp)) {
              opQueue.pop();
              continue;
            }
            break;
          }

          rewriteOperationAsVectorize(b, groupIdx, &movingOperation);

          moveOperationsToCurrentForBody(groupIdx, b, inductionVars,
                                         currentLoopStateIdxMap, loopState,
                                         movingOperation);

          int accValIdx = currentLoopStateIdxMap
              [originalOperandLoopArgsMap[multireductionOp.getAcc()]];

          Value reductionResult = makeArithReduction(
              b, loc, multireductionOp.getKind(), multireductionOp.getSource(),
              loopState[accValIdx]);

          movePostOpToCurrentAnchor(
              b, anchorIdx, groupIdx, ValueRange(), b.getBlock(), opQueue,
              movingOperation, inductionVars, currentLoopStateIdxMap, loopState,
              nextAnchorResults, forResultOrignalResultMap);

          nextAnchorResults.clear();
          nextAnchorResults.emplace_back(reductionResult);
          nextAnchorResultsIdxMap[reductionResult] = 0;
          forResultOrignalResultMap[reductionResult] =
              multireductionOp->getResults()[0];
          originalResultForResultMap[multireductionOp->getResults()[0]] =
              reductionResult;
          getResultInCurrentOps(anchorIdx, groupIdx, movingOperation,
                                nextAnchorResults, forResultOrignalResultMap);

          maybeYieldValue(b, loc, nextAnchorResults);
        }
      });

  return forOp;
}

// Generate for loop for parallel axis of `vector.multi_reduction`.
// This function also call reduction axis for loop
scf::ForOp ForLoopGenerator::parallelAxisGenerateForLoop(
    OpBuilder &opBuilder, const int groupIdx, const size_t parallelIdx,
    DenseMap<Value, int> &currentLoopStateIdxMap, const ValueRange &initArgs,
    SmallVector<Value, 4> &nextAnchorResults,
    DenseMap<Value, int> &nextAnchorResultsIdxMap,
    SmallVector<Value, 5> &inductionVars,
    DenseMap<Value, Value> &originalOperandLoopArgsMap,
    DenseMap<Value, Value> &loopArgsOriginalOperandMap,
    DenseMap<Value, Value> &forResultOrignalResultMap) {
  auto &rdCanonicalizer = getMultiRdCanonicalizers()[groupIdx];
  vector::MultiDimReductionOp &multiReductionOp =
      rdCanonicalizer.getCandidateOps()[0];
  VectorType vectorType = rdCanonicalizer.getSourceType();
  IRRewriter rewriterOfFunc(func);

  SmallVector<int64_t, 4> &parallelAxis = rdCanonicalizer.getParallelAxis();
  const Location &loc = multiReductionOp.getLoc();
  Value zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  Value forSteps = makeIndexArithConstantOp(opBuilder, loc, 1);

  // last dim reduction need to a generate dim=16 loop for fused with pre-op
  int dimSize = 0;
  if (parallelIdx == parallelAxis.size()) {
    dimSize = getFusionStrategy().getGroupMaxSteps()[groupIdx];
  } else {
    dimSize = vectorType.getShape()[parallelAxis[parallelIdx]];
  }
  Value numIter = makeIndexArithConstantOp(opBuilder, loc, dimSize);
  // Create a loop and move vectorized operation into loops.
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, initArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);
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

        if (parallelIdx < parallelAxis.size()) {
          // 1. move pre-Op to current body
          DenseMap<Value, int> nextAnchorArgsIdxMap;
          SmallVector<Value, 4> nextAnchorArgs;
          std::queue<Operation *> movedQueue;
          movePreOpToCurrentAnchor(
              parallelIdx, groupIdx, b, inductionVars, loopState,
              currentLoopStateIdxMap, nextAnchorArgsIdxMap, nextAnchorArgs,
              opQueue, movedQueue, originalOperandLoopArgsMap,
              loopArgsOriginalOperandMap);

          if (parallelIdx == parallelAxis.size() - 1) {
            // Ensure accumalate expression in this parallel anchor position.
            // If it not appear in current anchor, we must move it in here.
            //   1. delete it in operation queue
            //   2. move it in current movedqueue
            DenseMap<Operation *, std::pair<Value, Value>> srcOpCanoniclizedMap;
            DenseSet<Value> argsSet(nextAnchorArgs.begin(),
                                    nextAnchorArgs.end());
            std::queue<Operation *> checkAccQueue(movedQueue);
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
              if (!argsSet.contains(accInitVal)) {
                nextAnchorArgs.emplace_back(accInitVal);
                nextAnchorArgsIdxMap[accInitVal] = nextAnchorArgs.size() - 1;
                loopArgsOriginalOperandMap[accInitVal] = multiReductionAcc;
                originalOperandLoopArgsMap[multiReductionAcc] = accInitVal;
              }

            } else {
              llvm::llvm_unreachable_internal(
                  "Wrong accumualte source value. Because "
                  "acc value must appear in here.");
            }
          }

          // 2. generate next for loop
          scf::ForOp nxtFor = parallelAxisGenerateForLoop(
              b, groupIdx, parallelIdx + 1, nextAnchorArgsIdxMap,
              nextAnchorArgs, nextAnchorResults, nextAnchorResultsIdxMap,
              inductionVars, originalOperandLoopArgsMap,
              loopArgsOriginalOperandMap, forResultOrignalResultMap);

          // 3. move postOp to current body
          movePostOpToCurrentAnchor(
              b, parallelIdx, groupIdx, nxtFor->getResults(),
              nxtFor->getBlock(), opQueue, movedQueue, inductionVars,
              currentLoopStateIdxMap, loopState, nextAnchorResults,
              forResultOrignalResultMap);

          // 4. generate loop results
          generateLoopResults(b, loc, parallelIdx, groupIdx, nextAnchorResults,
                              nextAnchorResultsIdxMap, nxtFor->getResults(),
                              movedQueue, forResultOrignalResultMap);
          maybeYieldValue(b, loc, nextAnchorResults);

        } else if (parallelIdx == parallelAxis.size()) {

          // get accumualte value
          Attribute initValueAttr;
          getReductionInitAttr(multiReductionOp, initValueAttr);

          auto accVal = b.create<arith::ConstantOp>(
              loc, DenseElementsAttr::get(
                       getVectorzedType(multiReductionOp, dimSize),
                       {initValueAttr}));

          DenseMap<Value, int> localAnchorArgsIdxMap;
          DenseMap<Value, Value> localOriginalOperandLoopArgsMap,
              localLoopArgsOriginalOperandMap;
          SmallVector<Value> argsArray;
          argsArray.emplace_back(accVal);
          localAnchorArgsIdxMap[accVal] = 0;
          size_t accLoopStateIdx = currentLoopStateIdxMap
              [originalOperandLoopArgsMap[multiReductionAcc]];
          localLoopArgsOriginalOperandMap[accVal] = multiReductionAcc;
          localOriginalOperandLoopArgsMap[multiReductionAcc] = accVal;

          for (auto [idx, x] : llvm::enumerate(loopState)) {
            if (idx == accLoopStateIdx) {
              continue;
            }
            argsArray.emplace_back(x);
            localAnchorArgsIdxMap[x] = argsArray.size() - 1;
            Value originalValue = loopArgsOriginalOperandMap[initArgs[idx]];
            localOriginalOperandLoopArgsMap[originalValue] = x;
            localLoopArgsOriginalOperandMap[x] = originalValue;
          }
          DenseMap<Value, Value> originalResultForResultMap;
          auto nxtFor = reductionAxisGenerateForLoop(
              b, groupIdx, 0, parallelIdx, localAnchorArgsIdxMap, argsArray,
              localOriginalOperandLoopArgsMap, localLoopArgsOriginalOperandMap,
              nextAnchorResults, nextAnchorResultsIdxMap, inductionVars,
              forResultOrignalResultMap, originalResultForResultMap);

          // insert accumulate value to original vector
          // TODO: fix first accumualte idx use map
          auto accRes = nxtFor->getResults()[0];

          Operation *reductionOp = b.create<vector::ReductionOp>(
              loc, multiReductionOp.getKind(), accRes);
          auto insertOp = b.create<vector::InsertOp>(
              loc, reductionOp->getResult(0), loopState[accLoopStateIdx], iv);

          // generate loop result
          SmallVector<Value> currentAnchorResults;
          DenseMap<Value, Value> currentResultMap;
          DenseMap<Value, int> currentResultIdxMap;

          currentAnchorResults.emplace_back(insertOp->getResults()[0]);
          // reduce axis for loop first result we has already processed above
          currentResultMap[insertOp->getResults()[0]] =
              multiReductionOp->getResults()[0];
          currentResultIdxMap[insertOp->getResults()[0]] = 0;

          for (auto [idx, x] : llvm::enumerate(nextAnchorResults)) {
            if (idx == 0) {
              continue;
            }
            Value originalResult = forResultOrignalResultMap[x];
            size_t forResultIdx = nextAnchorResultsIdxMap[x];
            currentAnchorResults.emplace_back(
                nxtFor->getResults()[forResultIdx]);
            currentResultIdxMap[nxtFor->getResults()[forResultIdx]] = idx;
            currentResultMap[nxtFor->getResults()[forResultIdx]] =
                originalResult;
          }
          nextAnchorResults.clear();
          nextAnchorResults = std::move(currentAnchorResults);
          forResultOrignalResultMap = std::move(currentResultMap);
          nextAnchorResultsIdxMap = std::move(currentResultIdxMap);
          maybeYieldValue(b, loc, nextAnchorResults);
        }
      });
}

scf::ForOp ForLoopGenerator::generateTransposeForLoopWithLastDim(
    OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
    const int tpSteps, const Location &loc, SmallVector<Value> &inductionVars,
    const ValueRange &iterArgs) {
  auto &tpCanonicalizer = getTransposeCanonicalizers()[grpIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getVector().getType();
  size_t rank = vtType.getRank();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  bool isTransposeDim = forDimIdx == tpCanonicalizer.getFirstTpIdx() or
                        forDimIdx == tpCanonicalizer.getSecondTpIdx();
  auto forSteps =
      makeIndexArithConstantOp(opBuilder, loc, isTransposeDim ? tpSteps : 1);
  auto numIter =
      makeIndexArithConstantOp(opBuilder, loc, vtType.getShape()[forDimIdx]);
  VectorType kernelType =
      VectorType::get({tpSteps, tpSteps}, vtType.getElementType());
  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (forDimIdx == rank - 1) {
          // transfer read from source tensor
          Value source = tpOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          vector::TransferWriteOp successorWriteOp;
          for (Operation *x : tpOp->getUsers()) {
            if (isa<vector::TransferWriteOp>(x)) {
              successorWriteOp = cast<vector::TransferWriteOp>(x);
            }
          }
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
              /*indices=*/inductionVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);
          SmallVector<int64_t> perm{1, 0};
          auto transposeOp = b.create<vector::TransposeOp>(
              loc, transferReadOp->getResults()[0], perm);
          SmallVector<Value> writeVars(inductionVars.begin(),
                                       inductionVars.end());
          writeVars[tpCanonicalizer.getSecondTpIdx()] =
              inductionVars[tpCanonicalizer.getFirstTpIdx()];
          writeVars[tpCanonicalizer.getFirstTpIdx()] =
              inductionVars[tpCanonicalizer.getSecondTpIdx()];
          auto writeOp = b.create<vector::TransferWriteOp>(
              loc, transposeOp->getResults()[0],
              successorWriteOp->getOperands()[1], writeVars, inBoundsVal);
          maybeYieldValue(b, loc, writeOp->getResults());
        } else {
          // outter loop
          auto nxtFor = generateTransposeForLoopWithLastDim(
              b, grpIdx, forDimIdx + 1, tpSteps, loc, inductionVars, loopState);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

scf::ForOp
ForLoopGenerator::generateMultiReductionForLoop(const size_t grpIdx) {
  auto &rdCanonicalizer = getMultiRdCanonicalizers()[grpIdx];
  auto multiReductionOp = rdCanonicalizer.getCandidateOps()[0];
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
  // move acc related operation to operation first
  std::queue<Operation *> rectifyQueue;
  DenseSet<Operation *> pushedSet;
  auto moveOperation = [&](std::queue<Operation *> &from,
                           std::queue<Operation *> &to) {
    while (!from.empty()) {
      auto cur = from.front();
      from.pop();
      if (pushedSet.contains(cur)) {
        continue;
      }
      to.push(cur);
      pushedSet.insert(cur);
    }
  };
  moveOperation(accRelatedOps, rectifyQueue);
  moveOperation(opQueue, rectifyQueue);
  opQueue = rectifyQueue;

  // get current loop init args
  SetVector<Value> &grpArgs = getGroupOpInitArgs()[grpIdx];
  SmallVector<Value> forLoopArgs(grpArgs.begin(), grpArgs.end());
  ValueRange initArgs(forLoopArgs);
  DenseMap<Value, int> currentLoopStateIdxMap;
  DenseMap<Value, int> nextAnchorResultsIdxMap;
  // map original operation operand with loop args
  DenseMap<Value, Value> originalOperandLoopArgsMap, loopArgsOriginalOperandMap,
      forResultOrignalResultMap;
  for (auto [idx, val] : llvm::enumerate(initArgs)) {
    currentLoopStateIdxMap[val] = idx;
    originalOperandLoopArgsMap[val] = val;
  }

  SmallVector<Value, 5> inductionVars;
  IRRewriter rewriter(func);
  OpBuilder opBuilder(rdCanonicalizer.getCandidateOps()[0]);
  SmallVector<Value, 4> nextAnchorResults;

  scf::ForOp forOp = parallelAxisGenerateForLoop(
      opBuilder, grpIdx, 0, currentLoopStateIdxMap, initArgs, nextAnchorResults,
      nextAnchorResultsIdxMap, inductionVars, originalOperandLoopArgsMap,
      loopArgsOriginalOperandMap, forResultOrignalResultMap);

  auto replaceIfFn = [&](OpOperand &use) {
    return use.getOwner()->getBlock() == forOp->getBlock();
  };
  for (auto x : nextAnchorResults) {
    auto originalResult = forResultOrignalResultMap[x];
    rewriter.replaceOpUsesWithIf(
        originalResult.getDefiningOp(),
        forOp->getResults()[nextAnchorResultsIdxMap[x]], replaceIfFn);
  }
  rewriter.replaceOp(getMultiRdCanonicalizers()[grpIdx].getCandidateOps()[0],
                     forOp);

  return forOp;
}

// generate simple data movement for loop
scf::ForOp ForLoopGenerator::generateScalarDataMovement(
    OpBuilder &opBuilder, const size_t grpIdx, const size_t forDimIdx,
    const Location &loc, SmallVector<Value> &inductionVars,
    const ValueRange &iterArgs, DenseMap<size_t, size_t> &tpAxisMap) {
  auto &tpCanonicalizer = getTransposeCanonicalizers()[grpIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getVector().getType();
  size_t rank = vtType.getRank();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  auto forSteps = makeIndexArithConstantOp(opBuilder, loc, 1);
  auto numIter =
      makeIndexArithConstantOp(opBuilder, loc, vtType.getShape()[forDimIdx]);
  VectorType kernelType = VectorType::get({1}, vtType.getElementType());
  // generate transpose for loop
  return opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (forDimIdx == rank - 1) {
          // transfer read from source tensor
          Value source = tpOp->getOperand(0);
          auto readSourceOp =
              cast<vector::TransferReadOp>(source.getDefiningOp());
          vector::TransferWriteOp successorWriteOp;
          for (Operation *x : tpOp->getUsers()) {
            if (isa<vector::TransferWriteOp>(x)) {
              successorWriteOp = cast<vector::TransferWriteOp>(x);
            }
          }
          auto padValue = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(vtType.getElementType()));
          SmallVector<bool> inBoundsVal(1, true);

          auto transferReadOp = b.create<vector::TransferReadOp>(
              loc,
              /*vectorType=*/kernelType,
              /*source=*/readSourceOp.getSource(),
              /*indices=*/inductionVars,
              /*padding=*/padValue,
              /*inBounds=*/inBoundsVal);
          SmallVector<Value> writeVars;
          size_t itrIdx = 0;
          while (itrIdx < rank) {
            writeVars.emplace_back(inductionVars[tpAxisMap[itrIdx]]);
            itrIdx++;
          }

          auto writeOp = b.create<vector::TransferWriteOp>(
              loc, transferReadOp->getResults()[0],
              successorWriteOp->getOperands()[1], writeVars, inBoundsVal);
          maybeYieldValue(b, loc, writeOp->getResults());
        } else {
          // outter loop
          auto nxtFor =
              generateScalarDataMovement(b, grpIdx, forDimIdx + 1, loc,
                                         inductionVars, loopState, tpAxisMap);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
}

/// generate transpose for loop
scf::ForOp ForLoopGenerator::generateTransposeForLoop(const size_t grpIdx) {

  // transpose rank must bigger than 2
  TransposeCanonicalizer &tpCanonicalizer =
      getTransposeCanonicalizers()[grpIdx];
  vector::TransposeOp &tpOp = tpCanonicalizer.getCandidateOps()[0];
  VectorType vtType = tpOp.getVector().getType();
  std::cout << " _________ check tp operation source."
            << "\n";
  vtType.dump();
  tpOp->getResultTypes()[0].dump();
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
  const int tpStep = 16;
  // currently we only support shape that is an integer multiple of tpStep
  if (vtType.getShape()[tpCanonicalizer.getFirstTpIdx()] % tpStep != 0 or
      vtType.getShape()[tpCanonicalizer.getSecondTpIdx()] % tpStep != 0) {
    isTwoDTranspose = false;
  }
  OpBuilder b(tpOp);
  SmallVector<Value> iterArgs;
  vector::TransferWriteOp successorWriteOp;
  for (Operation *x : tpOp->getUsers()) {
    if (isa<vector::TransferWriteOp>(x)) {
      successorWriteOp = cast<vector::TransferWriteOp>(x);
    }
  }
  iterArgs.emplace_back(successorWriteOp->getOperands()[1]);
  SmallVector<Value> inductionVars;
  IRRewriter rewriter(func);

  if (permuteSet.contains(rank - 1) and isTwoDTranspose) {
    std::cout << " can use 16x16 : " << std::endl;
    scf::ForOp forOp = generateTransposeForLoopWithLastDim(
        b, grpIdx, 0, tpStep, tpOp.getLoc(), inductionVars, iterArgs);

    for (Operation *x : tpOp->getUsers()) {
      if (isa<vector::TransferWriteOp>(x)) {
        rewriter.replaceOp(x, forOp);
      }
    }
    return forOp;
  }
  // findTransposeAxisMap(DenseMap<size_t, size_t> & tpAxisMap);
  DenseMap<size_t, size_t> tpAxisMap;
  size_t itrIdx = 0;
  while (itrIdx < rank) {
    tpAxisMap[itrIdx] = permutation[itrIdx];
    itrIdx++;
  }
  // scalar data movement
  scf::ForOp forOp = generateScalarDataMovement(
      b, grpIdx, 0, tpOp.getLoc(), inductionVars, iterArgs, tpAxisMap);
  for (Operation *x : tpOp->getUsers()) {
    if (isa<vector::TransferWriteOp>(x)) {
      rewriter.replaceOp(x, forOp);
    }
  }
  std::cout << " scalar data movement." << std::endl;
  forOp->dump();
  return forOp;
}

template <class T>
SmallVector<T, 4> &SpecialOperationCanonicalizer<T>::getCandidateOps() {
  return candidateRdOps;
};

void MultiReductionCanonicalizer::initReductionAxis() {
  auto reductionAxisRange =
      getCandidateOps()[0].getReductionDims().getAsValueRange<IntegerAttr>();
  auto reductionRange = llvm::to_vector<4>(map_range(
      reductionAxisRange, [](const APInt &a) { return a.getZExtValue(); }));
  reductionAxis.assign(reductionRange.begin(), reductionRange.end());
}

void MultiReductionCanonicalizer::initParallelAxis() {
  llvm::SmallDenseSet<int64_t, 4> reductionAxisSet(reductionAxis.begin(),
                                                   reductionAxis.end());
  for (int64_t i = 0; i < typeRank; ++i) {
    if (!reductionAxisSet.contains(i)) {
      parallelAxis.push_back(i);
    }
  }
  llvm::sort(parallelAxis.begin(), parallelAxis.end());
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
  if (reductionAxisSet.contains(typeRank - 1)) {
    res = true;
  }
  haslastDimReduction = res;
  return res;
}

void MultiReductionCanonicalizer::prepareSpecialOperationInfo() {
  if (getCandidateOps().empty()) {
    return;
  }
  sourceType = getCandidateOps()[0].getSourceVectorType();
  accType = mlir::dyn_cast<VectorType>(getCandidateOps()[0].getAcc().getType());
  getTypeRank();
  getReductionAxisAndParallelAxis();
  hasLastDimReduction();
};

void TransposeCanonicalizer::prepareSpecialOperationInfo() {
  if (getCandidateOps().empty()) {
    return;
  }
}

bool TransposeCanonicalizer::isTwoDTranspose() {
  ArrayRef<int64_t> permutation = getCandidateOps()[0].getPermutation();
  size_t rank = permutation.size();
  int diffCount = 0;
  // get the first transpose axis
  size_t itrIdx = 0;
  while (itrIdx < rank) {
    if ((int64_t)itrIdx != permutation[itrIdx]) {
      diffCount += 1;
    }
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
  return diffCount == 2;
}

template <class T> void addDummyInit(SmallVector<T, 8> &canonicalizer) {
  canonicalizer.emplace_back(T({}));
};

void CanonicalizerVectorOperation::clearSpecialOperationCanonicalizers() {
  getMultiRdCanonicalizers().clear();
  getBroadcastCanonicalizers().clear();
  getTransposeCanonicalizers().clear();
  getShapeCastCanonicalizers().clear();
}

void CanonicalizerVectorOperation::dummyInitSpecialOperation() {
  addDummyInit<MultiReductionCanonicalizer>(getMultiRdCanonicalizers());
  addDummyInit<BroadcastCanonicalizer>(getBroadcastCanonicalizers());
  addDummyInit<TransposeCanonicalizer>(getTransposeCanonicalizers());
  addDummyInit<ShapeCastCanonicalizer>(getShapeCastCanonicalizers());
}

void CanonicalizerVectorOperation::initSpeicalOperationCanonicalizers() {
  clearSpecialOperationCanonicalizers();
  SmallVector<std::queue<Operation *>, 8> &opGroups =
      getFusionStrategy().getOpGroups();
  for (auto &grp : opGroups) {
    dummyInitSpecialOperation();
    if (grp.empty()) {
      continue;
    }
    std::queue<Operation *> tempQ(grp);
    while (!tempQ.empty()) {
      auto op = tempQ.front();
      tempQ.pop();
      if (isa<vector::MultiDimReductionOp>(op)) {
        getMultiRdCanonicalizers().back().getCandidateOps().emplace_back(
            cast<vector::MultiDimReductionOp>(op));
        getMultiRdCanonicalizers().back().prepareSpecialOperationInfo();
      } else if (isa<vector::BroadcastOp>(op)) {
        getBroadcastCanonicalizers().back().getCandidateOps().emplace_back(
            cast<vector::BroadcastOp>(op));
      } else if (isa<vector::TransposeOp>(op)) {
        getTransposeCanonicalizers().back().getCandidateOps().emplace_back(
            cast<vector::TransposeOp>(op));
      } else if (isa<vector::ShapeCastOp>(op)) {
        getShapeCastCanonicalizers().back().getCandidateOps().emplace_back(
            cast<vector::ShapeCastOp>(op));
      }
    }
  }
}

void CanonicalizerVectorOperation::canonicalizeSpecialOperation() {
  // multireduction operation
  OpBuilder::InsertionGuard guard(rewriter);

  initSpeicalOperationCanonicalizers();
  // traverse all groups
  llvm::SmallVector<MultiReductionCanonicalizer, 8> &multiRdCanonicalizers =
      getMultiRdCanonicalizers();
  llvm::SmallVector<TransposeCanonicalizer, 8> &transposeCanonicalizers =
      getTransposeCanonicalizers();
  for (auto [groupId, rdCanonicalizer] :
       llvm::enumerate(multiRdCanonicalizers)) {
    SmallVector<vector::MultiDimReductionOp, 4> &rdOps =
        rdCanonicalizer.getCandidateOps();
    if (!rdOps.empty()) {
      // generate MultiReduction for loops
      (void)generateMultiReductionForLoop(groupId);
    }
    SmallVector<vector::TransposeOp, 4> &transposeOps =
        transposeCanonicalizers[groupId].getCandidateOps();
    if (!transposeOps.empty()) {
      (void)generateTransposeForLoop(groupId);
    }
  }
}

void CanonicalizerVectorOperation::run() {
  auto &fusionStrategy = getFusionStrategy();
  if (kind == CanonicalizerKind::OperationsGroup) {
    // 1. Analysis the operation's operands and results
    // We need to analyze which operation results are needed by other
    // operations, and we need to pass these results correctly. Mapping the
    // operation result value to forloop yeild result value. We can replace
    // the operation operand as: map(operand, forloop yield result) -> operand
    // = loop yield result We put all the operation result into this map.

    // 1.a. Find results which should be generated by current group for
    // using as operands to other operations?

    // Traverse all operations. If the operand of operations in other groups
    // or outside the group is the result of the current group operation, then
    // the current operation needs to generate a result. We use `setvector` to
    // save the results that need to be generated by the current group.

    //  1.b. What operands are needed to find in the current group, and where
    //  can they be obtained ?

    //  Thanks to 1.a, we get the result generated by the operations of
    //  each group, and this result will use `for loop yield` to generate a
    //  new result. Since the scope of the parent block of mlir is covered
    //  the current operation, the current operation does not need to pass
    //  these `for loop results` to the `iterArgs` of the required `for loop`.
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

    // Query groupResultYeildSet to map operaion result value to scf.yield
    // result value.
    analysisGroupOperaion();
    // printGroupOps(fusionStrategy.getOpGroups());
    // Speical Operation Canonicalization
    canonicalizeSpecialOperation();

    // 2.Generate vectorized IR for each operation group
    for (size_t idx = 0; idx < fusionStrategy.getOpGroups().size(); ++idx) {
      generateGroupOpVectorizedIR(idx);
    }

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

  if (mlir::isa<vector::TransferReadOp>(op) ||
      mlir::isa<vector::TransferWriteOp>(op)) {
    if (!isReadWriteOnLastDim(op)) {
      LDBG("Operation is not last dim read/write" << *op << "\n");
      return false;
    }
  }

  return true;
}

//
void setOperationCorrectOperand(
    Operation *op, const ValueRange &iterArgs,
    const DenseMap<Value, int> &operandIdxMap, ArrayRef<Value> inductionVars,
    const DenseMap<Operation *, AffineMap> &opPermuationMap) {
  for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
    if (operandIdxMap.contains(opd)) {
      op->setOperand(idx, iterArgs[operandIdxMap.at(opd)]);
    }
  }
  int offset = isa<vector::TransferWriteOp>(op) ? 2 : 1;
  if (dyn_cast<vector::TransferWriteOp>(op) ||
      dyn_cast<vector::TransferReadOp>(op)) {

    assert(opPermuationMap.contains(op));
    auto permutationMap = opPermuationMap.at(op);

    auto dimExpr = permutationMap.getResults();
    for (auto [idx, x] : llvm::enumerate(dimExpr)) {
      if (mlir::dyn_cast<AffineDimExpr>(x)) {
        auto dim = mlir::dyn_cast<AffineDimExpr>(x).getPosition();
        op->setOperand(dim + offset, inductionVars[dim]);
      }
    }
  }
}

scf::ForOp ForLoopGenerator::constructNestedForOp(
    const size_t forDimIdx, const size_t groupIdx, OpBuilder &b,
    const Location &loc, const ValueRange &iterArgs, VectorType type,
    const ArrayRef<int64_t> &dims, SmallVector<Value, 5> &inductionVars,
    const DenseMap<Value, int> &operandIdxMap) {
  const int loop_step = getDataTypeValidSteps(type);
  // loop initialization variable
  auto zero = makeIndexArithConstantOp(b, loc, 0);
  auto forSteps = makeIndexArithConstantOp(
      b, loc, forDimIdx == dims.size() - 1 ? loop_step : 1);
  auto numIter = makeIndexArithConstantOp(b, loc, dims[forDimIdx]);

  // Create a loop and move vectorized operation into loops.
  auto forOp = b.create<scf::ForOp>(
      b.getUnknownLoc(), zero, numIter, forSteps, iterArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        // inner most body of the loop
        if (forDimIdx == dims.size() - 1) {
          moveOperationsToCurrentForBody(
              groupIdx, b, inductionVars, operandIdxMap, loopState,
              getFusionStrategy().getOpGroups()[groupIdx]);
          llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &resultSet =
              getGroupOpResults()[groupIdx];
          SmallVector<Value> results(resultSet.size());
          size_t idx = 0;
          for (auto itr = resultSet.begin(); itr != resultSet.end(); itr++) {
            results[idx++] = itr->first;
          }
          maybeYieldValue(b, loc, results);
        } else {
          // outter loop
          auto nxtFor =
              constructNestedForOp(forDimIdx + 1, groupIdx, b, loc, loopState,
                                   type, dims, inductionVars, operandIdxMap);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });
  return forOp;
}

bool isSameVectorType(Operation *op1, Operation *op2) {
  auto type1 = getOperationVectorType(op1);
  auto type2 = getOperationVectorType(op2);
  if (failed(type1) || failed(type2)) {
    return false;
  }
  auto sp1 = type1.value();
  auto sp2 = type2.value();
  if (sp1.getRank() != sp2.getRank()) {
    return false;
  }
  bool isSame = true;
  // from front to back
  for (long i = 0; i < sp1.getRank(); i++) {
    if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
      isSame = false;
      break;
    }
  }
  return isSame;
}

bool VectorFusionStrategy::isCompatibleVectorType(Operation *op1,
                                                  Operation *op2) {
  auto type1 = getOperationVectorType(op1);
  auto type2 = getOperationVectorType(op2);
  if (failed(type1) || failed(type2)) {
    return false;
  }
  auto sp1 = type1.value();
  auto sp2 = type2.value();
  // if (isReadOrWriteOperation(op1) or isReadOrWriteOperation(op2)) {
  //   if (sp1.getRank() != sp2.getRank()) {
  //     return false;
  //   }
  //   for (long i = 0; i < sp1.getRank(); i++) {
  //     if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
  //       return false;
  //     }
  //   }
  // }
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
  if (bcAxis.empty()) {
    bcAxis.emplace_back(-1);
  }
}

void getOperationDataAxis(Operation *op, SmallVector<int64_t> &dataAxis) {
  return TypeSwitch<Operation *>(op)
      .Case<vector::MultiDimReductionOp>(
          [&](vector::MultiDimReductionOp multiReductionOp) {
            auto rdDimsRange = multiReductionOp.getReductionDims()
                                   .getAsValueRange<IntegerAttr>();
            auto reductionDims = llvm::to_vector<4>(map_range(
                rdDimsRange, [](const APInt &a) { return a.getZExtValue(); }));
            dataAxis.assign(reductionDims.begin(), reductionDims.end());
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
      })
      .Default([&](Operation *op) {
        // default is last axis
        dataAxis.emplace_back(
            mlir::dyn_cast<ShapedType>(op->getResultTypes().front()).getRank() -
            1);
      });
}

bool hasDataDependency(Operation *op1, Operation *op2) {
  if (!isSpecialOp(op1) and !isSpecialOp(op2)) {
    return false;
  }
  // op1 must be special operation
  if (!isSpecialOp(op1)) {
    return hasDataDependency(op2, op1);
  }
  // TODO: Remove this condition to support special operation fusion in the
  // future
  if (disableSpecialOp) {
    return true;
  }
  auto hasSameAxis = [](const SmallVector<int64_t> &dims1,
                        const SmallVector<int64_t> &dims2) {
    DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());
    for (auto x : dims1) {
      if (checkSet.contains(x)) {
        return true;
      }
    }
    return false;
  };
  auto res =
      TypeSwitch<Operation *, bool>(op1)
          .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp shapeCastOp) {
            SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            return hasSameAxis(dims1, dims2);
          })
          .Case<vector::MultiDimReductionOp>(
              [&](vector::MultiDimReductionOp multiReductionOp) {
                // has two cases: op1 is special operation, op2 is normal
                // operation op1 and op2 is both speicial operation
                SmallVector<int64_t> dims2, reductionDims, parallelDims;
                getOperationDataAxis(op1, reductionDims);
                getOperationDataAxis(op2, dims2);
                DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());
                auto op2VectorType = getOperationVectorType(op2);
                if (!isSpecialOp(op2)) {
                  if (isSameVectorType(op1, op2)) {
                    return false;
                  }
                  // all reduction axis should be op2's data axis
                  bool reduceDependent = false;
                  for (auto x : reductionDims) {
                    if (!checkSet.contains(x)) {
                      reduceDependent = true;
                      break;
                    }
                  }
                  if (!reduceDependent) {
                    return false;
                  }
                  // all parallel axis should equal to op2's axis
                  checkSet.clear();
                  checkSet.insert(reductionDims.begin(), reductionDims.end());
                  auto rdRank =
                      multiReductionOp.getSourceVectorType().getRank();
                  for (auto i = 0; i < rdRank; i++) {
                    if (!checkSet.contains(i)) {
                      parallelDims.emplace_back(i);
                    }
                  }
                  checkSet.clear();
                  checkSet.insert(parallelDims.begin(), parallelDims.end());
                  auto rank = op2VectorType->getRank();
                  for (auto i = 0; i < rank; i++) {
                    if (!checkSet.contains(i)) {
                      return true;
                    }
                  }

                  return false;
                }

                return true;
              })
          .Case<vector::BroadcastOp>([&](vector::BroadcastOp broadcastOp) {
            SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            return true;
            if (!isSpecialOp(op2)) {
              return hasSameAxis(dims1, dims2);
            } else {
            }
            return true;
          })
          .Case<vector::TransposeOp>([&](vector::TransposeOp transposeOp) {
            SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            return true;
            if (!isSpecialOp(op2)) {
              return hasSameAxis(dims1, dims2);
            } else {
            }
            return true;
          })
          .Default([&](Operation *op) { return false; });

  return res;
}

bool VectorFusionStrategy::isNeedNewGroup(Operation *op) {
  // 1. check previous operation
  if (!opGroups.back().empty()) {
    auto prevOp = opGroups.back().back();
    // not in the same operation
    if (prevOp->getParentOp() != op->getParentOp()) {
      return true;
    }
    // special operation need to check data dependency axis
    if (hasDataDependency(prevOp, op)) {
      return true;
    }

    // previous operation vector type is not compatible with current operation
    if (!isCompatibleVectorType(prevOp, op)) {
      return true;
    }
  }
  return false;
}

void VectorFusionStrategy::updateGroupBitgestVectorType(VectorType vectorType) {
  int64_t rank = vectorType.getRank();
  llvm::SmallDenseMap<size_t, VectorType> &groupVectorType =
      getGroupBiggestRankVectorType();

  if (groupVectorType.contains(opGroups.size() - 1)) {
    VectorType bigestType = groupVectorType[opGroups.size() - 1];
    if (bigestType.getRank() < rank) {
      groupVectorType[opGroups.size() - 1] = vectorType;
    }
    return;
  }

  groupVectorType[opGroups.size() - 1] = vectorType;
}

void VectorFusionStrategy::addOperationToGroup(Operation *op) {
  assert(op);
  VectorType vectorType = getOperationVectorType(op).value();
  if (isNeedNewGroup(op)) {
    opGroups.emplace_back(std::queue<Operation *>());
  } else {
    updateGroupBitgestVectorType(vectorType);
  }
  opGroups.back().push(op);
  opGroupIndexMap[op] = opGroups.size() - 1;
  opAnchorPos[op] = getOperationVectorType(op)->getRank() - 1;
}

// We classify the operations we are interested in after filtering. Operations
// of in the same group have no data dependencies. Those operations can generate
// a same outter for loop.
void VectorFusionStrategy::classifyOperations() {
  if (opGroups.empty()) {
    // dummpy
    opGroups.emplace_back(std::queue<Operation *>());
  }
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (filterOperation(op)) {
      addOperationToGroup(op);
    }
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

            if (mlir::isa<ElementsAttr>(value)) {
              auto valueType = mlir::dyn_cast<ElementsAttr>(value);
              if (valueType.isSplat()) {
                if (mlir::isa<FloatType>(valueType.getElementType())) {
                  initValueAttr = FloatAttr::get(
                      resultElementType,
                      valueType.getSplatValue<APFloat>().convertToDouble());
                } else {
                  initValueAttr = IntegerAttr::get(
                      resultElementType,
                      valueType.getSplatValue<APInt>().getSExtValue());
                }
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
    if (mlir::dyn_cast<VectorType>(x.getType())) {
      if (!opMap.contains(x.getDefiningOp())) {
        auto result = setOutGroupOperationOperandResult(x.getDefiningOp(),
                                                        newOperandType);
        op->setOperand(idx, result);
      } else {
        x.setType(newOperandType);
      }
    }
  }
  for (auto x : op->getResults()) {
    if (mlir::dyn_cast<VectorType>(x.getType())) {
      x.setType(newOperandType);
    }
  }
};

void ForLoopGenerator::createNewConstantOp(
    Operation *srcOp, vector::TransferWriteOp *transferWriteOp) {
  auto &opPermuationMap = getOpPermuationMap();
  IRRewriter srcWriter(srcOp);
  auto newOperandType = getVectorzedType(mlir::cast<Operation *>(srcOp));
  auto srcConstantOp = dyn_cast<arith::ConstantOp>(srcOp);
  Operation *newConstantOp;
  if (mlir::isa<ElementsAttr>(srcConstantOp.getValue())) {
    auto valueType = mlir::dyn_cast<ElementsAttr>(srcConstantOp.getValue());
    if (valueType.isSplat()) {
      FailureOr<Value> res = createArithSplatConstantOp(
          srcWriter, srcOp->getLoc(), valueType, newOperandType);
      if (failed(res)) {
        llvm::llvm_unreachable_internal("Wrong to create constant op.");
      }
      newConstantOp = res.value().getDefiningOp();
    } else {
      newConstantOp = srcWriter.create<arith::ConstantOp>(
          srcOp->getLoc(), srcConstantOp.getValue());
    }

    newConstantOp->getResult(0).setType(newOperandType);
    transferWriteOp->setOperand(0, newConstantOp->getResult(0));
    opPermuationMap.insert(
        {mlir::cast<Operation *>(srcOp), transferWriteOp->getPermutationMap()});
    setOpVectorizationPermutationMap(
        mlir::cast<Operation *>(srcOp), srcWriter,
        mlir::dyn_cast<RankedTensorType>(
            transferWriteOp->getResults()[0].getType()),
        transferWriteOp->getPermutationMap());
  }
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
                  if (mlir::isa<arith::ConstantOp>(srcOp)) {
                    createNewConstantOp(srcOp, &transferWriteOp);
                  } else {
                    opPermuationMap.insert(
                        {transferWriteOp, transferWriteOp.getPermutationMap()});
                    transferWriteOp->getOperand(0).setType(newOperandType);

                    setOpVectorizationPermutationMap(
                        transferWriteOp, rewriter,
                        mlir::dyn_cast<RankedTensorType>(
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
                      mlir::dyn_cast<RankedTensorType>(
                          transferReadOp.getSource().getType()),
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
            LDBG(" DPS operation : " << *op << "\n");
            return transferWriteOp->getOperand(1);
          })
      .Case<vector::TransferReadOp>([&](vector::TransferReadOp transferReadOp) {
        LDBG(" DPS operation : " << *op << "\n");
        return transferReadOp->getOperand(0);
      })
      .Default([&](Operation *op) {
        LDBG("Try to get not DPS operation inits: " << *op << "\n");
        return failure();
      });
}

void CanonicalizerCommonUsedData::updateOpOperandResultInGroups(
    size_t opGid, Operation *op, Value &init, const Value &result) {
  std::queue<Operation *> tmpOpQueue(getFusionStrategy().getOpGroups()[opGid]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    if (curOp == op) {
      if (!failed(getOperationVectorType(init.getDefiningOp()))) {
        newOpQueue.push(init.getDefiningOp());
        getFusionStrategy().getOpGroupIndexMap()[init.getDefiningOp()] = opGid;
        getFusionStrategy().getOpAnchorPos()[init.getDefiningOp()] =
            getFusionStrategy().getOpAnchorPos()[op];
      }

      newOpQueue.push(op);

      if (result && !failed(getOperationVectorType(result.getDefiningOp()))) {
        newOpQueue.push(result.getDefiningOp());
        getFusionStrategy().getOpGroupIndexMap()[result.getDefiningOp()] =
            opGid;
        getFusionStrategy().getOpAnchorPos()[result.getDefiningOp()] =
            getFusionStrategy().getOpGroupIndexMap()[op];
      }
    } else {
      newOpQueue.push(curOp);
    }
  }
  getFusionStrategy().getOpGroups()[opGid] = newOpQueue;
}

void VectorFusionStrategy::run() { classifyOperations(); }

void CanonicalizerCommonUsedData::generateEmptyTensorAndWrite(
    Operation *sourceOp,
    DenseMap<Operation *, std::pair<Value, Value>> &srcOpCanoniclizedMap,
    size_t anchorPos, ReturnTypeKind retKind) {
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getFusionStrategy().getOpGroupIndexMap();
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs = getGroupOpInitArgs();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOpResults();
  size_t sourceOpGid = opGroupIndexMap[sourceOp];

  auto [tsr, writeOpresult] = canonicalizeSourceOperation(sourceOp);
  auto writeOp = writeOpresult.getDefiningOp<vector::TransferWriteOp>();
  srcOpCanoniclizedMap.insert({sourceOp, {tsr, writeOpresult}});
  updateOpOperandResultInGroups(sourceOpGid, sourceOp, tsr, writeOpresult);
  groupOpInitArgs[sourceOpGid].insert(tsr);
  groupOpResults[sourceOpGid].insert({writeOpresult, {retKind, anchorPos}});
  // write opeartion anchor pos is same with current operation
  getFusionStrategy().getOpAnchorPos()[writeOp] =
      cast<vector::TransferWriteOp>(writeOp).getVectorType().getRank() - 1;
  getOpPermuationMap()[writeOp] = writeOp.getPermutationMap();
}

void VectorOperationAnalyzer::analysisEmptyGroupAndMaxSteps() {
  auto &groupOpResults = getGroupOpResults();
  auto &opGroups = getFusionStrategy().getOpGroups();

  // If the group operations do not have result need to be returned, these are
  // useless code.
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    if (groupOpResults[idx].empty()) {
      std::queue<Operation *>().swap(grp);
    }
    uint32_t steps = std::numeric_limits<uint32_t>::max();

    auto &grpSteps = getFusionStrategy().getGroupMaxSteps();
    while (idx >= grpSteps.size()) {
      grpSteps.emplace_back(steps);
    }
    std::queue<Operation *> tmpQueue(grp);
    auto calculateOpSteps = [&](Type type) {
      auto opType = mlir::dyn_cast<VectorType>(type);
      if (opType)
        steps = std::min(steps, (uint32_t)getDataTypeMAXSIMDLength(opType));
    };
    while (!tmpQueue.empty()) {
      auto op = tmpQueue.front();
      tmpQueue.pop();
      if (mlir::isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp>(op)) {
        calculateOpSteps(op->getOperandTypes()[0]);
      }
      calculateOpSteps(op->getResultTypes()[0]);
    }
    grpSteps[idx] = steps;
  }
}

void VectorOperationAnalyzer::specialOperationAnchorRectify() {
  auto &opGroups = getFusionStrategy().getOpGroups();
  for (auto [idx, grp] : llvm::enumerate(opGroups)) {
    std::queue<Operation *> tmpQueue(grp);
    while (!tmpQueue.empty()) {
      auto op = tmpQueue.front();
      tmpQueue.pop();
      if (isa<vector::MultiDimReductionOp>(op)) {
        auto accSourceOp = op->getOperand(1).getDefiningOp();
        getFusionStrategy().getOpAnchorPos()[accSourceOp] =
            getOperationVectorType(accSourceOp)->getRank() - 1;
      }
    }
  }
}

// analysis operation result of current group whether needed by other
// operation which out of current group
void VectorOperationAnalyzer::analysisGroupOperationResults() {
  DenseMap<Operation *, std::pair<Value, Value>> srcOpCanoniclizedMap;
  DenseSet<Operation *> movedOperationSet;
  DenseMap<Operation *, size_t> &opGroupIndexMap =
      getFusionStrategy().getOpGroupIndexMap();
  SmallVector<SetVector<Value>, 8> &groupOpInitArgs = getGroupOpInitArgs();
  SmallVector<llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>>, 8>
      &groupOpResults = getGroupOpResults();
  DenseMap<Operation *, size_t> &OpAnchorPos =
      getFusionStrategy().getOpAnchorPos();

  auto updateReturnResultKind = [&](Operation *sourceOp, size_t sourceOpGid,
                                    ReturnTypeKind rtKind) {
    Value sourceResult;
    if (srcOpCanoniclizedMap.contains(sourceOp)) {
      sourceResult = srcOpCanoniclizedMap[sourceOp].second;
    } else {
      sourceResult = sourceOp->getResults()[0];
    }
    size_t srcOpAnchor = groupOpResults[sourceOpGid][sourceResult].second;
    ReturnTypeKind prevRtKind = groupOpResults[sourceOpGid][sourceResult].first;
    srcOpAnchor = std::min(srcOpAnchor, OpAnchorPos[sourceOp]);
    if (prevRtKind != rtKind) {
      groupOpResults[sourceOpGid][sourceResult] =
          std::make_pair(ReturnTypeKind::RT_Both, srcOpAnchor);
    } else if (rtKind == ReturnTypeKind::RT_InGroup) {
      groupOpResults[sourceOpGid][sourceResult] =
          std::make_pair(rtKind, srcOpAnchor);
    }
  };

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
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
        ReturnTypeKind rtKind = inSameGroupNeedReturn
                                    ? ReturnTypeKind::RT_InGroup
                                    : ReturnTypeKind::RT_OutGroup;

        if (notInSameGroup or outOfGroup or inSameGroupNeedReturn) {
          // update init iterargs
          auto dstRet = getOperationOperateTensor(sourceOp);
          // need to generate tensor.emtpy and vector.transfer_write, write
          // operand to tensor and read operand from the tensor, generate
          // vector.transfer_read
          if (failed(dstRet)) {
            // already generate result tensor, special operation do the
            // transformation by itself
            if (isSpecialOp(sourceOp) and inSameGroupNeedReturn) {
              continue;
            }
            if (!srcOpCanoniclizedMap.contains(sourceOp)) {
              generateEmptyTensorAndWrite(sourceOp, srcOpCanoniclizedMap,
                                          OpAnchorPos[sourceOp], rtKind);
            } else {
              // udpate result return type
              updateReturnResultKind(sourceOp, sourceOpGid, rtKind);
            }

            auto opInit = canonicalizeCurrentOperation(
                op, srcOpCanoniclizedMap[sourceOp].second, idx);
            updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);

          } else {
            // if source operation is transfer_read, we need to generate a
            // same transfer_read operation like source operation.
            if (mlir::isa<vector::TransferReadOp>(sourceOp)) {
              auto transferReadOp = cast<vector::TransferReadOp>(sourceOp);
              auto opInit = canonicalizeCurrentOperation(op, dstRet.value(),
                                                         idx, &transferReadOp);
              updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);

            } else {
              groupOpInitArgs[sourceOpGid].insert(dstRet.value());
              updateReturnResultKind(sourceOp, sourceOpGid, rtKind);
            }
          }
        }
      } else if (isa_and_nonnull<arith::ConstantOp>(sourceOp)) {
        auto constantOp = cast<arith::ConstantOp>(sourceOp);
        IRRewriter rewriter(constantOp);
        if (mlir::isa<ElementsAttr>(constantOp.getValue())) {
          if (!srcOpCanoniclizedMap.contains(sourceOp)) {
            auto [tsr, writeOpresult] = canonicalizeSourceOperation(sourceOp);
            srcOpCanoniclizedMap.insert({sourceOp, {tsr, writeOpresult}});
          }
          auto opInit = canonicalizeCurrentOperation(
              op, srcOpCanoniclizedMap[sourceOp].second, idx);
          updateOpOperandResultInGroups(opGroupIndexMap[op], op, opInit);
        }
      }
    }
    if (mlir::isa<tensor::EmptyOp>(op) && !movedOperationSet.contains(op)) {
      auto parentBlock = op->getBlock();
      op->moveBefore(parentBlock, parentBlock->getOperations().begin());
      movedOperationSet.insert(op);
    }
  });
  analysisEmptyGroupAndMaxSteps();
  specialOperationAnchorRectify();
#undef RESULT_RETURN_TYPE
  LDBG("Complete analysis group operation results\n");
}

void VectorOperationAnalyzer::analysisGroupOperaion() {
  // Results
  analysisGroupOperationResults();
}

mlir::FailureOr<scf::ForOp> ForLoopGenerator::generateVectorizedForLoop(
    const size_t groupId, IRRewriter &rewriter, VectorType vectorType) {
  auto &resultSet = getGroupOpResults();
  auto &initArgs = getGroupOpInitArgs()[groupId];
  assert(!resultSet.empty() && "Expected non-empty value");
  // prepare for loop iterargs
  SmallVector<Value, 4> operands;
  DenseMap<Value, int> operandIdxMap;
  for (auto [idx, x] : llvm::enumerate(initArgs)) {
    operands.emplace_back(x);
    operandIdxMap[x] = operands.size() - 1;
  }
  ValueRange forIterArgs(operands);
  auto shapes = vectorType.getShape();
  SmallVector<Value, 5> inductionVars;
  // generate for loop
  auto forOp = constructNestedForOp(
      0, groupId, rewriter, rewriter.getUnknownLoc(), forIterArgs, vectorType,
      shapes, inductionVars, operandIdxMap);
  return forOp;
}

void updateLoopResultUses(
    llvm::MapVector<Value, std::pair<ReturnTypeKind, size_t>> &opResults,
    scf::ForOp *forOp) {
  if (opResults.empty()) {
    return;
  }
  IRRewriter rewriter(*forOp);
  OpBuilder::InsertionGuard g(rewriter);
  // Only different group operation operand need to be replaced due to same
  // group operation should directly use original operand.

  Operation *producerOp = opResults.begin()->first.getDefiningOp();
  auto needToReplaced = [&](OpOperand &operand) {
    return producerOp->getBlock() != operand.getOwner()->getBlock();
  };
  // update loop result uses
  for (auto [retIdx, rt] : llvm::enumerate(opResults)) {
    rewriter.replaceUsesWithIf(rt.first, forOp->getResult(retIdx),
                               needToReplaced);
  }
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
  auto &groupOpResults = getGroupOpResults();
  VectorType groupType =
      getFusionStrategy().getGroupBiggestRankVectorType()[idx];
  IRRewriter rewriter(grp.back());
  rewriter.setInsertionPointAfter(grp.back());
  // 1. Rewrite operation as vectorized form
  // 2. Generate loop
  rewriteOperationAsVectorize(rewriter, idx);
  auto forOp = generateVectorizedForLoop(idx, rewriter, groupType);
  // special operation do not need to change anything
  if (failed(forOp)) {
    return;
  }
  // 3 Update loop result uses
  updateLoopResultUses(groupOpResults[idx], &forOp.value());
  moveLoopInvariantCode(forOp.value());
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
    // canonicalize vector operation, default use vector-based fusion
    // strategy.
    HardWareInfo hwInfo;
    // default has avx512f instructions
    // hwInfo.favx512f = false;
    CanonicalizerVectorOperation canonicalizer(
        func, CanonicalizerKind::OperationsGroup, hwInfo);
    canonicalizer.run();

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