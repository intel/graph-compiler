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
// TODO: remove it in the future
bool disableSpecialOp = true;

struct HardwareInfo {
  bool favx512f = true;
  bool favx2 = true;
} HW;

// void printGroupOps(llvm::SmallVector<std::queue<Operation *>, 8> &opGroups) {
//   for (auto grp : opGroups) {
//     if (grp.empty()) {
//       continue;
//     }
//     std::cout << "__________________ group start_____________" << std::endl;
//     std::queue<Operation *> tmpQ(grp);
//     while (!tmpQ.empty()) {
//       auto cur = tmpQ.front();
//       tmpQ.pop();
//       cur->dump();
//     }
//     std::cout << "__________________ group end_____________" << std::endl;
//   }
// }

bool isSpecialOp(Operation *op) {
  return llvm::isa<vector::TransposeOp>(op) ||
         llvm::isa<vector::BroadcastOp>(op) ||
         llvm::isa<vector::ReductionOp>(op) ||
         llvm::isa<vector::ShapeCastOp>(op) ||
         llvm::isa<vector::MultiDimReductionOp>(op) ||
         llvm::isa<func::CallOp>(op);
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
  return llvm::isa<vector::MaskOp>(op) ||
         llvm::isa<vector::ConstantMaskOp>(op) ||
         llvm::isa<vector::MaskedLoadOp>(op) ||
         llvm::isa<vector::MaskedStoreOp>(op) ||
         llvm::isa<vector::CreateMaskOp>(op);
}

bool isReadOrWriteOperation(Operation *op) {
  return llvm::isa<vector::TransferReadOp>(op) ||
         llvm::isa<vector::TransferWriteOp>(op);
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
int get_nearest_vector_step(const int step) {
  assert(step > 0);
  int nbits = 0, n = step;
  while (n) {
    n = n >> 1;
    nbits++;
  }
  assert(nbits <= 6 || (nbits == 7 && step == 64));
  return (1 << (nbits - 1)) == step ? step : (1 << nbits);
}

int generateValidSteps(int steps, VectorType type) {
  return type.getShape().back() >= steps
             ? (steps > 16 ? 16 : steps)
             : get_nearest_vector_step(type.getShape().back());
}

// expr equals `vector rank` - 1
bool isLastDim(const AffineExpr &expr, const size_t rank) {
  return mlir::isa<AffineDimExpr>(expr) &&
         mlir::dyn_cast<AffineDimExpr>(expr).getPosition() == rank - 1;
}

[[nodiscard]] int getDataTypeValidSteps(const VectorType &type) {
  auto typebits = type.getElementTypeBitWidth();
  const int favx512bits = 512;
  const int favx2bits = 256;
  if (HW.favx512f) {
    return generateValidSteps(favx512bits / typebits, type);
  } else if (HW.favx2) {
    return generateValidSteps(favx2bits / typebits, type);
  } else {
    // invalid
    LDBG("Please check the hardware information.");
    assert(false && "Invalid hardware.");
    return -1;
  }
}

// Get the maximum number of current data types that a register can hold
[[nodiscard]] int getDataTypeMAXSIMDLength(const VectorType &type) {
  auto typebits = type.getElementTypeBitWidth();
  const int favx512bits = 512;
  const int favx2bits = 256;
  if (HW.favx512f) {
    // return generateValidSteps(favx512bits / typebits, type);
    return favx512bits / typebits;
  } else if (HW.favx2) {
    // return generateValidSteps(favx2bits / typebits, type);
    return favx2bits / typebits;
  } else {
    // invalid
    LDBG("Please check the hardware information.");
    assert(false && "Invalid hardware.");
    return -1;
  }
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
              [&](arith::ConstantOp constantOp) { return failure(); })
          .Default([&](Operation *op) -> mlir::FailureOr<VectorType> {
            if (!op->getResults().empty()) {
              auto t = mlir::dyn_cast<VectorType>(op->getResultTypes().front());
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

VectorType getVectorzedType(Operation *op, uint32_t loop_step = 0) {
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

// Constructs the 16 bit representation for a half precision value from a float
// value. This implementation is adapted from Eigen.
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

// Converts the 16 bit representation of a half precision value to a float
// value. This implementation is adapted from Eigen.
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
    else if (llvm::isa<FloatType>(t1))
      result = 0.0f;
    else
      llvm_unreachable("invalid value types for ADD reduction");
    break;
  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
    assert(llvm::isa<FloatType>(t1) && "expected float values");
    result = std::get<T>(numeric_limits_minimum(t));
    break;
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    assert(llvm::isa<FloatType>(t1) && "expected float values");
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
    else if (llvm::isa<FloatType>(t1))
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
void maybeYieldValue(OpBuilder &b, Location loc, ValueRange value) {
  bool hasRetVal = !value.empty();
  if (hasRetVal) {
    b.create<scf::YieldOp>(loc, value);
  } else {
    b.create<scf::YieldOp>(loc);
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

Operation *
createTransferReadOpBefore(Operation *op, const Value &operand,
                           vector::TransferReadOp *srcReadOp = nullptr) {
  auto operandType = mlir::dyn_cast<ShapedType>(operand.getType());

  IRRewriter rewriter(op);
  auto zero =
      rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
  auto padValue = rewriter.create<arith::ConstantOp>(
      rewriter.getUnknownLoc(),
      rewriter.getZeroAttr(operandType.getElementType()));

  if (srcReadOp) {
    auto resultType = mlir::dyn_cast<ShapedType>(srcReadOp->getType());
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
    return t;
  }
}

// canonicalizing operation as tensor empty and transfer write the operation
// result into the empty tensor
[[nodiscard]] std::pair<Value, Value>
canonicalizeSourceOperation(Operation *op) {
  // auto emtpyOp = createTensorEmptyBefore(op);
  auto resultTensor = getOperationResultTensor(op);
  auto writeOp = createTransferWriteOpAfter(op, resultTensor);
  return std::make_pair(resultTensor, writeOp->getResults()[0]);
}

[[nodiscard]] Value
canonicalizeCurrentOperation(Operation *op, const Value &transferReadOperand,
                             size_t operandIdx,
                             vector::TransferReadOp *srcReadOp = nullptr) {
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

void getOpSourceOps(Operation *op, llvm::DenseSet<Operation *> &srcOps) {
  llvm::SmallVector<Value> srcOperands = op->getOperands();
  std::deque<Value> srcOperandsQueue(srcOperands.begin(), srcOperands.end());
  llvm::DenseSet<Operation *> visited;
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

bool isSrcRelated(const llvm::DenseSet<Operation *> &srcOps, Operation *op) {
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
  llvm::DenseSet<Operation *> srcOps;
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

enum class MultiReduceOpAxisKind { Reduction, Parallel };
void updateReduceReadWriteOperationOperand(
    const llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::SmallVector<int64_t, 4> &parallelAxis, Operation *op,
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

vector::TransferReadOp cloneReductionTransferRead(
    Value &source, OpBuilder &b, IRMapping &readMap,
    const llvm::SmallVector<int64_t, 4> &parallelAxis,
    llvm::SmallVector<Value, 5> &inductionVars, bool lastDimReduction,
    MultiReduceOpAxisKind rdKind = MultiReduceOpAxisKind::Parallel) {
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
                       const llvm::SmallVector<int64_t, 4> &parallelAxis,
                       llvm::SmallVector<Value, 5> &inductionVars) {
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
    const size_t groupIdx, OpBuilder &b,
    const llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Value, int> &operandIdxMap,
    const ValueRange &loopState, const std::queue<Operation *> &queue) {
  std::queue<Operation *> opQueue =
      queue.empty() ? getFusionStrategy().getOpGroups()[groupIdx] : queue;
  auto &opPermuationMap = getOpPermuationMap();
  while (!opQueue.empty()) {
    auto x = opQueue.front();
    opQueue.pop();
    x->moveBefore(b.getBlock(), b.getBlock()->end());
    // check operation type to set correct operand
    setOperationCorrectOperand(x, loopState, operandIdxMap, inductionVars,
                               opPermuationMap);
  }
}

scf::ForOp ForLoopGenerator::reductionAxisGenerateForLoop(
    OpBuilder &opBuilder, const int groupIdx, const size_t reductionIdx,
    ValueRange &initArgs, llvm::SmallVector<Value, 5> &inductionVars) {
  MultiReductionCanonicalizer rdCanonicalizer =
      getMultiRdCanonicalizers()[groupIdx];
  auto &multireductionOp = rdCanonicalizer.getCandidateOps()[0];
  const auto loc = multireductionOp->getLoc();
  auto &reductionAxis = rdCanonicalizer.getReductionAxis();
  auto lastDimReduction = rdCanonicalizer.hasLastDimReduction();
  auto vectorType = rdCanonicalizer.getSourceType();
  const int loopStep = getDataTypeValidSteps(vectorType);
  auto isStandaloneOp = rdCanonicalizer.getIsStandAloneOp();

  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  auto forSteps = makeIndexArithConstantOp(
      opBuilder, loc,
      (reductionIdx == reductionAxis.size() - 1 && lastDimReduction) ? loopStep
                                                                     : 1);
  auto numIter = makeIndexArithConstantOp(
      opBuilder, loc, vectorType.getShape()[reductionAxis[reductionIdx]]);
  auto forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, initArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);

        if (reductionIdx == reductionAxis.size() - 1) {

          if (isStandaloneOp) {
            IRRewriter rewriter(b);
            IRMapping readMap;
            Value reductionTarget = multireductionOp.getSource();
            llvm::SmallVector<int64_t, 4> parallelAxis;
            auto newReadOp = cloneReductionTransferRead(
                reductionTarget, b, readMap, parallelAxis, inductionVars,
                lastDimReduction, MultiReduceOpAxisKind::Reduction);
            auto reductionResult =
                makeArithReduction(b, loc, multireductionOp.getKind(),
                                   newReadOp->getResult(0), loopState.back());
            maybeYieldValue(b, loc, reductionResult);
          } else {
            auto &analysisResults = getGroupOpResults()[groupIdx];

            auto &sourceOps =
                getMultiRdCanonicalizers()[groupIdx].getSourceRelatedOps();
            auto &grpArgs = getGroupOpIterArgs()[groupIdx];

            rewriteOperationAsVectorize(b, groupIdx, sourceOps);
            llvm::DenseMap<Value, int> operandIdxMap;
            llvm::SmallVector<Value> resultArray;
            // dummy
            resultArray.emplace_back(Value());
            std::queue<Operation *> tmpSourceOps(sourceOps);
            // move operation into current for loop body
            // accVal is first loopstate
            int start = 1;
            while (!tmpSourceOps.empty()) {
              auto cur = tmpSourceOps.front();
              tmpSourceOps.pop();
              auto curOperands = cur->getOperands();
              for (auto x : curOperands) {
                if (grpArgs.contains(x)) {
                  operandIdxMap[x] = start++;
                }
              }
              if (analysisResults.contains(cur->getResults()[0])) {
                resultArray.emplace_back(cur->getResults()[0]);
                getMultiRdCanonicalizers()[groupIdx]
                    .getOriginalOpResults()
                    .insert(cur->getResults()[0]);
                getMultiRdCanonicalizers()[groupIdx].getResultIdxMap().insert(
                    {cur->getResults()[0], resultArray.size() - 1});
              }
            }
            moveOperationsToCurrentForBody(groupIdx, b, loopState,
                                           operandIdxMap, inductionVars);

            auto reductionResult = makeArithReduction(
                b, loc, multireductionOp.getKind(),
                multireductionOp.getSource(), loopState.back());
            resultArray[0] = reductionResult;

            maybeYieldValue(b, loc, resultArray);
          }
        } else {
          // outter loop
          auto nxtFor = reductionAxisGenerateForLoop(
              b, groupIdx, reductionIdx + 1, loopState, inductionVars);
          maybeYieldValue(b, loc, nxtFor->getResults());
        }
      });

  return forOp;
}

scf::ForOp ForLoopGenerator::parallelAxisGenerateForLoop(
    OpBuilder &opBuilder, const int groupIdx, const size_t parallelIdx,
    ValueRange &initArgs, llvm::SmallVector<Value, 5> &inductionVars,
    Value &originalWriteResult) {
  auto &rdCanonicalizer = getMultiRdCanonicalizers()[groupIdx];
  auto &multiReductionOp = rdCanonicalizer.getCandidateOps()[0];
  auto &vectorType = rdCanonicalizer.getSourceType();
  auto &accType = rdCanonicalizer.getAccType();
  IRRewriter rewriterOfFunc(func);

  auto &parallelAxis = rdCanonicalizer.getParallelAxis();
  auto isStandaloneOp = rdCanonicalizer.getIsStandAloneOp();
  auto lastDimReduction = rdCanonicalizer.hasLastDimReduction();
  const auto &loc = multiReductionOp.getLoc();
  auto zero = makeIndexArithConstantOp(opBuilder, loc, 0);
  auto forSteps = makeIndexArithConstantOp(opBuilder, loc, 1);

  // last dim reduction need to a generate dim=16 loop
  int dimSize = 0;
  if (parallelIdx == parallelAxis.size()) {
    dimSize = 16;
  } else {
    dimSize = vectorType.getShape()[parallelAxis[parallelIdx]];
  }
  auto numIter = makeIndexArithConstantOp(opBuilder, loc, dimSize);
  // Create a loop and move vectorized operation into loops.
  auto forOp = opBuilder.create<scf::ForOp>(
      loc, zero, numIter, forSteps, initArgs,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
        inductionVars.emplace_back(iv);
        auto &fusionStrategy = getFusionStrategy();
        auto &opIndexMap = fusionStrategy.getOpGroupIndexMap();

        assert(opIndexMap.contains(multiReductionOp) &&
               " Must constains multireduction operation.");

        auto opIndex = opIndexMap[multiReductionOp];
        auto &opGroups = fusionStrategy.getOpGroups();
        auto opQueue = opGroups[opIndex];
        auto multiReductionAcc = multiReductionOp.getAcc();

        if (parallelIdx == parallelAxis.size() - 1) {
          // four kinds of group operations
          // If fused a operation, it means  multirection must just
          // constains last dim to do the reduction.
          // 1. just multireduction
          //    two cases:
          //    1. constaints last dims
          //     for ... parallel axis:
          //         transfer_read from accSource tensor
          //         arith.constant : vector<16xf32>
          //         for ... 16:
          //             for ... reduction axis:
          //                add
          //             scalar = reduction vector
          //             scalar insert into accVector
          //         transfer_write accVector into emtpy tensor
          //    2. not last dims
          //      for ... generate axis:
          //             transfer_read from accSource tensor
          //             transfer_read from source tensor
          //             accVector = add
          //             transfer_write accVector into emtpy tensor
          // 2. prev-op + multireduction
          //     In this case, there will be no tensor.empty + transfer_read
          //     operation, but the multireduction should write in an empty
          //     tensor
          //     for ... parallel axis:
          //        accVector and related accVector operation should be here
          //        extract from accVector scalar
          //        airth.constant : vector<16xf32>
          //          for ... reduction axis:
          //             prevop source op
          //             add
          //        scalar = reduction vector
          //        scalar insert into accVector
          //        transfer_write accVector into empty tensor
          //
          // 3. post-op + multireduction
          //     for ... parallel axis:
          //         transferread from accSource tensor
          //         arith.constant : vector<16xf32>
          //         for ... reduction axis:
          //             add
          //             postOp
          //             post Op transferWrite emtpy tensor
          //         scalar = reduction vector
          //         scalar insert into accVector
          //         transfer_write accVector into emtpy tensor
          // 4. prev-op + multireduction + post-op
          //     for ... parallel axis:
          //         accVector operation
          //         extract from accVector a scalar
          //         arith.constant : vector<16xf32>
          //         for ... reduction axis:
          //             prev-op source op and related source operation
          //             add
          //             postOp
          //             post Op transferWrite emtpy tensor
          //         scalar = reduction vector
          //         scalar insert into accVector
          //         transfer_write accVector into emtpy tensor

          if (isStandaloneOp) {
            // read operation
            IRMapping accReadMap;
            auto accReadOp = multiReductionAcc.getDefiningOp();
            assert(mlir::isa<vector::TransferReadOp>(accReadOp));
            accReadMap.map(accReadOp->getOperand(0), loopState.back());

            auto newAccReadOp = cloneReductionTransferRead(
                multiReductionAcc, b, accReadMap, parallelAxis, inductionVars,
                lastDimReduction, MultiReduceOpAxisKind::Parallel);
            // constructe next for loop
            Attribute initValueAttr;
            getReductionInitAttr(multiReductionOp, initValueAttr);

            auto accVal = b.create<arith::ConstantOp>(
                loc, DenseElementsAttr::get(accType, {initValueAttr}));

            ValueRange newIterArgs(accVal);
            auto nxtFor = reductionAxisGenerateForLoop(
                b, groupIdx, 0, newIterArgs, inductionVars);

            // insert accumulate value to original vector
            auto accRes = nxtFor->getResults()[0];

            Operation *reductionOp = b.create<vector::ReductionOp>(
                loc, multiReductionOp.getKind(), accRes);
            auto insertOp =
                b.create<vector::InsertOp>(loc, reductionOp->getResult(0),
                                           newAccReadOp->getResults()[0], 0);

            // write vector back to tensor
            vector::TransferWriteOp accWriteOp = nullptr;
            for (auto [idx, x] : llvm::enumerate(
                     multiReductionOp->getResults()[0].getUsers())) {
              if (idx == 0 && mlir::isa<vector::TransferWriteOp>(x)) {
                accWriteOp = mlir::dyn_cast<vector::TransferWriteOp>(x);
                break;
              }
            }
            assert(accWriteOp &&
                   " Not transfer_write operation. Current multireduction "
                   "operation may have wrong analysis IR.");
            IRMapping accWriteindiceMap;
            accWriteindiceMap.map(accWriteOp.getOperand(0),
                                  insertOp->getResults()[0]);
            auto writeResult = accWriteOp->getResults()[0];
            auto newAccWriteOp = makeNewTransferWriteOp(
                writeResult, accWriteindiceMap, b, parallelAxis, inductionVars);
            originalWriteResult = newAccWriteOp->getResult(0);

            maybeYieldValue(b, loc, newAccWriteOp->getResults());
          } else {
            auto prevOp = opQueue.front();
            auto postOp = opQueue.back();
            auto &prevOps = getMultiRdCanonicalizers()[groupIdx].getPrevOps();
            auto &postOps = getMultiRdCanonicalizers()[groupIdx].getPostOps();
            auto &accRelatedOps =
                getMultiRdCanonicalizers()[groupIdx].getAccRelatedOps();
            auto &sourceRelatedOps =
                getMultiRdCanonicalizers()[groupIdx].getSourceRelatedOps();

            if (mlir::isa<vector::MultiDimReductionOp>(prevOp)) {

            } else {
              if (mlir::isa<vector::MultiDimReductionOp>(postOp)) {
                // prevOp + reduction op
              } else {
                // prevOp + reduction op + postOp
                // reduction op + postOp
                getPrevOps(prevOps, opQueue, multiReductionOp);
                getPostOps(postOps, opQueue, multiReductionOp);
                // analysis acc related operation
                classifySourceRelatedOps(
                    accRelatedOps, sourceRelatedOps,
                    multiReductionOp.getSource().getDefiningOp(), prevOps);

                rewriteOperationAsVectorize(b, groupIdx, accRelatedOps);
                auto &grpArgs = getGroupOpIterArgs()[groupIdx];
                llvm::DenseMap<Value, int> operandIdxMap;
                for (auto [idx, x] : llvm::enumerate(grpArgs)) {
                  operandIdxMap[x] = idx;
                }
                moveOperationsToCurrentForBody(groupIdx, b, inductionVars,
                                               operandIdxMap, loopState,
                                               accRelatedOps);
                auto &grpResults = getGroupOpResults()[groupIdx];
                // next for loop
                llvm::SmallVector<Value> iterArgsArray;
                iterArgsArray.emplace_back(multiReductionAcc);
                std::queue<Operation *> tmpSourceOps(sourceRelatedOps);
                while (!tmpSourceOps.empty()) {
                  auto cur = tmpSourceOps.front();
                  tmpSourceOps.pop();
                  auto curResults = cur->getResults();
                  for (auto x : curResults) {
                    if (grpResults.contains(x)) {
                      for (auto y : cur->getOperands()) {
                        if (grpArgs.contains(y)) {
                          iterArgsArray.emplace_back(y);
                        }
                      }
                    }
                  }
                }
                ValueRange reductionAxisArgs(iterArgsArray);
                auto nxtFor = parallelAxisGenerateForLoop(
                    b, groupIdx, parallelIdx + 1, reductionAxisArgs,
                    inductionVars, originalWriteResult);

                rewriteOperationAsVectorize(b, groupIdx, postOps);
                moveOperationsToCurrentForBody(groupIdx, b, loopState,
                                               operandIdxMap, inductionVars,
                                               postOps);

                auto replaceIfFn = [&](OpOperand &use) {
                  return use.getOwner()->getBlock() == nxtFor->getBlock();
                };
                rewriterOfFunc.replaceOpUsesWithIf(
                    multiReductionOp, nxtFor->getResults()[0], replaceIfFn);
                auto &originalResults =
                    getMultiRdCanonicalizers()[groupIdx].getOriginalOpResults();
                for (auto [idx, x] : llvm::enumerate(originalResults)) {
                  rewriterOfFunc.replaceOpUsesWithIf(
                      x.getDefiningOp(), nxtFor->getResults()[idx + 1],
                      replaceIfFn);
                }
                llvm::SmallVector<Value> resultsArray;
                llvm::SmallDenseMap<Value, int> parallelIdxMap;
                for (auto &x : grpResults) {
                  if (originalResults.contains(x)) {
                    auto &idxMap = rdCanonicalizer.getResultIdxMap();
                    resultsArray.emplace_back(nxtFor->getResults()[idxMap[x]]);
                  } else {
                    resultsArray.emplace_back(x);
                  }
                  parallelIdxMap.insert({x, resultsArray.size() - 1});
                }
                rdCanonicalizer.setResultIdxMap(parallelIdxMap);
                maybeYieldValue(b, loc, resultsArray);

                // prepare iterArgs
              }
            }
          }

        } else {
          if (parallelIdx == parallelAxis.size() && !isStandaloneOp) {

            Attribute initValueAttr;
            getReductionInitAttr(multiReductionOp, initValueAttr);

            auto accVal = b.create<arith::ConstantOp>(
                loc, DenseElementsAttr::get(getVectorzedType(multiReductionOp),
                                            {initValueAttr}));
            llvm::SmallVector<Value> argsArray;
            argsArray.emplace_back(accVal);
            for (auto [idx, x] : llvm::enumerate(loopState)) {
              if (idx == 0)
                continue;
              argsArray.emplace_back(x);
            }
            ValueRange newIterArgs(argsArray);
            auto nxtFor = reductionAxisGenerateForLoop(
                b, groupIdx, 0, newIterArgs, inductionVars);
            // insert accumulate value to original vector
            auto accRes = nxtFor->getResults()[0];

            Operation *reductionOp = b.create<vector::ReductionOp>(
                loc, multiReductionOp.getKind(), accRes);
            auto insertOp = b.create<vector::InsertOp>(
                loc, reductionOp->getResult(0), initArgs[0], iv);
            auto insertResult = insertOp->getResults()[0];

            // result
            llvm::SmallVector<Value> retResults;
            retResults.emplace_back(insertResult);
            for (auto [idx, x] : llvm::enumerate(nxtFor->getResults())) {
              if (idx == 0) {
                continue;
              }
              retResults.emplace_back(x);
            }
            ValueRange retResultsArray(retResults);
            maybeYieldValue(b, loc, retResultsArray);

          } else {
            auto nxtFor = parallelAxisGenerateForLoop(
                b, groupIdx, parallelIdx + 1, loopState, inductionVars,
                originalWriteResult);
            maybeYieldValue(b, loc, nxtFor->getResults());
          }
        }
      });
  return forOp;
}

scf::ForOp
ForLoopGenerator::generateMultiReductionForLoop(const size_t grpIdx) {
  auto &grpArgs = getGroupOpIterArgs()[grpIdx];
  llvm::SmallVector<Value> forLoopArgs(grpArgs.begin(), grpArgs.end());
  llvm::SmallVector<Value, 5> inductionVars;
  ValueRange initArgs(forLoopArgs);
  Value originalWriteResult;
  auto &rdCanonicalizer = getMultiRdCanonicalizers()[grpIdx];
  auto &rdResultMap = rdCanonicalizer.getResultIdxMap();
  IRRewriter rewriter(func);

  OpBuilder opBuilder(rdCanonicalizer.getCandidateOps()[0]);

  scf::ForOp forOp = parallelAxisGenerateForLoop(
      opBuilder, grpIdx, 0, initArgs, inductionVars, originalWriteResult);

  auto replaceIfFn = [&](OpOperand &use) {
    return use.getOwner()->getBlock() == forOp->getBlock();
  };
  for (auto &grpResult : getGroupOpResults()[grpIdx]) {
    rewriter.replaceOpUsesWithIf(grpResult.getDefiningOp(),
                                 forOp->getResults()[rdResultMap[grpResult]],
                                 replaceIfFn);
  }

  rewriter.replaceOp(getMultiRdCanonicalizers()[grpIdx].getCandidateOps()[0],
                     forOp);
  return forOp;
}

template <class T>
llvm::SmallVector<T, 4> &SpecialOperationCanonicalizer<T>::getCandidateOps() {
  return candidateRdOps;
};

void MultiReductionCanonicalizer::initReductionAxis() {
  auto reductionAxisRange =
      getCandidateOps()[0].getReductionDims().getAsValueRange<IntegerAttr>();
  auto reductionRange = llvm::to_vector<4>(llvm::map_range(
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

template <class T> void addDummyInit(llvm::SmallVector<T, 8> &canonicalizer) {
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
  auto &opGroups = getFusionStrategy().getOpGroups();
  for (auto &grp : opGroups) {
    dummyInitSpecialOperation();
    if (grp.empty()) {
      continue;
    }
    std::queue<Operation *> tempQ(grp);
    while (!tempQ.empty()) {
      auto op = tempQ.front();
      tempQ.pop();
      if (mlir::isa<vector::MultiDimReductionOp>(op)) {
        getMultiRdCanonicalizers().back().getCandidateOps().emplace_back(
            mlir::dyn_cast<vector::MultiDimReductionOp>(op));
        getMultiRdCanonicalizers().back().prepareSpecialOperationInfo();
      } else if (mlir::isa<vector::BroadcastOp>(op)) {
        getBroadcastCanonicalizers().back().getCandidateOps().emplace_back(
            mlir::dyn_cast<vector::BroadcastOp>(op));
      } else if (mlir::isa<vector::TransposeOp>(op)) {
        getTransposeCanonicalizers().back().getCandidateOps().emplace_back(
            mlir::dyn_cast<vector::TransposeOp>(op));
      } else if (mlir::isa<vector::ShapeCastOp>(op)) {
        getShapeCastCanonicalizers().back().getCandidateOps().emplace_back(
            mlir::dyn_cast<vector::ShapeCastOp>(op));
      }
    }
  }
}

LogicalResult CanonicalizerVectorOperation::canonicalizeReductionOperation() {
  OpBuilder::InsertionGuard guard(rewriter);

  initSpeicalOperationCanonicalizers();
  // traverse all groups
  auto &multiRdCanonicalizers = getMultiRdCanonicalizers();
  for (auto [groupId, rdCanonicalizer] :
       llvm::enumerate(multiRdCanonicalizers)) {
    auto &candidateOps = rdCanonicalizer.getCandidateOps();
    if (candidateOps.empty()) {
      continue;
    }
    // generate MultiReduction for loops
    // (void)generateMultiReductionForLoop(groupId);
  }
  return success();
}

void CanonicalizerVectorOperation::canonicalizeSpecialOperation() {
  // multireduction operation
  auto result = canonicalizeReductionOperation();
  if (failed(result)) {
    LDBG("Failed to canonicalize reduction operation\n");
    assert(0 && "Failed to canonicalize reduction operation");
  }
}

void CanonicalizerVectorOperation::run() {
  auto &fusionStrategy = getFusionStrategy();
  if (kind == CanonicalizerKind::OperationsGroup) {
    // 1. Analysis the operation's operands and results
    // We need to analyze which operation results are needed by other
    // operations, and we need to pass these results correctly. Mapping the
    // operation result value to forloop yeild result value. We can replace the
    // operation operand as: map(operand, forloop yield result) -> operand =
    // loop yield result We put all the operation result into this map.

    // 1.a. Find results which should be generated by current group for
    // using as operands to other operations?

    // Traverse all operations. If the operand of operations in other groups or
    // outside the group is the result of the current group operation, then the
    // current operation needs to generate a result. We use `setvector` to save
    // the results that need to be generated by the current group.

    //  1.b. What operands are needed to find in the current group, and where
    //  can they be obtained ?

    //  Thanks to 1.a, we get the result generated by the operations of
    //  each group, and this result will use `for loop yield` to generate a
    //  new result. Since the scope of the parent block of mlir is covered
    //  the current operation, the current operation does not need to pass these
    //  `for loop results` to the `iterArgs` of the required `for loop`. It
    //  only needs to replace the operand of the current operation with the
    //  corresponding `for loop yield result`.

    // However, for some operations that are not DPS, we need to canonicalize
    // them. Canonicalization means that the operand of this operation is a
    // vector but we can't get this vector due to it locates in another block
    // which has a different scope. Therefore, it is necessary to write the
    // vector results into a temporary tensor to save it. Then the vector needs
    // to be read from the tensor before the current operation operate on it.
    // Therefore,  `empty tensor`, `transfer_write` and `transfer_read` need to
    // be inserted at target place.

    // Query groupResultYeildSet to map operaion result value to scf.yield
    // result value.
    analysisGroupOperaion();
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
    const llvm::DenseMap<Value, int> &operandIdxMap,
    const llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Operation *, AffineMap> &opPermuationMap) {
  for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
    if (operandIdxMap.contains(opd)) {
      op->setOperand(idx, iterArgs[operandIdxMap.at(opd)]);
    }
  }
  int offset = isa<vector::TransferWriteOp>(op) ? 2 : 1;
  if (llvm::dyn_cast<vector::TransferWriteOp>(op) ||
      llvm::dyn_cast<vector::TransferReadOp>(op)) {
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
    const Location &loc, const ValueRange &iterArgs, const VectorType &type,
    const llvm::ArrayRef<int64_t> &dims,
    llvm::SmallVector<Value, 5> &inductionVars,
    const llvm::DenseMap<Value, int> &operandIdxMap) {
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
          moveOperationsToCurrentForBody(groupIdx, b, inductionVars,
                                         operandIdxMap, loopState);
          auto &resultSet = getGroupOpResults()[groupIdx];
          maybeYieldValue(b, loc, resultSet.getArrayRef());
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

bool isCompatibleVectorType(Operation *op1, Operation *op2) {
  auto type1 = getOperationVectorType(op1);
  auto type2 = getOperationVectorType(op2);
  if (failed(type1) || failed(type2)) {
    return false;
  }
  auto sp1 = type1.value();
  auto sp2 = type2.value();
  if (isReadOrWriteOperation(op1) or isReadOrWriteOperation(op2)) {
    if (sp1.getRank() != sp2.getRank()) {
      return false;
    }
    for (long i = 0; i < sp1.getRank(); i++) {
      if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
        return false;
      }
    }
  }
  if (sp1.getRank() != sp2.getRank()) {
    return false;
  }
  bool isCompatible = true;
  // from front to back
  for (long i = 0; i < sp1.getRank(); i++) {
    if (sp1.getDimSize(i) != sp2.getDimSize(i)) {
      isCompatible = false;
      break;
    }
  }

  return isCompatible;
}

bool isPartialCompatible(Operation *op1, Operation *op2) {
  auto type1 = getOperationVectorType(op1);
  auto type2 = getOperationVectorType(op2);
  if (failed(type1) || failed(type2)) {
    return false;
  }
  auto sp1 = type1.value();
  auto sp2 = type2.value();
  // must be total same
  if (isReadOrWriteOperation(op1) or isReadOrWriteOperation(op2)) {
    return false;
  }
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
                         llvm::SmallVector<int64_t> &res) {
  unsigned rankA = a.size();
  unsigned rankB = b.size();
  assert(rankA < rankB && "May be invalid shape cast operation.");

  auto isOne = [](int64_t v) { return v == 1; };

  // Special-case for n-D to 0-d shape cast. 'b' must be all ones to be shape
  // casted to a 0-d vector.
  if (rankA == 0 && llvm::all_of(b, isOne)) {
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
    if (i < rankA && llvm::all_of(a.slice(i), isOne))
      i = rankA;
    if (j < rankB && llvm::all_of(b.slice(j), isOne))
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
                        llvm::SmallVector<int64_t> &bcAxis) {
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

void getOperationDataAxis(Operation *op, llvm::SmallVector<int64_t> &dataAxis) {
  return TypeSwitch<Operation *>(op)
      .Case<vector::MultiDimReductionOp>(
          [&](vector::MultiDimReductionOp multiReductionOp) {
            auto rdDimsRange = multiReductionOp.getReductionDims()
                                   .getAsValueRange<IntegerAttr>();
            auto reductionDims = llvm::to_vector<4>(llvm::map_range(
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
  auto hasSameAxis = [](const llvm::SmallVector<int64_t> &dims1,
                        const llvm::SmallVector<int64_t> &dims2) {
    llvm::DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());
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
            llvm::SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            return hasSameAxis(dims1, dims2);
          })
          .Case<vector::MultiDimReductionOp>(
              [&](vector::MultiDimReductionOp multiReductionOp) {
                // op1 is special operation, op2 is normal operation
                // op1 and op2 is both speicial operation
                llvm::SmallVector<int64_t> dims2, reductionDims, parallelDims;
                getOperationDataAxis(op1, reductionDims);
                getOperationDataAxis(op2, dims2);
                llvm::DenseSet<int64_t> checkSet(dims2.begin(), dims2.end());

                if (!isSpecialOp(op2)) {
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
                  auto rank =
                      mlir::dyn_cast<ShapedType>(op2->getResultTypes()[0])
                          .getRank();
                  for (auto i = 0; i < rank; i++) {
                    if (!checkSet.contains(i)) {
                      return true;
                    }
                  }

                  return false;
                } else {
                  // TODO: reduce operation fused with other special operation
                  if (mlir::isa<vector::TransposeOp>(op2)) {
                    return true;
                  } else if (mlir::isa<vector::BroadcastOp>(op2)) {
                    return true;
                  }
                  //...
                }

                return true;
              })
          .Case<vector::BroadcastOp>([&](vector::BroadcastOp broadcastOp) {
            llvm::SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            if (!isSpecialOp(op2)) {
              return hasSameAxis(dims1, dims2);
            } else {
            }
            return true;
          })
          .Case<vector::TransposeOp>([&](vector::TransposeOp transposeOp) {
            llvm::SmallVector<int64_t> dims1, dims2;
            getOperationDataAxis(op1, dims1);
            getOperationDataAxis(op2, dims2);
            if (!isSpecialOp(op2)) {
              return hasSameAxis(dims1, dims2);
            } else {
            }
            return true;
          })
          .Default([&](Operation *op) { return false; });

  return res;
}

bool isNeedNewGroup(llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
                    Operation *op) {
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
      // TODO: Support partial compatible operation fusion
      if (isPartialCompatible(prevOp, op)) {
      }
      return true;
    }
  }
  return false;
}

void addOperationToGroup(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap, Operation *op) {
  //
  if (isNeedNewGroup(opGroups, op)) {
    opGroups.emplace_back(std::queue<Operation *>());
  }
  opGroups.back().push(op);
  opGroupIndexMap[op] = opGroups.size() - 1;
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
      addOperationToGroup(opGroups, opGroupIndexMap, op);
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
                  resultElementType,
                  llvm::cast<FloatAttr>(value).getValueAsDouble());
            } else {
              initValueAttr = IntegerAttr::get(
                  resultElementType, llvm::cast<IntegerAttr>(value).getInt());
            }

            auto cntOp = rewriter.create<arith::ConstantOp>(
                rewriter.getUnknownLoc(),
                DenseElementsAttr::get(newOperandType, {initValueAttr}));
            return cntOp->getResults()[0];
          })
          .Default([&](Operation *op) { return Value(); });
  return ret;
}

void setOperationOperandResult(
    Operation *op, const VectorType &newOperandType,
    const llvm::DenseMap<Operation *, size_t> &opMap) {
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
  auto srcConstantOp = mlir::dyn_cast<arith::ConstantOp>(srcOp);
  Operation *newConstantOp;
  if (mlir::isa<ElementsAttr>(srcConstantOp.getValue())) {
    auto valueType = mlir::dyn_cast<ElementsAttr>(srcConstantOp.getValue());
    if (valueType.isSplat()) {
      if (mlir::isa<FloatType>(valueType.getElementType())) {
        newConstantOp = srcWriter.create<arith::ConstantOp>(
            srcOp->getLoc(),
            FloatAttr::get(newOperandType, valueType.getSplatValue<APFloat>()));
      } else {
        newConstantOp = srcWriter.create<arith::ConstantOp>(
            srcOp->getLoc(),
            IntegerAttr::get(newOperandType, valueType.getSplatValue<APInt>()));
      }
    } else {
      assert(0 && "Not support non-splat constant value.");
    }
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

/// Rewrite the operations in the group to vectorized form.
void ForLoopGenerator::rewriteOperationAsVectorize(
    OpBuilder &rewriter, size_t groupId, const std::queue<Operation *> &queue) {
  auto &groupOps =
      queue.empty() ? getFusionStrategy().getOpGroups()[groupId] : queue;
  auto &opMap = getFusionStrategy().getOpGroupIndexMap();
  auto &opPermuationMap = getOpPermuationMap();
  std::queue<Operation *> transformQueue(groupOps);
  auto groupSteps = getFusionStrategy().getGroupMaxSteps()[groupId];

  while (!transformQueue.empty()) {
    auto op = transformQueue.front();
    transformQueue.pop();
    auto lowerResult =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<vector::TransferWriteOp>(
                [&](vector::TransferWriteOp transferWriteOp) {
                  IRRewriter rewriter(transferWriteOp);
                  auto newOperandType =
                      getVectorzedType(transferWriteOp, groupSteps);
                  auto srcOp = transferWriteOp->getOperand(0).getDefiningOp();
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
                  auto newOperandType =
                      getVectorzedType(transferReadOp, groupSteps);
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
                  llvm::llvm_unreachable_internal(
                      "It should not appear this operation.");
                  return failure();
                })
            .Case<arith::ExtFOp>([&](arith::ExtFOp extfOp) {
              auto newOperandType = getVectorzedType(extfOp, groupSteps);
              extfOp->getResult(0).setType(newOperandType);
              return success();
            })
            .Default([&](Operation *op) {
              if (isSpecialOp(op)) {
                llvm::llvm_unreachable_internal(
                    "It should not appear this operation.");
                return failure();
              }
              setOperationOperandResult(op, getVectorzedType(op, groupSteps),
                                        opMap);
              return success();
            });
    if (failed(lowerResult)) {
      LDBG("Failed to rewrite operation: " << *op << "\n");
      assert(false && "Failed to rewrite operation");
    }
  }
}

mlir::FailureOr<Value> getOperationOperateTensor(Operation *op) {
  return llvm::TypeSwitch<Operation *, mlir::FailureOr<Value>>(op)
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

void updateOpOperandResultInGroups(
    llvm::SmallVector<std::queue<Operation *>, 8> &opGroups,
    llvm::DenseMap<Operation *, size_t> &opGroupIndexMap, size_t opGid,
    Operation *op, Value &init, const Value &result = Value()) {
  auto tmpOpQueue(opGroups[opGid]);
  std::queue<Operation *> newOpQueue;
  while (!tmpOpQueue.empty()) {
    auto curOp = tmpOpQueue.front();
    tmpOpQueue.pop();
    if (curOp == op) {
      if (!failed(getOperationVectorType(init.getDefiningOp()))) {
        newOpQueue.push(init.getDefiningOp());
        opGroupIndexMap[init.getDefiningOp()] = opGid;
      }

      newOpQueue.push(op);

      if (result && !failed(getOperationVectorType(result.getDefiningOp()))) {
        newOpQueue.push(result.getDefiningOp());
        opGroupIndexMap[result.getDefiningOp()] = opGid;
      }
    } else {
      newOpQueue.push(curOp);
    }
  }
  opGroups[opGid] = newOpQueue;
}

void VectorFusionStrategy::run() { classifyOperations(); }

void VectorOperationAnalysizer::generateEmptyTensorAndWrite(
    Operation *sourceOp, llvm::DenseMap<Operation *, std::pair<Value, Value>>
                             &srcOpCanoniclizedMap) {
  auto &opGroups = getFusionStrategy().getOpGroups();
  auto &opGroupIndexMap = getFusionStrategy().getOpGroupIndexMap();
  auto &groupOpIterArgs = getGroupOpIterArgs();
  auto &groupOpResults = getGroupOpResults();
  auto sourceOpGid = opGroupIndexMap[sourceOp];

  auto [resultTensor, result] = canonicalizeSourceOperation(sourceOp);
  srcOpCanoniclizedMap.insert({sourceOp, {resultTensor, result}});
  updateOpOperandResultInGroups(opGroups, opGroupIndexMap, sourceOpGid,
                                sourceOp, resultTensor, result);
  groupOpIterArgs[sourceOpGid].insert(resultTensor);
  groupOpResults[sourceOpGid].insert(result);
}

void VectorOperationAnalysizer::analysisEmptyGroupAndMaxSteps() {
  auto &groupOpResults = getGroupOpResults();
  auto &opGroups = getFusionStrategy().getOpGroups();

  // If the group operations do not have result need to be returned, these are
  // useless code.
  for (auto [idx, grp] : enumerate(opGroups)) {
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

// analysis operation result of current group whether needed by other
// operation which out of current group
void VectorOperationAnalysizer::analysisGroupOperationResults() {
  llvm::DenseMap<Operation *, std::pair<Value, Value>> srcOpCanoniclizedMap;
  llvm::DenseSet<Operation *> movedOperationSet;
  auto &opGroups = getFusionStrategy().getOpGroups();
  auto &opGroupIndexMap = getFusionStrategy().getOpGroupIndexMap();
  auto &groupOpIterArgs = getGroupOpIterArgs();
  auto &groupOpResults = getGroupOpResults();
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (auto [idx, opd] : llvm::enumerate(op->getOperands())) {
      auto sourceOp = opd.getDefiningOp();

      if (opGroupIndexMap.contains(sourceOp)) {
        auto sourceOpGid = opGroupIndexMap[sourceOp];
        bool notInSameGroup =
            opGroupIndexMap.contains(op) && sourceOpGid != opGroupIndexMap[op];
        bool outOfGroup = !opGroupIndexMap.contains(op);

        if (notInSameGroup or outOfGroup) {
          // update init iterargs
          auto dstRet = getOperationOperateTensor(sourceOp);
          // need to generate tensor.emtpy and vector.transfer_write, write
          // operand to tensor and read operand from the tensor, generate
          // vector.transfer_read
          if (failed(dstRet)) {
            // already generate result tensor
            if (!srcOpCanoniclizedMap.contains(sourceOp)) {
              generateEmptyTensorAndWrite(sourceOp, srcOpCanoniclizedMap);
            }

            auto opInit = canonicalizeCurrentOperation(
                op, srcOpCanoniclizedMap[sourceOp].second, idx);
            updateOpOperandResultInGroups(opGroups, opGroupIndexMap,
                                          opGroupIndexMap[op], op, opInit);

          } else {
            // if source operation is transfer_read, we need to generate a
            // same transfer_read operation like source operation.
            if (mlir::isa<vector::TransferReadOp>(sourceOp)) {
              auto transferReadOp =
                  mlir::dyn_cast<vector::TransferReadOp>(sourceOp);
              auto opInit = canonicalizeCurrentOperation(op, dstRet.value(),
                                                         idx, &transferReadOp);
              updateOpOperandResultInGroups(opGroups, opGroupIndexMap,
                                            opGroupIndexMap[op], op, opInit);

            } else {
              groupOpIterArgs[sourceOpGid].insert(dstRet.value());
              groupOpResults[sourceOpGid].insert(opd);
            }
          }
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
  LDBG("Complete analysis group operation results\n");
}

void VectorOperationAnalysizer::analysisGroupOperaion() {
  // Results
  analysisGroupOperationResults();
}

mlir::FailureOr<scf::ForOp> ForLoopGenerator::generateVectorizedForLoop(
    const size_t groupId, IRRewriter &rewriter, const VectorType &vectorType) {
  auto &resultSet = getGroupOpResults();
  auto &dstOperandSet = getGroupOpIterArgs()[groupId];
  assert(!resultSet.empty() && "Expected non-empty value");
  // prepare for loop iterargs
  llvm::SmallVector<Value, 4> operands;
  llvm::DenseMap<Value, int> operandIdxMap;
  for (auto [idx, x] : llvm::enumerate(dstOperandSet)) {
    operands.emplace_back(x);
    operandIdxMap[x] = operands.size() - 1;
  }
  ValueRange iterArgs(operands);
  auto shapes = vectorType.getShape();
  llvm::SmallVector<Value, 5> inductionVars;
  // generate for loop
  auto forOp = constructNestedForOp(
      0, groupId, rewriter, rewriter.getUnknownLoc(), iterArgs, vectorType,
      shapes, inductionVars, operandIdxMap);
  return forOp;
}

void updateLoopResultUses(llvm::SetVector<Value> &opResults,
                          scf::ForOp *forOp) {
  if (opResults.empty()) {
    return;
  }
  IRRewriter rewriter(*forOp);
  OpBuilder::InsertionGuard g(rewriter);
  // Only different group operation operand need to be replaced due to same
  // group operation should directly use original operand.
  auto producerOp = opResults.front().getDefiningOp();
  auto needToReplaced = [&](OpOperand &operand) {
    return producerOp->getBlock() != operand.getOwner()->getBlock();
  };
  // update loop result uses
  for (auto [retIdx, rt] : llvm::enumerate(opResults)) {
    producerOp = rt.getDefiningOp();
    rewriter.replaceUsesWithIf(rt, forOp->getResult(retIdx), needToReplaced);
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
  auto getType = getOperationVectorType(grp.front());
  if (failed(getType)) {
    LDBG("Failed to get vector type for operation: " << *grp.front() << "\n");
    return;
  }
  auto opShapes = getType.value();
  IRRewriter rewriter(grp.back());
  rewriter.setInsertionPointAfter(grp.back());
  // 1. Rewrite operation as vectorized form
  // 2. Generate loop
  rewriteOperationAsVectorize(rewriter, idx);
  auto forOp = generateVectorizedForLoop(idx, rewriter, opShapes);
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
    // canonicalize vector operation, default use vector-based fusion strategy.
    CanonicalizerVectorOperation canonicalizer(
        func, CanonicalizerKind::OperationsGroup);
    canonicalizer.run();
    // patterns.add<TransferReadWriteOpFolder>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

} // namespace gc
} // namespace mlir