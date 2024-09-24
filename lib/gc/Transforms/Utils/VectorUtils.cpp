//===- VectorUtils.cpp - analysis vector ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gc/Transforms/Utils/VectorUtils.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace gc {

#define DEBUG_TYPE "vector-utils"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define SAFE_EXPAND(X) X
#define LDBG(X) LLVM_DEBUG(DBGS() << SAFE_EXPAND(X) << "\n")

static inline void moveOpBeginingOfBlock(Operation *op) {
  Block *block = op->getBlock();
  if (block->getOperations().empty())
    llvm_unreachable("Emtpy block.");

  if (&block->front() == op)
    return;
  op->moveBefore(&block->front());
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
            .Case<vector::BroadcastOp>([&](vector::BroadcastOp bcOp) {
              return moveFront(op, operationPosition);
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
void moveOpsFrontOrBack(func::FuncOp *func, MLIRContext *ctx,
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
    llvm_unreachable("unsupported data type");
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
    llvm_unreachable("unsupported data type");
    return (int64_t)0;
  }
}

mlir::FailureOr<VectorType> getOperationVectorType(Operation *op,
                                                   bool isPrevOp) {
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
                    dyn_cast<VectorType>(op->getOperandTypes()[0]))
              return shapedType;

            return failure();
          });
  if (!failed(ret) and isDynamicType(ret.value()))
    return failure();

  return ret;
}

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

            if (op->getResultTypes().empty() or
                not isa<VectorType>(op->getResultTypes()[0]))
              return cast<VectorType>(op->getOperandTypes()[0]);

            if (op->getOperandTypes().empty() or
                not isa<VectorType>(op->getOperandTypes()[0]))
              return cast<VectorType>(op->getResultTypes()[0]);

            auto opdType = cast<VectorType>(op->getOperandTypes()[0]);
            auto retType = cast<VectorType>(op->getResultTypes()[0]);
            return opdType.getRank() > retType.getRank() ? opdType : retType;
          });
  if (!failed(ret) and isDynamicType(ret.value()))
    return failure();

  return ret;
}

int getNearestVectorStep(const int step) {
  if (step <= 0)
    llvm_unreachable("Wrong step.");

  int nbits = 0, n = step;
  while (n) {
    n = n >> 1;
    nbits++;
  }
  if (nbits > 6 and (nbits != 7 or step != 64))
    llvm_unreachable("wrong nbits appear");
  return (1 << (nbits - 1)) == step ? step : (1 << nbits);
}

Value makeIndexArithConstantOp(OpBuilder &opBuilder, const Location &loc,
                               int64_t x) {
  return opBuilder.create<arith::ConstantOp>(
      loc, opBuilder.getIndexType(),
      opBuilder.getIntegerAttr(opBuilder.getIndexType(), x));
}

Value findOriginalTensor(Value writeTensor, Block *block) {
  while (auto wtOp = dyn_cast_or_null<vector::TransferWriteOp>(
             writeTensor.getDefiningOp())) {
    if (block != writeTensor.getDefiningOp()->getBlock())
      break;

    writeTensor = wtOp->getOperand(1);
  }
  return writeTensor;
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
      .Default([&](Operation *op) { return failure(); });
}

} // namespace gc
} // namespace mlir