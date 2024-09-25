//===-- VectorUtils.h - vector fusion analysis ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_UTILS_VECTORUTILS_H
#define GC_TRANSFORMS_UTILS_VECTORUTILS_H
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include <queue>
#include <stdint.h>
#include <variant>

namespace mlir {
namespace gc {

enum class OPPRIORITY : uint8_t {
  FIRST = 0,
  SECOND,
  THIRD,
  LAST,
  OTHERS = 255,
};
/// Need to move some operations like extract_slice or insert_slice.
/// Because those operation may interpret our analysis result. e.g.:
/// ```
/// clang-format off
/// %21 = vector.transfer_read %18[%c0, %c0], %cst {in_bounds = [true, true]} :
/// tensor<16x16xf32>, vector<16x16xf32> %22 = arith.addf %21, %20 :
/// vector<16x16xf32> %23 = vector.transfer_write %22, %extracted_slice_12[%c0,
/// %c0] {in_bounds = [true, true]} : vector<16x16xf32>, tensor<16x16xf32>
/// %inserted_slice_13 = tensor.insert_slice %18 into %arg14[%arg13, 0] [16, 16]
/// [1, 1] : tensor<16x16xf32> into tensor<32x16xf32> %extracted_slice_14 =
/// tensor.extract_slice %arg16[%arg13, 0] [16, 16] [1, 1] : tensor<32x16xf32>
/// to tensor<16x16xf32> %24 = vector.transfer_read %cst_0[%c0, %c0], %cst
/// {in_bounds = [true, true]} : tensor<16x16xf32>, vector<16x16xf32> %25 =
/// arith.maximumf %22, %24 : vector<16x16xf32> %26 = vector.transfer_write %25,
/// %extracted_slice_14[%c0, %c0] {in_bounds = [true, true]} :
/// vector<16x16xf32>, tensor<16x16xf32> %inserted_slice_15 =
/// tensor.insert_slice %23 into %arg15[%arg13, 0] [16, 16] [1, 1] :
/// tensor<16x16xf32> into tensor<32x16xf32> %inserted_slice_16 =
/// tensor.insert_slice %26 into %arg16[%arg13, 0] [16, 16] [1, 1] :
/// tensor<16x16xf32> into tensor<32x16xf32> clang-format on
/// ```
/// The maximumf and addf operation can be a same group, but the extract_slice
/// operation interpret us.
/// The move operation(extra_slice) will check its parameters. In order to
/// ensure that it does not affect the correctness of the result, we will only
/// move the moved op after the op to which the parameters belong to. If it's
/// operand is all the block argument, we will move it to the begining of the
/// block.
/// insert_slice just move them to the privious of the first operation which
/// use it.
void moveOpsFrontOrBack(func::FuncOp *func, IRRewriter &rewriter,
                        OPPRIORITY start, OPPRIORITY end);

/// build a constant operation of index type
Value makeIndexArithConstantOp(OpBuilder &opBuilder, const Location &loc,
                               int64_t x);

/// find the original tensor
Value findOriginalTensor(Value writeTensor, Block *block);
/// get operation read or write tensor
mlir::FailureOr<Value> getOperationOperateTensor(Operation *op);

/// set correct operand for the operation
void setOperationCorrectOperand(
    Operation *op, ValueRange iterArgs, DenseMap<Value, int> &operandIdxMap,
    DenseMap<Value, Value> &originalOperandLoopArgsMap,
    ArrayRef<Value> inductionVars,
    DenseMap<Operation *, AffineMap> &opPermuationMap);

/// Get vector type of the operation \param op
/// \param isPrevOp whether the operation is a previous operation, if it is not
/// prev-op, may need to use result vectortype
/// default will return the opeation result type
mlir::FailureOr<VectorType> getOperationVectorType(Operation *op,
                                                   bool isPrevOp = true);

/// select nearest even step
int getNearestVectorStep(const int step);

/// get operation vector type
/// \param isPrevOp whether the operation is a previous operation, if it is not
/// prev-op, may need to use result vectortype
/// default will return the opeation result type
mlir::FailureOr<VectorType> getOperationMaxVectorType(Operation *op);
union Float32Bits {
  uint32_t u;
  float f;
};
uint16_t float2half(float floatValue);
float half2float(uint16_t halfValue);
uint16_t float2bfloat(float floatValue);
float bfloat2float(uint16_t bfloatBits);
std::variant<float, int64_t> numeric_limits_minimum(Type type);
std::variant<float, int64_t> numericLimitsMaximum(Type type);

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
    if (not isa<FloatType>(t1))
      llvm_unreachable("Expected float values.");
    result = std::get<T>(numeric_limits_minimum(t));
    break;
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    if (not isa<FloatType>(t1))
      llvm_unreachable("Expected float values.");
    result = std::get<T>(numericLimitsMaximum(t));
    break;
  case vector::CombiningKind::MAXSI:
  case vector::CombiningKind::MAXUI:
    if (not t1.isIntOrIndex())
      llvm_unreachable("Expected int or index values.");
    result = std::get<T>(numeric_limits_minimum(t));
    break;
  case vector::CombiningKind::MINSI:
  case vector::CombiningKind::MINUI:
    if (not t1.isIntOrIndex())
      llvm_unreachable("Expected int or index values.");
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

template <typename TARGETOP>
void getSameBlockTargetOp(Operation *op,
                          std::queue<Operation *> &candidateOps) {
  if (isa<TARGETOP>(op)) {
    candidateOps.push(op);
    return;
  }
  auto getSameBlockSrcOp = [](Operation *trackSrcOp,
                              std::queue<Operation *> &trackOps,
                              std::queue<Operation *> &candidateOps) {
    for (Value opd : trackSrcOp->getOperands()) {
      if (isa<BlockArgument>(opd) or
          opd.getDefiningOp()->getBlock() != trackSrcOp->getBlock())
        continue;
      if (isa<TARGETOP>(opd.getDefiningOp()))
        candidateOps.push(opd.getDefiningOp());
      else
        trackOps.push(opd.getDefiningOp());
    }
  };

  std::queue<Operation *> trackOps;
  getSameBlockSrcOp(op, trackOps, candidateOps);
  while (not trackOps.empty()) {
    Operation *cadidateOp = trackOps.front();
    trackOps.pop();
    getSameBlockSrcOp(cadidateOp, trackOps, candidateOps);
  }
}

} // namespace gc
} // namespace mlir

#endif