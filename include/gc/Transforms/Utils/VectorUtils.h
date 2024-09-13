//===-- VectorUtils.h - vector fusion analysis ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_UTILS_VECTORUTILS_H
#define GC_TRANSFORMS_UTILS_VECTORUTILS_H
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>
#include <stdint.h>
#include <variant>

namespace mlir {
namespace gc {
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

} // namespace gc
} // namespace mlir

#endif