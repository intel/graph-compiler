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
#include <limits>
#include <stdint.h>
#include <variant>

namespace mlir {
namespace gc {
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