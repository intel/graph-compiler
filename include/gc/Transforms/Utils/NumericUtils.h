//===-- NumericUtils.h - numeric utilities ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_UTILS_NUMERICUTILS_H
#define GC_TRANSFORMS_UTILS_NUMERICUTILS_H
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
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

} // namespace gc
} // namespace mlir

#endif