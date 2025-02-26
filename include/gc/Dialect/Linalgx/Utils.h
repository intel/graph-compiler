//===- Utils.h - linalgx utils ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECTS_LINALGX_UTILS_H
#define GC_DIALECTS_LINALGX_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace linalgx {

/// @brief enum of type of matmul packing
enum class PackingType : int {
  MM4D = 0,    // MKmk x NKkn
  MM2D4D,      // MK x NKkn
  VNNI_MM2D,   // MK x NKknV
  VNNI_MM4D,   // MKmk x NKknV
  VNNI_BRMM3D, // BMK x BKNV
  NUM_TYPES,
};

/// @brief make a generic packed matmul Op based on PackingType
/// @param builder builder
/// @param loc location
/// @param opType the PackingType
/// @param inputs matmul A, B
/// @param outputs matmul C
/// @return the generic packed matmul Op
FailureOr<linalg::GenericOp>
makeGenericPackedMatmulOp(OpBuilder &builder, Location loc, PackingType opType,
                          ValueRange inputs, ValueRange outputs);

/// @brief identify a generic packed matmul Op based on PackingType
/// @param op the op
/// @param opType the PackingType
/// @return true if op is a generic packed matmul Op
bool isGenericPackedMatmulOp(Operation *op, PackingType opType);

template <typename... Args>
inline bool isGenericPackedMatmulOp(Operation *op, PackingType first,
                                    Args... args) {
  return isGenericPackedMatmulOp(op, first) ||
         isGenericPackedMatmulOp(op, args...);
}

/// @brief identify a generic packed matmul Op based on any PackingType
/// @param op the op
/// @return true if op is a generic packed matmul Op
template <int T, int N> inline bool isAnyGenericPackedMatmulOp(Operation *op) {
  return isGenericPackedMatmulOp(op, (PackingType)N) ||
         isAnyGenericPackedMatmulOp<T + 1, N>(op);
}
constexpr int NUM_ALL_TYPES = (int)PackingType::NUM_TYPES;
template <>
inline bool
isAnyGenericPackedMatmulOp<NUM_ALL_TYPES, NUM_ALL_TYPES>(Operation *op) {
  return false;
}
inline bool isAnyGenericPackedMatmulOp(Operation *op) {
  return isAnyGenericPackedMatmulOp<0, NUM_ALL_TYPES>(op);
}

/// @brief identify a matmul Op based on ContractionOp and PackingType
/// @param op the op
/// @return true if op is a matmul Op
bool isMatmulOp(Operation *op);

} // namespace linalgx
} // namespace mlir

#endif // GC_DIALECTS_LINALGX_UTILS_H
