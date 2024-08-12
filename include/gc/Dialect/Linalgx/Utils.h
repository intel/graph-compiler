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
enum class PackingType {
  MM4D,        // MKmk x NKkn
  VNNI_MM2D,   // MK x NKknV
  VNNI_MM4D,   // MKmk x NKknV
  VNNI_BRMM3D, // BMK x BKNV
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

} // namespace linalgx
} // namespace mlir

#endif // GC_DIALECTS_LINALGX_UTILS_H
