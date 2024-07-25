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

enum class VnniOpType {
  NONE = 0,
  MM2D,
  MM4D,
  BRMM3D,
};

FailureOr<linalg::GenericOp>
makeGenericVnniMatmulOp(OpBuilder &builder, Location loc, VnniOpType opType,
                        ValueRange inputs, ValueRange outputs);

bool isGenericVnniMatmulOp(Operation *op, VnniOpType opType);

} // namespace linalgx
} // namespace mlir

#endif // GC_DIALECTS_LINALGX_UTILS_H
