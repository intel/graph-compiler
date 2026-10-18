//===- TensorMaskingOpInterface.h --------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_TENSOR_MASKING_OP_INTERFACE_H
#define GC_TRANSFORMS_TENSOR_MASKING_OP_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

// clang-format off
#include "gc/Transforms/TensorMaskingOpInterface.h.inc"
// clang-format on

namespace mlir::gc {

void registerTensorMaskingOpInterfaceForLinalg(DialectRegistry &registry);

} // namespace mlir::gc

#endif // GC_TRANSFORMS_TENSOR_MASKING_OP_INTERFACE_H
