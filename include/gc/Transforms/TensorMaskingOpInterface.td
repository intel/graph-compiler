//===- TensorMaskingOpInterface.td --------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_TENSOR_MASKING_OP_INTERFACE
#define GC_TRANSFORMS_TENSOR_MASKING_OP_INTERFACE

include "mlir/IR/OpBase.td"

def TensorMaskingOpInterface : OpInterface<"TensorMaskingOpInterface"> {
  let cppNamespace = "::mlir::gc";

  let description = [{
    Interface for operations that can produce a padded + masked version of
    themselves. The `getMaskedImplementation` method pads operands to multiples
    of the given sizes and inserts masking logic to preserve correctness.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns a masked (padded) implementation of this operation.
        `padMultiples` specifies the target alignment for each iteration
        dimension. A value of 0 means no padding for that dimension.
        Returns the replacement values for the original operation's results.
      }],
      /*retTy=*/"FailureOr<SmallVector<Value>>",
      /*methodName=*/"getMaskedImplementation",
      /*args=*/(ins
        "OpBuilder &":$builder,
        "ArrayRef<OpFoldResult>":$padMultiples)
    >,
  ];
}

#endif // GC_TRANSFORMS_TENSOR_MASKING_OP_INTERFACE
