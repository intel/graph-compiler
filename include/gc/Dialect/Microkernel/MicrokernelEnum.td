//===- MicrokernelEnum.td - microkernel dialect enum -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MICROKERNEL_ENUM
#define MICROKERNEL_ENUM

include "mlir/IR/EnumAttr.td"
include "gc/Dialect/Microkernel/MicrokernelDialect.td"

def Microkernel_BrgemmFlags : I64EnumAttr<
    "BrgemmFlags", "see: microkernel_brgemm_flags",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"BETA_0", 1, "beta_0">,
      I64EnumAttrCase<"STRIDE", 2, "stride">,
      I64EnumAttrCase<"LIST", 4, "list">
    ]> {
  let cppNamespace = "::mlir::microkernel";
}

#endif // MICROKERNEL_ENUM
