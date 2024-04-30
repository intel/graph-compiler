//===- LinalgxDialect.td - linalgx dialect -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_DIALECT
#define LINALGX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Linalgx dialect definition.
//===----------------------------------------------------------------------===//

def LinalgxDialect : Dialect {
    let name = "linalgx";
    let summary = "A dialect for linalg extensions.";
    let description = [{
        The dialect primarily aims to fill the gap for matmul ops
        missing tiling semantics when consuming packed tensors. 
    }];
    let cppNamespace = "::mlir::linalgx";

    let useDefaultTypePrinterParser = 1;
}

#endif // LINALGX_DIALECT
