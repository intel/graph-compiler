//===- MicrokernelDialect.td - microkernel dialect ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MICROKERNEL_DIALECT
#define MICROKERNEL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Microkernel dialect definition.
//===----------------------------------------------------------------------===//

def Microkernel_Dialect : Dialect {
    let name = "microkernel";
    let summary = "A dialect for microkernel abstraction.";
    let description = [{
        The dialect wraps the BRGEMM API to set up the HW context etc.
    }];
    let cppNamespace = "::mlir::microkernel";

    let useDefaultTypePrinterParser = 1;
}

#endif // MICROKERNEL_DIALECT
