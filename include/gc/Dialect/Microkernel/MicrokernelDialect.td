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
        This dialect contains wrappers for microkernel primitives like BRGEMM.
    }];
    let cppNamespace = "::mlir::microkernel";
}

//===----------------------------------------------------------------------===//
// Base microkernel operation definition.
//===----------------------------------------------------------------------===//

class Microkernel_Op<string mnemonic, list<Trait> traits = []> :
        Op<Microkernel_Dialect, mnemonic, traits>;

#endif // MICROKERNEL_DIALECT
