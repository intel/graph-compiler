//===- LinalgxOps.td - linalgx dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_OPS
#define LINALGX_OPS

include "gc/Dialect/Linalgx/IR/LinalgxDialect.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalgx_Op<string mnemonic, list<Trait> traits = []> :
    Op<LinalgxDialect, mnemonic, traits>;

#endif // LINALGX_OPS