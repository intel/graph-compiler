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

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalgx_Op<string mnemonic, list<Trait> traits = []> :
    Op<LinalgxDialect, mnemonic, traits>;

def Linalgx_ScaledDotProductAttentionOp
 : Linalgx_Op<"scaled_dot_product_attention",
  [AttrSizedOperandSegments,
   DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>]> {
  let summary = "Attention structure.";
  let description = [{
    Q, K, V, attention_mask.
    Output = SoftMax(Q @ K.transpose(-2, -1) + attention_mask) @ V.
  }];
  let arguments = (ins
      Variadic<AnyRankedTensor>:$inputs,
      Variadic<AnyRankedTensor>:$outputs);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    (`->` type($results)^)?
  }];
}
#endif // LINALGX_OPS