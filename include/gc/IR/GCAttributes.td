//===- GCAttributes.td - Graph Compiler attributes --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_IR_GCATTRIBUTES_TD
#define GC_IR_GCATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def GCDialect : Dialect {
  let name = "gc";
  let cppNamespace = "::mlir::gc";
  let summary = "Graph Compiler metadata";
  let useDefaultAttributePrinterParser = 1;
}

def FusionGroupAttr : AttrDef<GCDialect, "FusionGroup"> {
  let mnemonic = "fusion_group";
  let summary = "Identifier for a group of operations intended for one kernel";
  let description = [{
    Operations in the same fusion group carry the same ID. The ID is unique
    within a function; checking group membership is the responsibility of the
    compiler pass that consumes this attribute.
  }];
  let parameters = (ins "int64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}

#endif // GC_IR_GCATTRIBUTES_TD
