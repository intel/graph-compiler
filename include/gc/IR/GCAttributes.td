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
  let summary = "Fusion group identifier and intended operation tile sizes";
  let description = [{
    Operations in the same fusion group carry the same ID. The ID is unique
    within a function; checking group membership is the responsibility of the
    compiler pass that consumes this attribute.

    The tile_sizes field is a dense i64 array containing the tile sizes intended
    for tiling the operation carrying this attribute. It is written without an
    explicit element type, for example:

    ```mlir
    #gc.fusion_group<42, tile_sizes = [32, 32]>
    ```
  }];
  let parameters = (ins "int64_t":$id,
                        "::mlir::DenseI64ArrayAttr":$tile_sizes);
  let assemblyFormat = "`<` $id `,` `tile_sizes` `=` $tile_sizes `>`";
}

#endif // GC_IR_GCATTRIBUTES_TD
