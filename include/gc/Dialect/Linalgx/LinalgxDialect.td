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
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"

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

    let dependentDialects = [
        "arith::ArithDialect",
        "affine::AffineDialect",
        "math::MathDialect",
        "memref::MemRefDialect",
        "tensor::TensorDialect",
    ];

    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        /// Attribute name used to memoize indexing maps for named ops.
        constexpr const static ::llvm::StringLiteral
            kMemoizedIndexingMapsAttrName = "linalg.memoized_indexing_maps";

        using RegionBuilderFunType = llvm::function_ref<
        void(ImplicitLocOpBuilder &b, Block &, ArrayRef<NamedAttribute>)>;
        RegionBuilderFunType getRegionBuilder(StringRef name) {
        return namedStructuredOpRegionBuilders.lookup(name);
        }
        private:
        llvm::StringMap<RegionBuilderFunType> namedStructuredOpRegionBuilders;
    }];
}

class Linalgx_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LinalgxDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def PackingMapAttr : Linalgx_Attr<"PackingMap", "packing_map"> {
  let summary = "An Attribute containing a map between index of matmul input/output";
  let description = [{
    A map between index of matmul input/output.
  }];

  let cppNamespace = "::mlir::linalgx";
  let parameters = (ins ArrayRefParameter<"uint64_t">:$first,
                        ArrayRefParameter<"uint64_t">:$second);

  let assemblyFormat = "`<` `[` $first `]` `->` `[` $second `]` `>`";

  let extraClassDeclaration = [{
    /// Index first is 0; Index second is 1
    unsigned getPackingSrcIndex() {
      return getFirst().size() == 1 ? 0 : 1;
    }
    unsigned getPackingDstIndex() {
      return getFirst().size() == 1 ? 1 : 0;
    }
    /// SrcDims.size() == 1; DstDims.size() >= 1
    ArrayRef<uint64_t> getPackingSrcDims() {
      return getPackingSrcIndex() == 0 ? getFirst()
                                       : getSecond();
    }
    ArrayRef<uint64_t> getPackingDstDims() {
      return getPackingDstIndex() == 0 ? getFirst()
                                       : getSecond();
    }
  }];
}

def PackingMapArrayAttr : TypedArrayAttrBase<PackingMapAttr,
  "packing_map array attr.">;

#endif // LINALGX_DIALECT
