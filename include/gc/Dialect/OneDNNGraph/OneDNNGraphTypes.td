//===-- OneDNNGraphTypes.td - DESC -------------------------*- tablegen -*-===//
// 
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// 
//===----------------------------------------------------------------------===//

#ifndef ONEDNNGRAPH_TYPES
#define ONEDNNGRAPH_TYPES

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.td"

//===----------------------------------------------------------------------===//
// OneDNNGraph type definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//
def OneDNNGraph_Float : AnyTypeOf<[F32,
			                       F16,
			                       BF16]>;

//===----------------------------------------------------------------------===//
// Integer types.
//===----------------------------------------------------------------------===//

def OneDNNGraph_Int : AnyTypeOf<[SI<8>,
                                 UI<8>]>;

def OneDNNGraph_DataType : AnyTypeOf<[
                            OneDNNGraph_Float,
                            OneDNNGraph_Int
                            ]>;

def OneDNNGraph_LogicalTensor : TensorOf<[OneDNNGraph_DataType]>;
def OneDNNGraph_FloatTensor : TensorOf<[OneDNNGraph_Float]>;

//===----------------------------------------------------------------------===//
// Enum types.
//===----------------------------------------------------------------------===//

def PropertyType 
  : I32EnumAttr<"PropertyType", "OneDNN Graph Logical Tensor Property Type", 
  [
    I32EnumAttrCase<"undef", 0>,
    I32EnumAttrCase<"variable", 1>,
    I32EnumAttrCase<"constant", 2>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::onednn_graph";
}

def PropertyTypeEnum : EnumAttr<OneDNNGraphDialect, PropertyType, "property_type"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ONEDNNGRAPH_TYPES
