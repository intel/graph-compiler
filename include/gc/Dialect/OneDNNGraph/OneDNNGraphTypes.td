//===-- OneDNNGraphTypes.td - DESC -------------------------*- tablegen -*-===//
// 
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// 
//===----------------------------------------------------------------------===//

#ifndef ONEDNNGRAPH_TYPES
#define ONEDNNGRAPH_TYPES

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.td"

//===----------------------------------------------------------------------===//
// OneDNNGraph type definitions
//===----------------------------------------------------------------------===//

def OneDNNGraph_DataType : AnyTypeOf<[
                            F16,
                            BF16, 
                            F32, 
                            SI<32>, 
                            SI<8>, 
                            UI<8>]>;

def OneDNNGraph_LogicalTensor : TensorOf<[OneDNNGraph_DataType]>;

#endif // ONEDNNGRAPH_TYPES
