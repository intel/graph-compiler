//===- OneDNNGraphOps.td - OneDNN input dialect ops --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONEDNNGRAPH_OPS
#define ONEDNNGRAPH_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "OneDNNGraphDialect.td"
include "OneDNNGraphTypes.td"

//===----------------------------------------------------------------------===//
// Base OneDNNGraph operation definition.
//===----------------------------------------------------------------------===//

class OneDNNGraph_Op<string mnemonic, list<Trait> traits = []> :
        Op<OneDNNGraphDialect, mnemonic, traits>;

class OneDNNGraph_ElemwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    OneDNNGraph_Op<mnemonic, traits # [SameOperandsAndResultElementType, InferTensorType, 
    ResultsBroadcastableShape]> {
  let arguments = (ins OneDNNGraph_FloatTensor:$input_0, 
                       OneDNNGraph_FloatTensor:$input_1);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

class OneDNNGraph_ElemwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    OneDNNGraph_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins OneDNNGraph_FloatTensor:$operand);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// OneDNNGraph op definitions
//===----------------------------------------------------------------------===//

def OneDNNGraph_MatMulOp : 
    OneDNNGraph_Op<"matmul", [SameOperandsAndResultElementType, InferTensorTypeAdaptor]> {
  let summary = "Generalized matrix multiplication";
  let description = [{
    `https://spec.oneapi.io/onednn-graph/latest/ops/matrix/MatMul_1.html`
  }];

  let arguments = (ins OneDNNGraph_FloatTensor:$input_a, 
                       OneDNNGraph_FloatTensor:$input_b, 
                       Optional<OneDNNGraph_LogicalTensor>:$bias,
                       DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                       DefaultValuedAttr<BoolAttr, "false">:$transpose_b);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def OneDNNGraph_ReLUOp : OneDNNGraph_ElemwiseUnaryOp<"relu"> {
  let summary = "element-wise relu";
  let description = [{
    `https://spec.oneapi.io/onednn-graph/latest/ops/activation/ReLU_1.html`
  }];
}

def OneDNNGraph_AddOp : OneDNNGraph_ElemwiseBinaryOp<"add", [Commutative]> {
  let summary = "element-wise addition with multi-directional broadcast";
  let description = [{
    `https://spec.oneapi.io/onednn-graph/latest/ops/arithmetic/Add_1.html`
  }];
}

#endif // ONEDNNGRAPH_OPS