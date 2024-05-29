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
include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.td"
include "gc/Dialect/OneDNNGraph/OneDNNGraphTypes.td"

//===----------------------------------------------------------------------===//
// Base OneDNNGraph operation definition.
//===----------------------------------------------------------------------===//

class OneDNNGraph_Op<string mnemonic, list<Trait> traits = []> :
        Op<OneDNNGraphDialect, mnemonic, traits>;

class OneDNNGraph_ElemwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    OneDNNGraph_Op<mnemonic, traits # 
    [SameOperandsAndResultType]> {
  let arguments = (ins OneDNNGraph_FloatTensor:$operand);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

class OneDNNGraph_ElemwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    OneDNNGraph_Op<mnemonic, traits # 
    [SameOperandsAndResultElementType, InferTensorTypeAdaptor, ResultsBroadcastableShape]> {
  let arguments = (ins OneDNNGraph_FloatTensor:$input_a, 
                       OneDNNGraph_FloatTensor:$input_b,
                       DefaultValuedOptionalAttr<BoolAttr, "true">:$auto_broadcast);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

class OneDNNGraph_ReduceOp<string mnemonic, list<Trait> traits = []> :
    OneDNNGraph_Op<mnemonic, traits # 
    [SameOperandsAndResultElementType, InferTensorTypeAdaptor]> {
  let arguments = (ins OneDNNGraph_FloatTensor:$operand,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$axes,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// OneDNNGraph op definitions
//===----------------------------------------------------------------------===//

// Matmul

def OneDNNGraph_MatMulOp : 
    OneDNNGraph_Op<"matmul", 
    [SameOperandsAndResultElementType, InferTensorTypeAdaptor]> {
  let summary = [{
    MatMul operation computes the product of two tensors with optional bias addition.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_matmul.html`
  }];

  let arguments = (ins OneDNNGraph_FloatTensor:$input_a, 
                       OneDNNGraph_FloatTensor:$input_b, 
                       Optional<OneDNNGraph_FloatTensor>:$bias,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$transpose_a,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$transpose_b);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

// Common Unary

def OneDNNGraph_ReLUOp : OneDNNGraph_ElemwiseUnaryOp<"relu"> {
  let summary = "element-wise relu";
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_relu.html`
  }];
}

def OneDNNGraph_SigmoidOp : OneDNNGraph_ElemwiseUnaryOp<"sigmoid"> {
  let summary = "element-wise sigmoid";
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_sigmoid.html`
  }];
}

// Special Unary

def OneDNNGraph_TypeCastOp : OneDNNGraph_Op<"type_cast", [SameOperandsAndResultShape]> {
  let summary = [{
    TypeCast operation performs element-wise cast from input data type 
    to the data type given by output tensor.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_typecast.html`
  }];

  let arguments = (ins OneDNNGraph_FloatTensor:$operand);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def OneDNNGraph_PowOp : OneDNNGraph_Op<"pow", [SameOperandsAndResultType]> {
  let summary = [{
    Pow operation performs an element-wise power operation on a given input
     tensor with a single value attribute beta as its exponent.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_pow.html`
  }];

  let arguments = (ins OneDNNGraph_FloatTensor:$operand, 
                       F32Attr:$beta);
  let results = (outs OneDNNGraph_FloatTensor:$result);

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

// Common Binary

def OneDNNGraph_AddOp : OneDNNGraph_ElemwiseBinaryOp<"add", [Commutative]> {
  let summary = [{
    Add operation performs element-wise addition operation with two 
    given tensors applying multi-directional broadcast rules.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_add.html`
  }];
}

def OneDNNGraph_MulOp : OneDNNGraph_ElemwiseBinaryOp<"mul", [Commutative]> {
  let summary = [{
    Multiply operation performs element-wise multiply operation with two 
    given tensors applying multi-directional broadcast rules.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_multiply.html`
  }];
}

def OneDNNGraph_SubOp : OneDNNGraph_ElemwiseBinaryOp<"sub"> {
  let summary = [{
    Subtract operation performs element-wise subtraction operation with 
    two given tensors applying multi-directional broadcast rules.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_subtract.html`
  }];
}

def OneDNNGraph_DivOp : OneDNNGraph_ElemwiseBinaryOp<"div"> {
  let summary = [{
    Divide operation performs element-wise division operation with two 
    given tensors applying multi-directional broadcast rules.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_divide.html`
  }];
}

// Common Reduce

def OneDNNGraph_ReduceSumOp : OneDNNGraph_ReduceOp<"reduce_sum"> {
  let summary = [{
    ReduceSum operation performs the reduction with addition on a given 
    src data along dimensions specified by axes.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_reducesum.html`
  }];
}

def OneDNNGraph_ReduceMeanOp : OneDNNGraph_ReduceOp<"reduce_mean"> {
  let summary = [{
    ReduceMean operation performs the reduction with finding the arithmetic 
    mean on a given src data along dimensions specified by axes.
  }];
  let description = [{
    `https://oneapi-src.github.io/oneDNN/dev_guide_op_reducemean.html`
  }];
}

#endif // ONEDNNGRAPH_OPS