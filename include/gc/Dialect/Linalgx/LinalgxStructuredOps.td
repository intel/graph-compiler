//===- LinalgxStructuredOps.td - linalgx dialect ops -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_STRUCTURED_OPS
#define LINALGX_STRUCTURED_OPS

include "LinalgxDialect.td"

include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/Dialect/Linalg/IR/LinalgEnums.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on ShapedType as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgxStructuredBase_Op<string mnemonic, list<Trait> props>
  : Op<LinalgxDialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<mlir::linalg::IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<mlir::linalg::LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  }];
}

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

//===----------------------------------------------------------------------===//
// Structured linalgx operation definition
//===----------------------------------------------------------------------===//

def Linalgx_SigmoidOp : LinalgxStructuredBase_Op<"sigmoid", 
      [AttrSizedOperandSegments]> {
  let summary = [{Applies sigmoid(x) elementwise.}];
  let description = [{No numeric casting is performed on the input operand.}];
  let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$outputs);
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<
      (ins 
        "TypeRange":$resultTensorTypes, 
        "ValueRange":$inputs,
        "ValueRange":$outputs,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, SigmoidOp::getRegionBuilder());
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static unsigned getNumRegionArgs() { return 2; }
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }
  }];
}

def Linalgx_Mm2DVnniOp 
    : LinalgxStructuredBase_Op<"mm2d_vnni", [AttrSizedOperandSegments]> {
  let summary = "Transposed matmul with 2d input and vnni packed weights";
  let description = [{
    Supported format: A[M, K] * B[N0, K0, k, n, v] -> C[M, N], with:
      N = N0 * n
      K = K0 * k * v; v = (2, 4)
  }];
  let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$outputs);
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<
      (ins 
        "TypeRange":$resultTensorTypes, 
        "ValueRange":$inputs,
        "ValueRange":$outputs,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Mm2DVnniOp::getRegionBuilder());
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static unsigned getNumRegionArgs() { return 3; }
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }
  }];
}

def Linalgx_Mm4DVnniOp 
    : LinalgxStructuredBase_Op<"mm4d_vnni", [AttrSizedOperandSegments]> {
  let summary = "Transposed matmul with 4d blocking input and vnni packed weights";
  let description = [{
    Supported format: A[M, K, m, k] * B[N, K, k0, n, v] -> C[M, N, m, n], with:
      k = k0 * v; v = (2, 4)
  }];
  let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$outputs);
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<
      (ins 
        "TypeRange":$resultTensorTypes, 
        "ValueRange":$inputs,
        "ValueRange":$outputs,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, Mm4DVnniOp::getRegionBuilder());
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static unsigned getNumRegionArgs() { return 3; }
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }
  }];
}

def Linalgx_PackedMatmulOp 
    : LinalgxStructuredBase_Op<"packed_matmul", [AttrSizedOperandSegments]> {
  let summary = "matmul with packed data format";
  let description = [{
    Use m_packing, n_packing and k_packing to define relation shape between C[M, N] = A[M, K] * B[K, N].
  }];
  let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$outputs,
      PackingMapArrayAttr:$m_packing,
      PackingMapArrayAttr:$n_packing,
      PackingMapArrayAttr:$k_packing
      );
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<
      (ins 
        "TypeRange":$resultTensorTypes, 
        "ValueRange":$inputs,
        "ValueRange":$outputs,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, PackedMatmulOp::getRegionBuilder());
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static unsigned getNumRegionArgs() { return 3; }
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }
  }];
}

def Linalgx_BatchReduceMatmulVnniOp 
    : LinalgxStructuredBase_Op<"batch_reduce_matmul_vnni", [AttrSizedOperandSegments]> {
  let summary = "Batch reduced matmul with 3d batch input and vnni packed weights";
  let description = [{
    Supported format: A[B, M, K] * B[B, k, N, v] -> C[M, N], with:
      K = k * v; v = (2, 4)
  }];
  let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$outputs);
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<
      (ins 
        "TypeRange":$resultTensorTypes, 
        "ValueRange":$inputs,
        "ValueRange":$outputs,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, BatchReduceMatmulVnniOp::getRegionBuilder());
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static unsigned getNumRegionArgs() { return 3; }
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }
  }];
}

def Linalgx_MultiBatchMatmulOp : LinalgxStructuredBase_Op<"multi_batch_matmul", 
      [AttrSizedOperandSegments, LinalgContractionOpInterface]> {
  let summary = "Batch matmul with variable batch dims";
  let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$outputs);
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<
      (ins 
        "TypeRange":$resultTensorTypes, 
        "ValueRange":$inputs,
        "ValueRange":$outputs,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, MultiBatchMatmulOp::getRegionBuilder());
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
    static unsigned getNumRegionArgs() { return 3; }
    std::string getLibraryCallName() {
      return "op_has_no_registered_library_name";
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }
  }];
}

#endif // LINALGX_STRUCTURED_OPS