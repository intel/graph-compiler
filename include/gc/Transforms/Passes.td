//===- Passes.td - Graph Compiler passes -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECT_GC_PASSES
#define GC_DIALECT_GC_PASSES

include "mlir/Pass/PassBase.td"

def TileLinalgNamed : Pass<"tile-named-linalg", "func::FuncOp"> {
  let summary = "Tile linalg named operations.";
  let dependentDialects =
      ["linalg::LinalgDialect", "scf::SCFDialect", "tensor::TensorDialect"];
}

def ConvertOneDNNGraphToLinalg : Pass<"convert-onednn-graph-to-linalg"> {
  let summary = "Lower the operations from the oneDNN Graph dialect into Linalg";
  let description = [{
    Lowers the `onednn_graph` ops to `linalg` ops.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "math::MathDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "linalg::LinalgDialect"
  ];
}

def ConstantSubgraphAnalysis : Pass<"constant-subgraph-analysis"> {
  let summary = "Constant Subgraph Analysis";
  let description = [{
    This pass implements a constant subgraph analysis.
  }];
  let constructor = "mlir::gc::createConstantSubgraphAnalysisPass()";
}

def ConstantTensorFolding : Pass<"constant-tensor-folding"> {
  let summary = "Constant Tensor Folding Transform";
  let description = [{
    This pass implements a constant tensor folding transform.
  }];
  let constructor = "mlir::gc::createConstantTensorFoldingPass()";
  let dependentDialects = [
      "tensor::TensorDialect",
      "linalg::LinalgDialect",
      "LLVM::LLVMDialect"];
}

#endif // GC_DIALECT_GC_PASSES
