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

def ConvertOneDNNGraphToLinalg : Pass<"convert-onednn-graph-to-linalg"> {
  let summary =
      "Lower the operations from the oneDNN Graph dialect into Linalg";
  let description = [{Lowers the `onednn_graph` ops to `linalg` ops.}];
  let dependentDialects = [
    "func::FuncDialect", "math::MathDialect", "arith::ArithDialect",
    "tensor::TensorDialect", "linalg::LinalgDialect", "linalgx::LinalgxDialect"
  ];
}

def DeepTileContractionNamedOp
    : Pass<"deep-tile-contraction-named-op", "func::FuncOp"> {
  let summary = "Tile linalg contraction named operation deeply";
  let description =
      [{The pass tries to tile the linalg contraction named op deeply.}];
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
  ];
}

def GCCPUPipeline : Pass<"gc-cpu-pipeline"> {
  let summary = "All-in-one pipeline for GC for CPU";
  let dependentDialects = [
    "onednn_graph::OneDNNGraphDialect", "tensor::TensorDialect",
    "memref::MemRefDialect", "linalg::LinalgDialect", "linalgx::LinalgxDialect",
    "LLVM::LLVMDialect", "scf::SCFDialect",
    "bufferization::BufferizationDialect", "omp::OpenMPDialect",
    "vector::VectorDialect"
  ];
}

#endif // GC_DIALECT_GC_PASSES
