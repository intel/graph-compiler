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

def LowerToTileVector : Pass<"lower-to-tile-vector"> {
  let summary = "Lower tensor to tile vector.";
  let description = [{
    Lower tensor to tile vector form.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::math::MathDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def CPUPhysicalRegisterPass : Pass<"CPU-physical-register-pass", "func::FuncOp"> {
  let summary = "Lower operation to cpu pysical register size.";
  let description = [{
    Physical register size lowering pass.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::math::MathDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // GC_DIALECT_GC_PASSES
