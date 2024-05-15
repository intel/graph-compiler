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

def CSA : Pass<"csa"> {
  let summary = "Constant Subgraph Analysis";
  let description = [{
    This pass implements a constant subgraph analysis.
  }];
  let constructor = "mlir::gc::createCSAPass()";
}

def CST : Pass<"cst"> {
  let summary = "Constant Subgraph Transform";
  let description = [{
    This pass implements a constant subgraph transform.
  }];
  let constructor = "mlir::gc::createCSTPass()";
}

#endif // GC_DIALECT_GC_PASSES
