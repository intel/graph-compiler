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

def LegalizeDTypeToF32 : Pass<"legalizedtype-to-f32"> {
  let summary = "Legalize floating-point math ops on low-precision floats";
  let description = [{
    On many targets, the math functions are not implemented for floating-point
    types less precise than IEEE single-precision (aka f32), such as half-floats
    or bfloat16.

    This pass explicitly legalizes these math functions by inserting
    `arith.extf` and `arith.truncf` pairs around said op, which preserves
    the original semantics while enabling lowering.

    This pass awares target machine flags. If fAVX512FP16 is true, then some ops
    has no need to do the legalization.
  }];
  let dependentDialects = ["mlir::math::MathDialect", "mlir::arith::ArithDialect"];
}

#endif // GC_DIALECT_GC_PASSES
