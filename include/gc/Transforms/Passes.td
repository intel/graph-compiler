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
      "linalg::LinalgDialect",
      "linalgx::LinalgxDialect"
  ];
}


def LinalgToXeGPU : Pass<"linalg-to-xegpu", "func::FuncOp"> {
  let summary = "Convert linalg dialect to XeGPU dialect.";
  let description = [{
    Lower linalg ops to XeGPU dialect.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "gpu::GPUDialect",
                           "xegpu::XeGPUDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect",
                           "arith::ArithDialect",
                           "math::MathDialect",
                           "vector::VectorDialect"];
  let options = [
    Option<"kTile", "k-tile", "int64_t",
           /*default=*/"32",
           "GEMM tile size for reduction dimension.">,
    Option<"stages", "stages", "int64_t",
           /*default=*/"1",
           "Number of cooperative prefetch stages.">,
    ListOption<"dpasTile", "dpas-tile", "int64_t",
               "DPAS register block sizes MxNxK">,
  ];
}

def TileConsumerAndFuseProducers : Pass<"tile-consumer-and-fuse-producers",
                                        "func::FuncOp"> {
  let summary = "Tile consumers and fuse producers";
  let description = [{
    The pass uses `TileConsumerAndFuseProducersUsingSCFForOp` to tile the
    consumer and fuse the consumer with the producers. The fusion anchor to matmul
    or conv-like patterns allows two additional options to control how many
    producers fuse together with the latched operation and how many consumers.
    Precisely, `max-depth` controls how many producers should be considered, while
    `start-from-last-consumer` allows to move the anchor point to the last fusable
    consumer of the conv or matmul-like pattern.
  }];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"maxDepth", "max-depth", "int64_t", "5",
           "Get producers till maxDepth">,
    Option<"numIters", "num-iters", "int64_t", "3",
           "Run fusion for the given number of iterations">,
    Option<"useForAll", "use-for-all", "bool", "true", "Use parallel forAll">,
    Option<"minTileFactor", "min-tile-factor", "int64_t", "2",
           "Minimum factor between dimension size and a tile size">,
    Option<"keepNamedOp", "keep-named-op", "bool", "false", "Don't generalize named ops">
  ];
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}

#endif // GC_DIALECT_GC_PASSES
