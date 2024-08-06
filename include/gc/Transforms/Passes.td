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

#ifdef GC_HAS_ONEDNN_DIALECT
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
#endif

#ifdef GC_USE_IMEX
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
#endif

def IterativeTilingAndFusion : Pass<"iterative-tiling-and-fusion",
                                        "func::FuncOp"> {
  let summary = "Iterative tiling and fusion for any tilable operation";
  let description = [{
    The pass tries to fuse any MLIR operation which can be tiled. Moreover, this pass aims to support:
      1. Matmul fusion with element-wise/reduce/broadcast ops.
      2. Pre-op and post-op fusion.
      3. Multi-consumer and multi-producer support.
      4. Multiple level of nest loops and candidates.
      5. Flexible option to control the boundary of iterative process.
      6. Cost-model to determine whether to fuse or not.

    It intends to control the granularity of fusion by `fusion-level`, E.g.
    * `0`: disable any fusion.
    * `1`:[Default] enable both producer and consumer fusion, covering any tilable operation including tensor.pack/tensor.fill/linalg.reduce etc but excluding branches forked by multiple uses.
    * `2`: `LEVEL 1` + extend to any topology including branches.
  }];
  let dependentDialects = ["func::FuncDialect", "linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];

  let options = [
    Option<"fusionLevel", "fusion-level", "int64_t",
           /*default=*/"1",
           "Control the granularity of fusion.">,
    Option<"useCostModel", "use-cost-model", "bool",
           /*default=*/"false",
           "Decide if enable cost model to control iterative fusion.">,
    ListOption<"defaultTileSize", "default-tile-size", "std::string",
           "Set default TileSize for the certain type of op, saying matmul:{32,32}">,
  ];
}

#endif // GC_DIALECT_GC_PASSES
