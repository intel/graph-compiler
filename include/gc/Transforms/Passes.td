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
    "vector::VectorDialect", "microkernel::MicrokernelDialect"
  ];
}

def LinalgLowerToLoop : Pass<"linalg-lower-to-loop", "ModuleOp"> {
  let summary = "Lower Linalg ops to loops(scf.for is inside parallel loop, "
                "otherwise scf.parallel)";
  let dependentDialects = [
    "arith::ArithDialect", "func::FuncDialect", "memref::MemRefDialect",
    "scf::SCFDialect", "LLVM::LLVMDialect", "linalg::LinalgDialect"
  ];
}
def AnyTilableFusion : Pass<"any-tilable-fusion",
                                        "func::FuncOp"> {
  let summary = "Fusion for any tilable operation";
  let description = [{
    The pass tries to fuse any MLIR operation which can be tiled. Moreover, this pass aims to support:
      1. Matmul fusion with element-wise/reduce/broadcast ops.
      2. Pre-op and post-op fusion.
      3. Multi-consumer and multi-producer support.
      4. Cost-model to determine whether to fuse or not.
      5. TBC

    It intends to control the granularity of fusion by `fusion-level`, E.g.
    * `0`: disable any fusion.
    * `1`:[Default] enable pre-op fusion + post-op fusion covering any tilable operation including tensor.pack/tensor.fill/linalg.reduce etc but excluding branches forked by multiple uses.
    * `2`: `LEVEL 1` + extend to any topology including branches.
    * `3`: `LEVEL 2` + support coarse-grained fusion.
  }];
  let dependentDialects = ["func::FuncDialect", "linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}

#endif // GC_DIALECT_GC_PASSES
