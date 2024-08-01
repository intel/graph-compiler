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

def ConvertMemRefToCPURuntime : Pass<"convert-memref-to-cpuruntime"> {
  let summary = "Lower the allocation / deallocation operations from the MemRef dialect into CPURuntime";
  let description = [{
    Lowers the `memref` allocation / deallocation ops to `CPURuntime` ops.
  }];
  let dependentDialects = [
      "memref::MemRefDialect",
      "cpuruntime::CPURuntimeDialect"
  ];
}

#ifdef GC_USE_GPU
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

#endif // GC_DIALECT_GC_PASSES
