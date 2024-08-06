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

def GpuLegalizeModule: Pass<"gc-gpu-legalize-module", ""> {
  let summary = "Legalizes a GPU module for spirv conversion.";
  let description = [{
    Update all the nested gpu modules with an appropriate spirv target
    information that is used further down in the pipeline.
  }];
  let dependentDialects = ["gpu::GPUDialect", "spirv::SPIRVDialect"];
}

def ConvertGpuSignaturesToLLVM: Pass<"gc-gpu-signatures-to-llvm", "gpu::GPUModuleOp"> {
  let summary = "Legalize GPU kernel signatures for runtime code conversion.";
  let dependentDialects = ["gpu::GPUDialect", "memref::MemRefDialect"];
}

def GpuGenAttachTarget: Pass<"gc-attach-gen-target", ""> {
  let summary = "Attaches Gen target to a GPU module.";
  let dependentDialects = ["gpu::GPUDialect", "gen::GENDialect"];
}

def GpuAttachAddrSpaces: Pass<"gc-attach-addr-spaces", "gpu::GPUModuleOp"> {
  let summary = "Attach addr spaces to memrefs";
  let dependentDialects = ["gpu::GPUDialect", "memref::MemRefDialect"];
}

#endif // GC_DIALECT_GC_PASSES
