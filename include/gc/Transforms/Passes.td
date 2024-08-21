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

#ifdef GC_HAS_ONEDNN_DIALECT
def ConvertOneDNNGraphToLinalg : Pass<"convert-onednn-graph-to-linalg"> {
  let summary =
      "Lower the operations from the oneDNN Graph dialect into Linalg";
  let description = [{Lowers the `onednn_graph` ops to `linalg` ops.}];
  let dependentDialects = [
    "func::FuncDialect", "math::MathDialect", "arith::ArithDialect",
    "tensor::TensorDialect", "linalg::LinalgDialect", "linalgx::LinalgxDialect"
  ];
}
#endif

def ConvertMemRefToCPURuntime : Pass<"convert-memref-to-cpuruntime", "func::FuncOp"> {
  let summary = "Lower the allocation / deallocation operations from the MemRef dialect into CPURuntime";
  let description = [{
    Lowers the `memref` allocation / deallocation ops to `CPURuntime` ops.
  }];
  let dependentDialects = [
      "memref::MemRefDialect",
      "cpuruntime::CPURuntimeDialect"
  ];
}

#ifdef GC_USE_IMEX
def LinalgToXeGPU : Pass<"linalg-to-xegpu", "func::FuncOp"> {
  let summary = "Convert linalg dialect to XeGPU dialect.";
  let description = [{Lower linalg ops to XeGPU dialect.}];
  let dependentDialects = [
    "linalg::LinalgDialect", "gpu::GPUDialect", "xegpu::XeGPUDialect",
    "scf::SCFDialect", "memref::MemRefDialect", "arith::ArithDialect",
    "math::MathDialect", "vector::VectorDialect"
  ];
  let options = [
    Option<"kTile", "k-tile", "int64_t",
           /*default=*/"32", "GEMM tile size for reduction dimension.">,
    Option<"stages", "stages", "int64_t",
           /*default=*/"1", "Number of cooperative prefetch stages.">,
    ListOption<"dpasTile", "dpas-tile", "int64_t",
               "DPAS register block sizes MxNxK">,
  ];
}
#endif // GC_USE_IMEX

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
      6. Default tiling when no op is tiled before fusion.
      7. Cost-model to determine whether to fuse or not.
  }];
  let dependentDialects = ["func::FuncDialect", "linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];

  let options = [
    Option<"useCostModel", "use-cost-model", "bool",
           /*default=*/"false",
           "Decide if enable cost model to control iterative fusion.">,
    ListOption<"defaultTileSize", "default-tile-size", "std::string",
           "Set default TileSize for the certain type of op, saying `matmul:{32,32}`">,
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

def VerifyTargetDescription : Pass<"verify-target-description", "ModuleOp"> {
  let summary = "Verify the target description from ModuleOp DLTI attribute.";
  let description = [{
    Verify the target description from ModuleOp DLTI attribute. Raise error for unexpected input(such as a negative number of num_threads), and raise warn for missing fields, and provide a default value(such as 32K for L1_cache_size).
  }];
  let dependentDialects = ["DLTIDialect"];
  let options = [
    Option<"device", "device", "std::string",
           /*default=*/"\"CPU\"",
           "The device to verify. Supported device: CPU, ">,
  ];
}

def SinkOpIntoInnerLoop : Pass<"sink-op-into-inner-loop"> {
  let summary = "Sink operations into inner loops";
  let description = [{The pass tries to sink operations into inner loops as deep as possible to maximize the chance for outer loop optimization.
  }];
  let dependentDialects = [];
}

def MergeNestedForall : Pass<"merge-nested-forall"> {
  let summary = "Merge nested scf.forall operations";
  let description = [{The pass tries to merge nested forall operations.}];
  let dependentDialects = ["scf::SCFDialect"];
}

def FoldTensorOperation : Pass<"fold-tensor-operation"> {
  let summary = "Fold some tensor operation";
  let description = [{
    Remove some useless tensor operations.
  }];
  let dependentDialects = [
    "::mlir::tensor::TensorDialect"
  ];
}

def LowerToTileVector : Pass<"lower-to-tile-vector", "func::FuncOp"> {
  let summary = "Lower tensor to tile (virtual) vector";
  let description = [{
    Lower operation operate on tensor to vector operation.
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

#endif // GC_DIALECT_GC_PASSES
