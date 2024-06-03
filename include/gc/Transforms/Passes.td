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

def MergeAlloc : Pass<"merge-alloc", "func::FuncOp">  {
  let summary = "Merge multiple memref.alloc and reuse the buffer";
  let description = [{
    The pass merges the "mergeable" memref.alloc allocations into a single
    memref.alloc in its ancestor "allocation scope", to enhance memory
    reuse and cache locality. A memref.alloc is "mergeable" if it is owned
    by the current function and it is statically shaped and has identity layout.
    An "allocation scope" is the nearest ancestor surrounding operation
    of memref.alloc, which extends AutomaticAllocationScope trait and is not
    scf.for. The function top-level block or the body of parallel-loop are
    examples of "allocation scope". If there are nested AutomaticAllocationScope,
    each level of the AutomaticAllocationScope is a different "allocation scope".
    A "mergeable" memref.alloc will be replaced by a memref.view on the "merged"
    buffer, with an offset. The "merged" buffer will be located at the begining
    of the block of the "allocation scope".
    The offset of each merged buffer is decided by this pass, by considering the
    lifetime of the original memref before merging. This pass schedules the
    offsets to 1) make sure the offsets and address ranges do not overlap if
    two "mergeable" allocations have overlapped lifetime, and 2) reuse the
    address ranges that are considered "hot" in cache for an later allocation. 
  }];
  let options = [
    Option<"optionCheck", "check", "bool",
       /*default=*/"false",
       "Skip the mutation of the IR and only mark the lifetime and scope on the"
       " operations. Useful for debugging and testing.">,
    Option<"optionNoLocality", "no-consider-locality", "bool",
       /*default=*/"false",
       "Don't consider the cache locality when reusing the buffers. "
       "This option may result in smaller total memory usage.">,
  ];
  let dependentDialects = ["memref::MemRefDialect", "arith::ArithDialect"];
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

def GCCPUPipeline: Pass<"gc-cpu-pipeline"> {
  let summary = "All-in-one pipeline for GC for CPU";
  let dependentDialects = ["onednn_graph::OneDNNGraphDialect",
      "tensor::TensorDialect",
      "memref::MemRefDialect",
      "linalg::LinalgDialect",
      "LLVM::LLVMDialect",
      "scf::SCFDialect",
      "bufferization::BufferizationDialect",
      "omp::OpenMPDialect",
      "vector::VectorDialect"];
}

#endif // GC_DIALECT_GC_PASSES
