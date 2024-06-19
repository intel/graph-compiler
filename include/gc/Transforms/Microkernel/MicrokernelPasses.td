//===- MicrokernelPasses.td - Graph Compiler microkernel passes *- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------===//

#ifndef GC_DIALECT_MICROKERNELPASSES
#define GC_DIALECT_MICROKERNELPASSES

include "mlir/Pass/PassBase.td"

def ConvertLinalgToMicrokernel: Pass<"convert-linalg-to-microkernel", "::mlir::func::FuncOp"> {
  let summary = "Lower eligible linalg ops to microkernels";
  let description = [{
    Convert eligible linalg ops to microkernel dialects based on pattern matching.
    For example:
    ```
    scf.forall {
      linalg.fill ins(...) outs(...) -> tensor<...>
      linalg.batch_reduce_matmul ins(...) outs(...) -> tensor<...>
    }
    ```
    Will be changed into
    ```
    scf.forall {
      linalg.fill ins(...) outs(...) -> tensor<...>
      %0 = microkernel.brgemm.dispatch(...)
      microkernel.brgemm.prologue(%0)
      microkernel.brgemm(%0, ...)
      microkernel.brgemm.epilogue(%0)
    }
    ```
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "linalg::LinalgDialect",
                           "linalgx::LinalgxDialect",
                           "microkernel::MicrokernelDialect"];
}

def ConvertMicrokernelToDnnlFunc: Pass<"convert-microkernel-to-dnnl-func", "::mlir::ModuleOp"> {
  let summary = "Lower microkernel dialects to dnnl func call";
  let description = [{
    Convert microkernel dialects to runtime function call to oneDNN library.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "LLVM::LLVMDialect",
                           "microkernel::MicrokernelDialect"];
}

def EarlyDispatchMicrokernel: Pass<"early-dispatch-microkernel", "::mlir::ModuleOp"> {
  let summary = "Early dispatch microkernel during compile time";
  let description = [{
    Early dispatch microkernel during compile time.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "LLVM::LLVMDialect",
                           "microkernel::MicrokernelDialect"];
}

def MergeBranchMicrokernelContext: Pass<"merge-branch-microkernel-context", "::mlir::ModuleOp"> {
  let summary = "Find and merge identical microkernel context operations in branches into one";
  let description = [{
    Find and merge identical microkernel context operations in branches into one.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect"];
}

def MicrokernelInvariantCodeMotion: Pass<"microkernel-invariant-code-motion", "::mlir::ModuleOp"> {
  let summary = "Hoist invariant microkernel code to avoid redundant execution";
  let description = [{
    Hoist invariant microkernel code to avoid redundant execution.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "LLVM::LLVMDialect",
                           "microkernel::MicrokernelDialect"];
}

#endif // GC_DIALECT_MICROKERNELPASSES
