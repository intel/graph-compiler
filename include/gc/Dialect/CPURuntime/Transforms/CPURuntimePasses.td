//===- CPURuntimePasses.td - CPU Runtime Passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CPURUNTIME_PASS
#define CPURUNTIME_PASS

include "mlir/Pass/PassBase.td"


def CPURuntimeAtExitToOmp: Pass<"cpuruntime-atexit-to-omp", "::mlir::func::FuncOp"> {
  let summary = "Lower at_parallel_exit to code in omp.parallel section";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    omp.parallel {
      omp.wsloop for (%arg1, %arg2) : index = (%c0, %c0) to (%c1, %c512) step (%c1, %c1) {
        memref.alloca_scope  {
          cpuruntime.at_parallel_exit {
            "your.op"()
            cpuruntime.parallel_exit.return
          }
        }
        omp.yield
      }
      omp.terminator
    }
    ```
    Will be changed into
    ```
    omp.parallel {
      omp.wsloop for  (%arg1, %arg2) : index = (%c0, %c0) to (%c1, %c512) step (%c1, %c1) {
        memref.alloca_scope  {
        }
        omp.yield
      }
      "your.op"()
      omp.terminator
    }
    ```
  }];
}


def CPURuntimeToLLVM: Pass<"convert-cpuruntime-to-llvm"> {
  let summary = "Convert cpuruntime to LLVM dialect";
  let description = [{
    This pass converts supported cpuruntime ops to LLVM dialect instructions.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // CPURUNTIME_PASS
