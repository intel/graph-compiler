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

def CPURuntimeToLLVM: Pass<"convert-cpuruntime-to-llvm"> {
  let summary = "Convert cpuruntime to LLVM dialect";
  let description = [{
    This pass converts supported cpuruntime ops to LLVM dialect instructions.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // CPURUNTIME_PASS
