//===- CPURuntimeDialect.td - CPU Runtime Dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CPUPARALLEL_DIALECT
#define CPUPARALLEL_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CPURuntime dialect definition.
//===----------------------------------------------------------------------===//

def CPURuntime_Dialect : Dialect {
    let name = "cpuruntime";
    let summary = "A dialect for CPU parallel primitives.";
    let description = [{
        This dialect contains primitives for CPU runtime.
    }];
    let cppNamespace = "::mlir::cpuruntime";
}

//===----------------------------------------------------------------------===//
// CPURuntime attribute definition.
//===----------------------------------------------------------------------===//

def CPURuntime_Main
    : I32EnumAttrCase<"main", 0>;
def CPURuntime_Thread
    : I32EnumAttrCase<"thread", 1>;

def CPURuntime_PoolTypeAttr : I32EnumAttr<
    "PoolType", "An enum for cpuruntime memory pool type",
    [CPURuntime_Main, CPURuntime_Thread]> {
  let cppNamespace = "::mlir::cpuruntime";
}

//===----------------------------------------------------------------------===//
// Base cpuruntime operation definition.
//===----------------------------------------------------------------------===//

class CPURuntime_Op<string mnemonic, list<Trait> traits = []> :
        Op<CPURuntime_Dialect, mnemonic, traits>;

#endif // CPUPARALLEL_DIALECT
