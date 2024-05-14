//===- CPURuntimeOps.td - CPU Runtime Ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CPURUNTIME_OPS
#define CPURUNTIME_OPS

include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"


def CPURuntime_AtParallelExitOp : CPURuntime_Op<"at_parallel_exit", [
       ParentOneOf<["scf::ForallOp", "scf::ParallelOp", "omp::WsloopOp", "memref::AllocaScopeOp"]>,
       SingleBlockImplicitTerminator<"ParallelExitReturnOp">
      ]> {
  let summary = "Runs the block once in all threads at the exit of the parallel section";
  let description = [{
    It executes the block for each thread working in the parallel section for
    once, at the exit of parallel section.
  }];

  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;

  // The default builder does not add a region with an empty body, add our own.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins)>,
  ];
}

def CPURuntime_ParallelExitReturnOp : CPURuntime_Op<"parallel_exit.return", [
       Pure,
       HasParent<"AtParallelExitOp">,
       Terminator, ReturnLike
      ]> {
  let summary = "Terminates at_parallel_exit block";
  let description = [{
    at_parallel_exit should ends with parallel_exit.return
  }];
  let assemblyFormat =
      [{  attr-dict }];
}


def CPURuntime_PrintfOp : CPURuntime_Op<"printf", [MemoryEffects<[MemWrite]>]>,
  Arguments<(ins StrAttr:$format,
                Variadic<AnyTypeOf<[AnyInteger, Index, AnyFloat]>>:$args)> {
  let summary = "C-style printf";
  let description = [{
    `cpuruntime.printf` takes a literal format string `format` and an arbitrary number of
    scalar arguments that should be printed.

    The format string is a C-style printf string, subject to any restrictions
    imposed by one's target platform.
  }];
  let assemblyFormat = [{
    $format attr-dict ($args^ `:` type($args))?
  }];
}


#endif // CPURUNTIME_OPS
