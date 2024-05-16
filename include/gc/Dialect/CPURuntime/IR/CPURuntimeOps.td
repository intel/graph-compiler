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
include "mlir/Interfaces/SideEffectInterfaces.td"


def CPURuntime_PrintfOp : CPURuntime_Op<"printf", [MemoryEffects<[MemWrite]>]>,
  Arguments<(ins StrAttr:$format,
                Variadic<AnyTypeOf<[AnyInteger, Index, AnyFloat]>>:$args)> {
  let summary = "C-style printf";
  let description = [{
    `cpuruntime.printf` takes a literal format string `format` and an arbitrary number of
    scalar arguments that should be printed.

    The format string is a C-style printf string, subject to any restrictions
    imposed by the target platform.
  }];
  let assemblyFormat = [{
    $format attr-dict ($args^ `:` type($args))?
  }];
}


#endif // CPURUNTIME_OPS
