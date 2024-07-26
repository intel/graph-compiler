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

def CPURuntime_AllocOp : CPURuntime_Op<"alloc", [
    AttrSizedOperandSegments,
    MemoryEffects<[MemAlloc<DefaultResource>]>
  ]> {

  let summary = "CPURuntime memory allocation operation in a stack way.";
  let description = [{
    The `cpuruntime.alloc` operation allocates a region of memory on the CPU. It is
    similar to the `std.alloc` op.

    Example:

    ```mlir
    // For dynamic size memref
    %memref = cpuruntime.alloc (%width) : memref<64x?xf32>

    // For static size memref
    %memref = cpuruntime.alloc () : memref<64x32xf32>
    // For memref with symbolic operand in affine map
    %memref = cpuruntime.alloc [%symbol_value] (%width) : memref<64x?xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
    ```
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes, Variadic<Index>:$symbolOperands);
  let results = (outs Res<AnyMemRef, "", [MemAlloc<DefaultResource, 0, FullEffect>]>:$memref);

  let assemblyFormat = [{
    `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];
}

def CPURuntime_ThreadAllocOp : CPURuntime_Op<"threadAlloc", [
    AttrSizedOperandSegments,
    MemoryEffects<[MemAlloc<DefaultResource>]>
  ]> {

  let summary = "CPURuntime memory thread local allocation operation in a stack way.";
  let description = [{
    The `cpuruntime.threadAlloc` operation allocates a region of memory on the CPU local thread. It is
    similar to the `cpuruntime.alloc` op.
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes, Variadic<Index>:$symbolOperands);
  let results = (outs Res<AnyMemRef, "", [MemAlloc<DefaultResource, 0, FullEffect>]>:$memref);

  let assemblyFormat = [{
    `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];
}

def CPURuntime_DeallocOp : CPURuntime_Op<"dealloc", [
    MemoryEffects<[MemFree]>
  ]> {

  let summary = "CPU memory deallocation operation";

  let description = [{
    The `cpuruntime.dealloc` operation frees the region of memory referenced by a
    memref which was originally created by the `cpuruntime.alloc` operation.
    It is similar to the `std.dealloc` op.
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemFree]>:$memref);


  let assemblyFormat = [{ $memref attr-dict `:` type($memref) }];
}

def CPURuntime_ThreadDeallocOp : CPURuntime_Op<"threadDealloc", [
    MemoryEffects<[MemFree]>
  ]> {

  let summary = "CPU thread local memory deallocation operation";

  let description = [{
    The `cpuruntime.threadDealloc` operation frees the region of memory referenced by a
    memref which was originally created by the `cpuruntime.threadAlloc` operation.
    It is similar to the `cpuruntime.dealloc` op.
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemFree]>:$memref);


  let assemblyFormat = [{ $memref attr-dict `:` type($memref) }];
}

#endif // CPURUNTIME_OPS
