//===- Passes.td - Graph Compiler passes ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECT_GC_PASSES
#define GC_DIALECT_GC_PASSES

include "mlir/Pass/PassBase.td"

def GpuDeviceProps : Pass<"gpu-dev-props", "ModuleOp"> {
  let summary = "Set GPU device properties.";
  let description = [{
    Set module wide GPU device attributes.
  }];
  let options = [
    Option<"id", "id", "size_t",
           /*default=*/"0", "Device id.">,
    Option<"name", "name", "std::string",
           /*default=*/"", "Device name.">,
    Option<"arch", "arch", "std::string",
           /*default=*/"\"bmg\"", "GPU architecture.">,
    Option<"maxWgSize", "max-wg-size", "size_t",
           /*default=*/"1024", "Maximum workgroup size.">,
    ListOption<"sgSizes", "sg-sizes", "size_t",
               "Subgroup sizes supported by the device.">
    ];
}

def TileContraction : Pass<"tile-contract", "func::FuncOp"> {
  let summary = "Tile contraction operations.";
  let description = [{
    Tile contraction operations and fuse their producers.
  }];
}

def TileAttention : Pass<"tile-attention", "func::FuncOp"> {
  let summary = "Tile attention operations.";
  let description = [{
    Tile 'linalgx.attention' ops.
  }];
}

def TileParallel : Pass<"tile-parallel", "func::FuncOp"> {
  let summary = "Tile parallel operations.";
  let description = [{
    Tile parallel operations and fuse their producers.
  }];
}

def AddContextArg : Pass<"add-ctx-arg", "func::FuncOp"> {
  let summary = "Add a context argument.";
  let description = [{
    Add a new memref argument to the function, that could be used to pass some context.
  }];
}

def Decomposition : Pass<"decomposition", "func::FuncOp"> {
  let summary = "Decompose complex operations.";
  let description = [{
    Decompose complex operations into simpler ones for easier optimization and code generation.
  }];
}

def SetGpuFastMath : Pass<"set-gpu-fast-math"> {
  let summary = "Set fastmath attribute on math.exp ops in GPU modules.";
  let description = [{
    Walks GPU modules and sets the fastmath=fast attribute on all math.exp
    operations.
  }];
}

def SetAttentionLayouts : Pass<"set-attention-layouts"> {
  let summary = "Set XeGPU layouts on attention-pattern dpas and store ops.";
  let description = [{
    Walks the module to find scf.for loops containing
    at least two xegpu.dpas ops, and sets layout attributes on those dpas ops
    and on any downstream xegpu.store_nd ops.
  }];
}

def SetAttentionPrefetch : Pass<"set-attention-prefetch"> {
  let summary = "Set XeGPU prefetches on attention-pattern dpas and store ops.";
  let description = [{
    Walks the module to find scf.for loops containing
    at least two xegpu.dpas ops, and sets prefetch attributes on those dpas ops
    and on any downstream xegpu.store_nd ops.
  }];
}

def HoistAttentionVLoad : Pass<"hoist-attention-v-load"> {
  let summary = "Hoist V-load to the beginning of flash-attention loop.";
  let description = [{
    Experimental pass that matches flash-attention patterns at the XeGPU level
    and moves the V-load (xegpu.load_nd before the second xegpu.dpas) to right
    after the K-load at the beginning of the scf.for loop.
  }];
}

def Vectorize : Pass<"vectorize", "func::FuncOp"> {
  let summary = "Vectorize structured operations.";
  let description = [{
    Vectorize structured operations that have vectorization support and apply
    vector transformation patterns.
  }];
}

def RemoveAllocs : Pass<"remove-allocs", "func::FuncOp"> {
  let summary = "Remove unnecessry allocs.";
  let description = [{
    Remove unnecessry memref.alloc operations.
  }];
}

def HoistForLoopTransferRead : Pass<"hoist-for-loop-transfer-read", "func::FuncOp"> {
  let summary = "Hoist transfer read/write operations out of for loops.";
  let description = [{
    Hoist transfer read/write operations out of for loops to improve memory access patterns.
  }];
}

def GpuKernelOutline : Pass<"gpu-kernel-outline", "ModuleOp"> {
  let summary = "Outline GPU kernels.";
  let description = [{
    Convert scf.forall operations into GPU kernels.
  }];
  let dependentDialects = ["scf::SCFDialect", "gpu::GPUDialect", "xevm::XeVMDialect"];
}

def GpuToGpuOcl : Pass<"gpu-to-gpuocl", "ModuleOp"> {
  let summary = "Convert the GPU operations to GpuOclRuntime calls.";
  let description = [{
    Convert the gpu alloc, dealloc, memcpy and launch operations to GpuOclRuntime calls.
  }];
  let options = [
    Option<"callFinish", "call-finish", "bool",
           /*default=*/"false",
           "Call finish() after each kernel launch.">
    ];
}

#endif // GC_DIALECT_GC_PASSES
