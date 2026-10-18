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

def TileGroupedMatmul : Pass<"tile-grouped-matmul", "func::FuncOp"> {
  let summary = "Tile grouped matmul operations.";
  let description = [{
    Tile 'linalgx.grouped_matmul' ops (work-group level) and the contractions
    they decompose into (sub-group level). Must run after 'tile-contract'.
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

def DropUnitDims : Pass<"drop-unit-dims", "func::FuncOp"> {
  let summary = "Drop unit dims.";
  let description = [{
    Drop unit dims to allow some operations, like vector transfers, be properly lowered to XeGPU.
  }];
  let dependentDialects = ["memref::MemRefDialect", "vector::VectorDialect"];
}

def RemoveAllocs : Pass<"remove-allocs", "func::FuncOp"> {
  let summary = "Remove unnecessry allocs.";
  let description = [{
    Remove unnecessry memref.alloc operations.
  }];
  let dependentDialects = ["gpu::GPUDialect", "memref::MemRefDialect",
                           "scf::SCFDialect"];
}

def TensorReshapeToLinalg : Pass<"tensor-reshape-to-linalg", "func::FuncOp"> {
  let summary = "Convert tensor.expand_shape/collapse_shape to linalg ops, if possible.";
  let description = [{
    Replaces tensor.expand_shape and tensor.collapse_shape with linalg.pack
    and linalg.unpack or linalg.broadcast, if possible.
  }];
}

def TensorConcatToLinalg : Pass<"tensor-concat-to-linalg", "func::FuncOp"> {
  let summary = "Convert tensor.concat to linalg operations.";
  let description = [{
    Eliminate tensor.concat by routing producers into output subviews
    or convert to linalg.copy.
  }];
}

def SetLayouts : Pass<"set-layouts"> {
  let summary = "Set XeGPU layouts.";
  let description = [{
    Sets layout attributes on anchor xegpu.* ops.
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
    Convert scf.forall operations into GPU kernels and attach the xevm target
    to the outlined kernel modules. The target chip is taken from the device
    attributes, see the GpuDeviceProps pass.
  }];
  let dependentDialects = ["scf::SCFDialect", "gpu::GPUDialect", "xevm::XeVMDialect"];
  let options = [
    Option<"cmdOptions", "cmd-options", "std::string",
           /*default=*/[{""}],
           "Command line options to pass to the downstream compiler. These are "
           "stored as the flags of the xevm target.">
    ];
}

def SetKernelDepends : Pass<"set-kernel-depends", "func::FuncOp"> {
  let summary = "Set dependencies between GPU kernels.";
  let description = [{
    Collect dependencies between scf.forall operations annotated as GPU kernels.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def MemrefCopyToGpu : Pass<"memref-copy-to-gpu", "ModuleOp"> {
  let summary = "Convert memref.copy to gpu.memcpy.";
  let dependentDialects = ["scf::SCFDialect", "gpu::GPUDialect",
                           "memref::MemRefDialect", "arith::ArithDialect"];
}

def GpuToGcGpu : Pass<"gpu-to-gc-gpu", "ModuleOp"> {
  let summary = "Convert the GPU operations to GpuRuntime calls.";
  let description = [{
    Convert the GPU alloc, dealloc, memcpy and launch operations to GpuRuntime calls.
    It changes the function name to `main` and adds additional function arguments for
    the GpuRuntime*, the external dependencies (Event* and uint32_t) and, if any
    events are returned, an Event** array. The runtime and the dependencies are
    passed into the function by the caller and the events, that were not consumed by
    GpuRuntime::wait(), are returned back to the caller via the array. The new
    function `eventNum()`, added by the Pass, returns the number of unconsumed events.
    If it's > 0, the last argument of the main function points to an array of
    `eventNum()` Event* elements, where the function stores the events.
  }];
  let options = [
    Option<"wait", "wait", "bool",
           /*default=*/"true",
           "Wait for all the events at the end of the main function instead of "
           "returning them to the caller.">
    ];
}

def PeeledForToIf : Pass<"peeled-for-to-if", "func::FuncOp"> {
  let summary = "Convert at-most-one-iteration peeled scf.for loops to scf.if.";
  let description = [{
    After loop peeling, the remainder loop has at most one iteration (trip count
    is 0 or 1). This pass detects such loops by matching the affine.apply
    pattern produced by peeling (lb = ub - (ub - orig_lb) % step) and converts
    them to scf.if, eliminating loop control-flow overhead on GPU.
  }];
}

def FoldPadRoundtrip : Pass<"fold-pad-roundtrip", "func::FuncOp"> {
  let summary = "Fold pad(extract_slice(src)) → src and pad(empty) → empty.";
  let description = [{
    Folds two patterns that arise after padding and hoisting:

    Pattern 1 — roundtrip elimination:
      %val    = tensor.extract_slice %src [0,...][d,...][1,...]
      %padded = tensor.pad %val low[0,...] high[S-d,...] → tensor<Sx...>
    Folded to %src  (valid when padded positions are not observed).

    Pattern 2 — padding of uninitialized tensor:
      %empty  = tensor.empty(%d) : tensor<?x...>
      %padded = tensor.pad %empty ... → tensor<Sx...>
    Folded to tensor.empty() : tensor<Sx...>.
  }];
}

def HoistStaticForInit : Pass<"hoist-static-for-init", "func::FuncOp"> {
  let summary = "Hoist extract_slice out of scf.for/scf.if to make results static.";
  let description = [{
    Hoists tensor.extract_slice operations out of region-based control flow
    (scf.for, scf.if) so that iter_args/results carry static types.

    For scf.for: recognizes loops where iter_args are initialized by
    linalg.fill(tensor.empty) or by extract_slice from a static tensor,
    padded to static shape inside the body, and sliced back at the yield.
    Transforms to carry static iter_args with a single extract_slice after.

    For scf.if: when all yield paths produce extract_slice from the same
    static type (zero offsets, unit strides), rewrites the if to yield
    static values and places a single extract_slice after the if.
  }];
}

def ApplyPaddingLevel : Pass<"apply-padding-level", "func::FuncOp"> {
  let summary = "Pad operations to align iteration domains to tile sizes.";
  let description = [{
    Pads operations implementing TensorMaskingOpInterface so that their
    iteration dimensions become multiples of the tile sizes specified in
    gc.tiling.wg_tile_sizes.
  }];
}

#endif // GC_DIALECT_GC_PASSES
