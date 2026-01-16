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
    Option<"vectorWidth", "vector-width", "size_t",
           /*default=*/"16", "The number of scalar elements that can be stored in a vector register.">,
    ListOption<"sgSizes", "sg-sizes", "size_t",
               "Subgroup sizes supported by the device.">
    ];
}

def GpuTilingAndFusion : Pass<"gpu-tiling", "func::FuncOp"> {
  let summary = "GPU tiling and fusion path.";
  let description = [{
    This pass tiles linalg operations and creates two nested scf.forall loops. When converting to gpu.launch,
    the inner loop is mapped to the block sizes and the outer - to grid sizes. The tiles calculation is based
    on the GPU device properties, retrieved from the GPU device attributes. If the GPU device attributes are 
    not specified, defaults to the pass options.
  }];
  let options = [
    Option<"numEus", "num-eus", "size_t",
           /*default=*/"448",
           "Number of Execution Units.">,
    Option<"numEusPerSlice", "num-eus-per-slice", "size_t",
           /*default=*/"8",
           "Number of Execution Units per slice.">,
    Option<"numThreadsPerEu", "num-threads-per-eu", "size_t",
           /*default=*/"8",
           "Number of threads per Execution Unit.">,
    Option<"localMemSize", "local-mem-size", "size_t",
           /*default=*/"131072",
           "The size of the local memory, shared across a work-group.">,
    Option<"vectorWidth", "vector-width", "size_t",
           /*default=*/"16",
           "The number of scalar elements that can be stored in a vector.">,
    Option<"workGroupSize", "work-group-size", "size_t",
           /*default=*/"1024",
           "The maximum workgroup size.">,
    Option<"subGroupSize", "sub-group-size", "size_t",
           /*default=*/"32",
           "Subgroup size.">
    ];
}

def AddContextArg : Pass<"add-ctx-arg", "func::FuncOp"> {
  let summary = "Add a context argument.";
  let description = [{
    Add a new memref argument to the function, that could be used to pass some context.
  }];
}

def Vectorize : Pass<"vectorize", "func::FuncOp"> {
  let summary = "Vectorize structured operations.";
  let description = [{
    Vectorize structured operations that have vectorization support and apply
    vector transformation patterns.
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
