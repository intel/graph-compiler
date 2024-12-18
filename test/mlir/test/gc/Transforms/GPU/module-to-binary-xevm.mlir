// RUN: gc-opt %s --gpu-to-llvm --convert-gpu-to-llvm-spv --gpu-module-to-binary | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.binary @entry_kernel
  // CHECK:[#gpu.object<#xevm.target,
  gpu.module @entry_kernel [#xevm.target] {
    gpu.func @entry_kernel(%arg0: index) kernel attributes {} {
      gpu.return
    }
  }
}

