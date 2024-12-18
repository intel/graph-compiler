// RUN: gc-opt %s --xevm-attach-target | FileCheck %s
module attributes {gpu.container_module} {
  //CHECK:gpu.module @entry_kernel [#xevm.target]
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: index) kernel attributes {} {
      gpu.return
    }
  }
}

