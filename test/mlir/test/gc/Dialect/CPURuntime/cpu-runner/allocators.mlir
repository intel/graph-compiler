// RUN: gc-opt %s --convert-cpuruntime-to-llvm --convert-func-to-llvm | gc-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

module {
  func.func @doAlloc() {
    %m0 = cpuruntime.alloc () : memref<13xf32>
    cpuruntime.dealloc %m0 : memref<13xf32>
    return
  }

  func.func @doThreadAlloc() {
    %m0 = cpuruntime.alloc () : memref<13xf32>
    cpuruntime.dealloc %m0 : memref<13xf32>
    return
  }

  func.func @main() {
    call @doAlloc() : () -> ()
    call @doThreadAlloc() : () -> ()
    return
  }
}