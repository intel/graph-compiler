// RUN: gc-opt %s --finalize-memref-to-llvm --convert-scf-to-cf --convert-cpuruntime-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | gc-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

module {
  func.func @doAlloc() {
    %m0 = cpuruntime.alloc () : memref<13xf32>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 42.0 : f32
    memref.store %cst, %m0[%c0] : memref<13xf32>
    %val = memref.load %m0[%c0] : memref<13xf32>
    cpuruntime.printf "main alloc stored val %f\n" %val : f32
    cpuruntime.dealloc %m0 : memref<13xf32>
    return
  }

  func.func @doThreadAlloc() {
    scf.forall (%arg2) in (3) {
      %m0 = cpuruntime.alloc threadLocal () : memref<13xf32>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 56.0 : f32
      memref.store %cst, %m0[%c0] : memref<13xf32>
      %val = memref.load %m0[%c0] : memref<13xf32>
      cpuruntime.printf "thread alloc stored val %f\n" %val : f32
      cpuruntime.dealloc %m0 : memref<13xf32>
    }
    return
  }

  func.func @main() {
    call @doAlloc() : () -> ()
    call @doThreadAlloc() : () -> ()
    return
  }
  // CHECK: stored val 42.000000
  // CHECK: thread alloc stored val 56.000000
  // CHECK: thread alloc stored val 56.000000
  // CHECK: thread alloc stored val 56.000000
}
