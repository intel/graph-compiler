// RUN: gc-opt -allow-unregistered-dialect %s | FileCheck %s
func.func @alloc() {
    // CHECK-LABEL: func @alloc()

    // CHECK: %[[m0:.*]] = cpuruntime.alloc () : memref<13xf32>
    %m0 = cpuruntime.alloc () : memref<13xf32>
    // CHECK: cpuruntime.dealloc %[[m0]] : memref<13xf32, 1>
    cpuruntime.dealloc %m0 : memref<13xf32>
    return
  }