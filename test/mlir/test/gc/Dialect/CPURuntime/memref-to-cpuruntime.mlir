// RUN: gc-opt --split-input-file --convert-memref-to-cpuruntime %s -verify-diagnostics | FileCheck %s

func.func @alloc() {
  // CHECK-LABEL: func @alloc()
  // CHECK: %[[m0:.*]] = cpuruntime.alloc() : memref<1024xf32>
  %m0 = memref.alloc() : memref<1024xf32>
  scf.forall (%i) in (32) {
  }
  // CHECK: cpuruntime.dealloc %[[m0]] : memref<1024xf32>
  cpuruntime.dealloc %m0 : memref<1024xf32>
  return
}

func.func @thread_alloc() {
  // CHECK-LABEL: func.func @thread_alloc()
  // CHECK: %[[m0:.*]] = cpuruntime.threadAlloc() : memref<1024xf32>
  scf.forall (%i) in (32) {
    %0 = memref.alloc() : memref<1024xf32>
    // CHECK: cpuruntime.threadDealloc %[[m0]] : memref<1024xf32>
    memref.dealloc %0 : memref<1024xf32>
    }
  return
}
