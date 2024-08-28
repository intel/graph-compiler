// RUN: gc-opt --split-input-file --convert-memref-to-cpuruntime %s -verify-diagnostics | FileCheck %s

func.func @alloc_dealloc_FIFO() -> memref<128xf32> {
  // CHECK-LABEL: func @alloc_dealloc_FIFO()
  // CHECK: %[[m0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: %[[m1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: %[[m2:.*]] = cpuruntime.alloc() : memref<128xf32>
  %alloc = memref.alloc() : memref<128xf32>
  %alloc_0 = memref.alloc() : memref<128xf32>
  %alloc_1 = memref.alloc() : memref<128xf32>
  // CHECK: memref.dealloc %[[m0]] : memref<128xf32>
  memref.dealloc %alloc_0 : memref<128xf32>
  // CHECK: memref.dealloc %[[m1]] : memref<128xf32>
  memref.dealloc %alloc : memref<128xf32>
  // CHECK: cpuruntime.dealloc %[[m2]] : memref<128xf32>
  return %alloc_1 : memref<128xf32>
}

func.func @alloc_dealloc_in_region() -> memref<128xf32> {
  // CHECK-LABEL: func @alloc_dealloc_FIFO()
  // CHECK: %[[m0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: %[[m1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: %[[m2:.*]] = cpuruntime.alloc() : memref<128xf32>
  %alloc = memref.alloc() : memref<128xf32>
  %alloc_0 = memref.alloc() : memref<128xf32>
  %alloc_1 = memref.alloc() : memref<128xf32>
  // CHECK: memref.dealloc %[[m0]] : memref<128xf32>
  memref.dealloc %alloc_0 : memref<128xf32>
  // CHECK: memref.dealloc %[[m1]] : memref<128xf32>
  memref.dealloc %alloc : memref<128xf32>
  // CHECK: cpuruntime.dealloc %[[m2]] : memref<128xf32>
  return %alloc_1 : memref<128xf32>
}
