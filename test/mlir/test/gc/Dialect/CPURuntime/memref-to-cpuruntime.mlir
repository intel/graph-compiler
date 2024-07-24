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

func.func @return_alloc() -> memref<32x18xf32> {
  // CHECK-LABEL: func @return_alloc() -> memref<32x18xf32>
  // CHECK: %[[m0:.*]] = memref.alloc() : memref<32x18xf32>
  %0 = memref.alloc() : memref<32x18xf32>
  return %0 : memref<32x18xf32>
}

func.func @yield_alloc() -> memref<32x18xf32> {
  // CHECK-LABEL: func @yield_alloc() -> memref<32x18xf32>
  // CHECK: %[[m0:.*]] = memref.alloc() : memref<32x18xf32>
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %lastBuffer = memref.alloc() : memref<32x18xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg1 = %lastBuffer) -> (memref<32x18xf32>) {
    %newBuffer = memref.alloc() : memref<32x18xf32>
    memref.dealloc %arg1 : memref<32x18xf32>
    scf.yield %newBuffer : memref<32x18xf32>
  }
  return %lastBuffer : memref<32x18xf32>
}

func.func @return_view_alloc() -> memref<16xf32> {
  // CHECK-LABEL: func @return_view_alloc() -> memref<16xf32>
  // CHECK: %[[m0:.*]] = memref.alloc() : memref<128xi8>
  %c0 = arith.constant 0: index
  %f0 = arith.constant 0.0: f32
  %alloc = memref.alloc() : memref<128xi8>
  %view = memref.view %alloc[%c0][] : memref<128xi8> to memref<32xf32>
  %subview = memref.subview %view[0][16][1] : memref<32xf32> to memref<16xf32>
  return %subview : memref<16xf32>
}

func.func @alloc_dealloc_view() {
  // CHECK-LABEL: func @alloc_dealloc_view()
  // CHECK: %[[m0:.*]] = cpuruntime.alloc() : memref<128xi8>
  %c0 = arith.constant 0: index
  %f0 = arith.constant 0.0: f32
  %alloc = memref.alloc() : memref<128xi8>
  %view = memref.view %alloc[%c0][] : memref<128xi8> to memref<32xf32>
  %subview = memref.subview %view[0][16][1] : memref<32xf32> to memref<16xf32>
  // CHECK: cpuruntime.dealloc
  memref.dealloc %subview : memref<16xf32>
  return
}
