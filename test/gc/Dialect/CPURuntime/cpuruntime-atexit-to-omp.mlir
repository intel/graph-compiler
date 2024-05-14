// RUN: gc-opt %s --cpuruntime-atexit-to-omp | FileCheck %s

module {
  func.func @parallel_insert_slice(%arg0: memref<512x512xf32>) -> memref<512x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    memref.copy %arg0, %alloc : memref<512x512xf32> to memref<512x512xf32>
    %0 = llvm.mlir.constant(1 : i64) : i64
    omp.parallel {
      omp.wsloop for  (%arg1, %arg2) : index = (%c0, %c0) to (%c1, %c512) step (%c1, %c1) {
        memref.alloca_scope  {
          %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512xf32>
          %subview = memref.subview %alloc[%arg1, 0] [1, 512] [1, 1] : memref<512x512xf32> to memref<512xf32, strided<[1], offset: ?>>
          memref.copy %alloc_0, %subview : memref<512xf32> to memref<512xf32, strided<[1], offset: ?>>
          memref.dealloc %alloc_0 : memref<512xf32>
          cpuruntime.at_parallel_exit {
            memref.prefetch %alloc[%c1,%c0], read, locality<3>, data  : memref<512x512xf32>
            cpuruntime.parallel_exit.return
          }
        }
        omp.yield
      }
      memref.prefetch %alloc[%c0,%c0], read, locality<3>, data  : memref<512x512xf32>
      omp.terminator
    }
    // CHECK-DAG:    %[[C1:.*]] = arith.constant 1
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK:        omp.parallel
    // CHECK-NEXT:      omp.wsloop
    // CHECK-NEXT:         memref.alloca_scope
    // CHECK-NOT:              cpuruntime.at_parallel_exit
    // CHECK:              omp.yield
    // CHECK:           memref.prefetch {{%alloc}}[%[[C0]], %[[C0]]]
    // CHECK-NEXT:      memref.prefetch {{%alloc}}[%[[C1]], %[[C0]]]
    // CHECK-NEXT:      omp.terminator
    return %alloc : memref<512x512xf32>
  }
}
