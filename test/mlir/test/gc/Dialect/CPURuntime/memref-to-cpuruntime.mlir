// UNSUPPORTED: target={{.*}}
// RUN: gc-opt --split-input-file --convert-memref-to-cpuruntime %s -verify-diagnostics | FileCheck %s

func.func @alloca() {
  // CHECK-LABEL: func @alloca()
  // CHECK: %[[m0:.*]] = cpuruntime.alloc() : memref<1024xf32>
  %m0 = memref.alloca() : memref<1024xf32>
  scf.forall (%i) in (32) {
  }
  // CHECK: cpuruntime.dealloc %[[m0]] : memref<1024xf32>
  return
}

func.func @thread_alloca() {
  // CHECK-LABEL: func.func @thread_alloca()
  // CHECK-NEXT: scf.forall {{.*}} {
  // CHECK-NEXT:   %[[m0:.*]] = cpuruntime.alloc thread_local() : memref<1024xf32>
  // CHECK-NEXT:   cpuruntime.dealloc thread_local %[[m0]] : memref<1024xf32>
  // CHECK-NEXT: }
  scf.forall (%i) in (32) {
    %0 = memref.alloca() : memref<1024xf32>
  }
  return
}

func.func @dynamic_ranked_alloca(%arg0: memref<*xf32>) {
  // CHECK-LABEL: func @dynamic_ranked_alloca(%arg0: memref<*xf32>)
  // CHECK: %[[RANK:.*]] = memref.rank %{{.*}} : memref<*xf32>
  // CHECK: %[[m0:.*]] = cpuruntime.alloc(%[[RANK]]) : memref<?xindex>
  // CHECK: cpuruntime.dealloc %[[m0]] : memref<?xindex>
  %0 = memref.rank %arg0 : memref<*xf32>
  %alloca = memref.alloca(%0) : memref<?xindex>
  return
}

func.func @loop_nested_if_alloca(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<64xf32>) {
  // CHECK-LABEL: func @loop_nested_if_alloca(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<64xf32>)
  // CHECK: %[[m0:.*]] = cpuruntime.alloc() : memref<64xf32>
  %alloca = memref.alloca() : memref<64xf32>
  %0 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3) -> (memref<64xf32>) {
    %1 = arith.cmpi eq, %arg5, %arg1 : index
    %2 = scf.if %1 -> (memref<64xf32>) {
      // CHECK: yield %[[m0]] : memref<64xf32>
      scf.yield %alloca : memref<64xf32>
    } else {
      // CHECK: %[[m1:.*]] = memref.alloca() : memref<2xf32>
      %alloca_0 = memref.alloca() : memref<2xf32>
      scf.yield %arg6 : memref<64xf32>
    }
    scf.yield %2 : memref<64xf32>
  }
  // CHECK: cpuruntime.dealloc %[[m0]] : memref<64xf32>
  return
}

func.func @alloca_sequence() {
  // CHECK-LABEL: func @alloca_sequence()
  // CHECK: %[[m0:.*]] = cpuruntime.alloc() : memref<128xf32>
  // CHECK: %[[m1:.*]] = cpuruntime.alloc() : memref<128xf32>
  // CHECK: %[[m2:.*]] = cpuruntime.alloc() : memref<128xf32>
  %alloc = memref.alloca() : memref<128xf32>
  %alloc_0 = memref.alloca() : memref<128xf32>
  %alloc_1 = memref.alloca() : memref<128xf32>
  // CHECK: cpuruntime.dealloc %[[m2]] : memref<128xf32>
  // CHECK: cpuruntime.dealloc %[[m1]] : memref<128xf32>
  // CHECK: cpuruntime.dealloc %[[m0]] : memref<128xf32>
  return
}

func.func @nested_alloca() {
  // CHECK-LABEL: func @nested_alloca()
  // CHECK: %[[m0:.*]] = cpuruntime.alloc() : memref<512xf32>
  // CHECK-NEXT: scf.forall {{.*}} {
  // CHECK-NEXT:   %[[m1:.*]] = cpuruntime.alloc thread_local() : memref<32xf32>
  // CHECK-NEXT:   cpuruntime.dealloc thread_local %[[m1]] : memref<32xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.forall {{.*}} {
  // CHECK-NEXT:   %[[m2:.*]] = cpuruntime.alloc thread_local() : memref<64xf32>
  // CHECK-NEXT:   cpuruntime.dealloc thread_local %[[m2]] : memref<64xf32>
  // CHECK-NEXT: }
  // CHECK:   cpuruntime.dealloc %[[m0]] : memref<512xf32>
  %0 = memref.alloca() : memref<512xf32>
  scf.forall (%i) in (32) {
    %1 = memref.alloca() : memref<32xf32>
  }
  scf.forall (%i) in (32) {
    %1 = memref.alloca() : memref<64xf32>
  }
  return
}
