// RUN: gc-opt %s --convert-cpuruntime-to-llvm | FileCheck %s

module {
  // CHECK: llvm.func @gcThreadAlignedFree(!llvm.ptr)
  // CHECK: llvm.func @gcThreadAlignedMalloc(i64) -> !llvm.ptr
  // CHECK: llvm.func @gcAlignedFree(!llvm.ptr)
  // CHECK: llvm.func @gcAlignedMalloc(i64) -> !llvm.ptr
  // CHECK: llvm.mlir.global internal constant @cpuprintfFormat_0("Hello world %f %d %lld\0A\00") {addr_space = 0 : i32}
  // CHECK: llvm.func @printf(!llvm.ptr,
  // CHECK-NEXT: func.func @doprint(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64)
  func.func @doprint(%t: f32, %t2: i32, %t3: i64) {
    // CHECK-NEXT: llvm.mlir.addressof
    // CHECK-DAG: %[[C1:.*]] = llvm.getelementptr
    // CHECK-SAME: !llvm.ptr, !llvm.array<24 x i8>
    // CHECK: %[[C2:.*]] = llvm.fpext %[[ARG0]] 
    // CHECK: %[[C3:.*]] = llvm.zext %[[ARG1]] 
    // CHECK-NOT: cpuruntime.printf
    // CHECK-NEXT: llvm.call @printf(%[[C1]], %[[C2]], %[[C3]], %[[ARG2]])
    cpuruntime.printf "Hello world %f %d %lld\n" %t, %t2, %t3 : f32, i32, i64
    return
  }

  func.func @doalloc() {
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(13 : index) : i64
    // CHECK: %[[c2:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: %[[null:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[c1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
    // CHECK: %[[call:.*]] = llvm.call @gcAlignedMalloc(%[[size_bytes]]) : (i64) -> !llvm.ptr
    // CHECK: %[[ptr:.*]] = llvm.bitcast %[[call]] : !llvm.ptr to !llvm.ptr
    // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[c1]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[c2]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[callfree:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[ptrfree:.*]] = llvm.bitcast %[[callfree]] : !llvm.ptr to !llvm.ptr
    // CHECK: llvm.call @gcAlignedFree(%[[ptrfree]]) : (!llvm.ptr) -> ()
    %m0 = cpuruntime.alloc () : memref<13xf32>
    cpuruntime.dealloc %m0 : memref<13xf32>
    return
  }

  func.func @do_thread_alloc() {
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(13 : index) : i64
    // CHECK: %[[c2:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: %[[null:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[c1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
    // CHECK: %[[call:.*]] = llvm.call @gcThreadAlignedMalloc(%[[size_bytes]]) : (i64) -> !llvm.ptr
    // CHECK: %[[ptr:.*]] = llvm.bitcast %[[call]] : !llvm.ptr to !llvm.ptr
    // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[c1]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: llvm.insertvalue %[[c2]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[callfree:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[ptrfree:.*]] = llvm.bitcast %[[callfree]] : !llvm.ptr to !llvm.ptr
    // CHECK: llvm.call @gcThreadAlignedFree(%[[ptrfree]]) : (!llvm.ptr) -> ()
    %m0 = cpuruntime.alloc thread_local () : memref<13xf32>
    cpuruntime.dealloc thread_local %m0 : memref<13xf32>
    return
  }

  // CHECK-LABEL: func @dynamic_alloc(
  //       CHECK:   %[[Marg:.*]]: index, %[[Narg:.*]]: index)
  func.func @dynamic_alloc(%arg0: index, %arg1: index) {
    //   CHECK-DAG:  %[[M:.*]] = builtin.unrealized_conversion_cast %[[Marg]]
    //   CHECK-DAG:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[Narg]]
    //  CHECK-NEXT:  %[[fortytwo:.*]] = llvm.mlir.constant(42 : index) : i64
    //  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
    //  CHECK-NEXT:  %[[mul:.*]] = llvm.mul %[[N]], %[[fortytwo]] : i64
    //  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %[[mul]], %[[M]] : i64
    //  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.zero : !llvm.ptr
    //  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    //  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
    //  CHECK-NEXT:  %[[call:.*]] = llvm.call @gcAlignedMalloc(%[[sz_bytes]]) : (i64) -> !llvm.ptr
    //  CHECK-NEXT:  llvm.bitcast %[[call]] : !llvm.ptr to !llvm.ptr
    //  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
    //  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[fortytwo]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[mul]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[one]], %{{.*}}[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  %[[callfree:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  %[[ptrfree:.*]] = llvm.bitcast %[[callfree]] : !llvm.ptr to !llvm.ptr
    //  CHECK-NEXT: llvm.call @gcAlignedFree(%[[ptrfree]]) : (!llvm.ptr) -> ()
    %m0 = cpuruntime.alloc(%arg0, %arg1) : memref<?x42x?xf32>
    cpuruntime.dealloc %m0 : memref<?x42x?xf32>
    return
  }

  // CHECK-LABEL: func @dynamic_thread_alloc(
  //       CHECK:   %[[Marg:.*]]: index, %[[Narg:.*]]: index)
  func.func @dynamic_thread_alloc(%arg0: index, %arg1: index) {
    //   CHECK-DAG:  %[[M:.*]] = builtin.unrealized_conversion_cast %[[Marg]]
    //   CHECK-DAG:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[Narg]]
    //  CHECK-NEXT:  %[[fortytwo:.*]] = llvm.mlir.constant(42 : index) : i64
    //  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
    //  CHECK-NEXT:  %[[mul:.*]] = llvm.mul %[[N]], %[[fortytwo]] : i64
    //  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %[[mul]], %[[M]] : i64
    //  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.zero : !llvm.ptr
    //  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    //  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
    //  CHECK-NEXT:  %[[call:.*]] = llvm.call @gcThreadAlignedMalloc(%[[sz_bytes]]) : (i64) -> !llvm.ptr
    //  CHECK-NEXT:  llvm.bitcast %[[call]] : !llvm.ptr to !llvm.ptr
    //  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
    //  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[fortytwo]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[mul]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  llvm.insertvalue %[[one]], %{{.*}}[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  %[[callfree:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    //  CHECK-NEXT:  %[[ptrfree:.*]] = llvm.bitcast %[[callfree]] : !llvm.ptr to !llvm.ptr
    //  CHECK-NEXT: llvm.call @gcThreadAlignedFree(%[[ptrfree]]) : (!llvm.ptr) -> ()
    %m0 = cpuruntime.alloc thread_local (%arg0, %arg1) : memref<?x42x?xf32>
    cpuruntime.dealloc thread_local %m0 : memref<?x42x?xf32>
    return
  }
}
