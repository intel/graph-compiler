// RUN: gc-opt %s --gpu-to-gc-gpu=wait=true | FileCheck %s
// RUN: gc-opt %s --gpu-to-gc-gpu=wait=false | FileCheck %s --check-prefix=RETURN
// RUN: gc-opt %s --gpu-to-gc-gpu=wait=true | gc-opt | FileCheck %s --check-prefix=PARSE

// The kernels are created lazily and cached in the global variables.
// CHECK-DAG: llvm.mlir.global internal constant @gcGpuKernel_k0_Bin("k0bin")
// CHECK-DAG: llvm.mlir.global internal constant @gcGpuKernel_k0_Name("k0\00")
// CHECK-DAG: llvm.mlir.global internal constant @gcGpuKernel_k0_ArgTypes(dense<[10, 6]> : tensor<2xi8>)
// CHECK-DAG: llvm.mlir.global internal @gcGpuKernel_k0_Ptr(#llvm.zero)
// CHECK-DAG: llvm.mlir.global internal @gcGpuKernel_k1_Ptr(#llvm.zero)
// CHECK-DAG: llvm.mlir.global internal @gcGpuKernel_k2_Ptr(#llvm.zero)
// CHECK: llvm.func internal @gcGpuCreateKernel_k0(%[[RT:.+]]: !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.call @gcGpuCreateKernel(%[[RT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.cmpxchg
// CHECK: llvm.call @gcGpuDestroyKernel
// CHECK: llvm.func internal @gcGpuGetKernel_k0(%{{.+}}: !llvm.ptr) -> !llvm.ptr attributes {always_inline}
// CHECK: llvm.load %{{.+}} atomic acquire <alignment = 8> : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @gcGpuCreateKernel_k0

// The entry function is renamed to main. The runtime pointer and the external
// dependencies are appended to its arguments.
// CHECK: llvm.func @main(%[[A:.+]]: !llvm.ptr, %[[B:.+]]: !llvm.ptr, %[[RUNTIME:.+]]: !llvm.ptr, %[[DEPS:.+]]: !llvm.ptr, %[[NDEPS:.+]]: i32) {

// A single array of nDeps + 1 elements is allocated for the merged dependencies
// and the external ones are copied to its beginning. Only k1 has events to
// merge, thus a single slot is reserved.
// CHECK-NEXT: %[[OFF:.+]] = llvm.zext %[[NDEPS]] : i32 to i64
// CHECK-NEXT: %[[ONE:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[SIZE:.+]] = llvm.add %[[OFF]], %[[ONE]] : i64
// CHECK-NEXT: %[[ARR:.+]] = llvm.alloca %[[SIZE]] x !llvm.ptr
// CHECK: "llvm.intr.memcpy"(%[[ARR]], %[[DEPS]], %{{.+}})
// CHECK-NEXT: %[[SLOT:.+]] = llvm.getelementptr %[[ARR]][%[[OFF]]]

// All the calls share a single buffer of the pointer arrays (the kernel
// argument pointers and the event lists) and a single buffer of the kernel
// argument values. Two slots are enough for both.
// CHECK-NEXT: %{{.+}} = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[PTRS:.+]] = llvm.alloca %{{.+}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT: %{{.+}} = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VALS:.+]] = llvm.alloca %{{.+}} x !llvm.ptr {alignment = 16 : i64}

// k0 takes a pointer and an i64, that are stored in the value buffer, and the
// pointers to the values are stored in the pointer buffer. k0 has no event
// dependencies, the external ones are passed as is.
// CHECK: llvm.store %[[A]], %[[VALS]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[VAL1:.+]] = llvm.getelementptr %[[VALS]][1]
// CHECK-NEXT: llvm.store %{{.+}}, %[[VAL1]] : i64, !llvm.ptr
// CHECK-NEXT: llvm.store %[[VALS]], %[[PTRS]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[PTR1:.+]] = llvm.getelementptr %[[PTRS]][1]
// CHECK-NEXT: llvm.store %[[VAL1]], %[[PTR1]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[K0:.+]] = llvm.call @gcGpuLaunch(%[[RUNTIME]], %{{.+}}, %[[PTRS]], {{.*}}, %[[DEPS]], %[[NDEPS]]) {gc.kernel_name = "k0"}

// The allocation waits for the k0 event, k1 depends on.
// CHECK: llvm.store %[[K0]], %[[PTRS]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.call @gcGpuWait(%[[RUNTIME]], %[[PTRS]], %{{.+}}) : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK: %[[PTR:.+]] = llvm.call @gcGpuAllocDevice(%[[RUNTIME]], %{{.+}}, %{{.+}}) : (!llvm.ptr, i64, i64) -> !llvm.ptr

// The copy has no event dependencies, k1 depends on the copy.
// CHECK: %[[CPY:.+]] = llvm.call @gcGpuMemcpy(%[[RUNTIME]], %[[B]], %[[PTR]], %{{.+}}, %[[DEPS]], %[[NDEPS]])

// The k1 event is stored to the shared deps array, following the external deps.
// CHECK: llvm.store %[[CPY]], %[[SLOT]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[N:.+]] = llvm.add %[[NDEPS]], %{{.+}} : i32
// CHECK-NEXT: %[[K1:.+]] = llvm.call @gcGpuLaunch(%[[RUNTIME]], %{{.+}}, %[[PTRS]], {{.*}}, %[[ARR]], %[[N]]) {gc.kernel_name = "k1"}

// The memory is not released before the copy and k1 are complete.
// CHECK: llvm.store %[[CPY]], %[[PTRS]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.store %[[K1]], %{{.+}} : !llvm.ptr, !llvm.ptr
// CHECK: llvm.call @gcGpuWait(%[[RUNTIME]]
// CHECK: llvm.call @gcGpuFree(%[[RUNTIME]], %[[PTR]])

// The k2 event is the only one we still own.
// CHECK: %[[K2:.+]] = llvm.call @gcGpuLaunch(%[[RUNTIME]], %{{.+}}, %[[PTRS]], {{.*}}, %[[DEPS]], %[[NDEPS]]) {gc.kernel_name = "k2"}
// CHECK: llvm.store %[[K2]], %[[PTRS]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.call @gcGpuWait(%[[RUNTIME]]
// CHECK: llvm.return

// The three buffers are the only stack allocations in the function.
// CHECK-NOT: llvm.alloca

// CHECK: llvm.func @eventNum() -> i32
// CHECK-NEXT: %[[NUM:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %[[NUM]]
// CHECK: llvm.func @destructor()
// CHECK-NEXT: llvm.fence acquire
// CHECK-COUNT-3: llvm.call @gcGpuDestroyKernel

// Without the wait option, the owned events are returned to the caller.
// RETURN: llvm.func @main(%{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr, %{{.+}}: i32, %[[EVENT:.+]]: !llvm.ptr) {
// RETURN: %[[K2:.+]] = llvm.call @gcGpuLaunch({{.*}}) {gc.kernel_name = "k2"}
// RETURN: llvm.store %[[K2]], %[[EVENT]] : !llvm.ptr, !llvm.ptr
// RETURN-NEXT: llvm.return

// RETURN: llvm.func @eventNum() -> i32
// RETURN-NEXT: %[[NUM:.+]] = llvm.mlir.constant(1 : i32) : i32
// RETURN-NEXT: llvm.return %[[NUM]]

// The generated module must be valid.
// PARSE: llvm.call @gcGpuLaunch

module attributes {gc.module = {kernels = {k1 = {depends = ["k0"]}}}, gpu.container_module} {
  llvm.func @entry(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(64 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    gpu.launch_func <%3 : !llvm.ptr> @k0::@k0 blocks in (%0, %1, %1) threads in (%1, %1, %1) : i64 args(%arg0 : !llvm.ptr, %0 : i64)
    %4 = llvm.call @mgpuMemAlloc(%0, %3, %2) : (i64, !llvm.ptr, i1) -> !llvm.ptr
    llvm.call @mgpuMemcpy(%4, %arg1, %0, %3) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    gpu.launch_func <%3 : !llvm.ptr> @k1::@k1 blocks in (%0, %1, %1) threads in (%1, %1, %1) : i64 args(%arg0 : !llvm.ptr, %4 : !llvm.ptr)
    llvm.call @mgpuMemFree(%4, %3) : (!llvm.ptr, !llvm.ptr) -> ()
    gpu.launch_func <%3 : !llvm.ptr> @k2::@k2 blocks in (%0, %1, %1) threads in (%1, %1, %1) : i64 args(%arg1 : !llvm.ptr)
    llvm.call @mgpuStreamSynchronize(%3) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%3) : (!llvm.ptr) -> ()
    llvm.return
  }
  gpu.binary @k0 [#gpu.object<#xevm.target<O = 3>, bin = "k0bin">]
  gpu.binary @k1 [#gpu.object<#xevm.target<O = 3>, bin = "k1bin">]
  gpu.binary @k2 [#gpu.object<#xevm.target<O = 3>, bin = "k2bin">]
  llvm.func @mgpuStreamCreate() -> !llvm.ptr
  llvm.func @mgpuStreamSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamDestroy(!llvm.ptr)
  llvm.func @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
  llvm.func @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
  llvm.func @mgpuMemFree(!llvm.ptr, !llvm.ptr)
}
