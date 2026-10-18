// RUN: gc-opt %s --gpu-to-gc-gpu=wait=false | FileCheck %s

// Multiple owned events are stored to the array, passed as the last argument.
// CHECK: llvm.func @main(%{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr, %{{.+}}: i32, %[[EVENTS:.+]]: !llvm.ptr) {
// CHECK: %[[K0:.+]] = llvm.call @gcGpuLaunch({{.*}}) {gc.kernel_name = "k0"}
// CHECK: %[[K1:.+]] = llvm.call @gcGpuLaunch({{.*}}) {gc.kernel_name = "k1"}
// CHECK: llvm.store %[[K0]], %[[EVENTS]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[SLOT1:.+]] = llvm.getelementptr %[[EVENTS]][1]
// CHECK-NEXT: llvm.store %[[K1]], %[[SLOT1]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: llvm.return

// CHECK: llvm.func @eventNum() -> i32
// CHECK-NEXT: %[[NUM:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: llvm.return %[[NUM]]

module attributes {gpu.container_module} {
  llvm.func @entry(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(64 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    gpu.launch_func <%2 : !llvm.ptr> @k0::@k0 blocks in (%0, %1, %1) threads in (%1, %1, %1) : i64 args(%arg0 : !llvm.ptr)
    gpu.launch_func <%2 : !llvm.ptr> @k1::@k1 blocks in (%0, %1, %1) threads in (%1, %1, %1) : i64 args(%arg0 : !llvm.ptr)
    llvm.call @mgpuStreamSynchronize(%2) : (!llvm.ptr) -> ()
    llvm.call @mgpuStreamDestroy(%2) : (!llvm.ptr) -> ()
    llvm.return
  }
  gpu.binary @k0 [#gpu.object<#xevm.target<O = 3>, bin = "k0bin">]
  gpu.binary @k1 [#gpu.object<#xevm.target<O = 3>, bin = "k1bin">]
  llvm.func @mgpuStreamCreate() -> !llvm.ptr
  llvm.func @mgpuStreamSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamDestroy(!llvm.ptr)
}
