// RUN: gc-opt %s --gpu-dev-props=arch=bmg --gpu-kernel-outline=cmd-options=-ze-opt-large-register-file | FileCheck %s

// The outlined kernel module must get a single xevm target with the command
// options as flags. The chip is taken from the device attributes; "bmg" is the
// default chip of the xevm target and is therefore not printed.

// CHECK:       gpu.module @kernel [#xevm.target<O = 3, flags = {"cmd-options" = ["-ze-opt-large-register-file"]}>] {
// CHECK:         gpu.func @kernel
module attributes {gc.module = {kernels = {kernel = {threads = [16, 1, 1]}}}} {
  func.func @entry(%arg0: memref<64x64xf16>) {
    %cst = arith.constant 0.0 : f16
    scf.forall (%i, %j) = (0, 0) to (64, 64) step (16, 16) {
      memref.store %cst, %arg0[%i, %j] : memref<64x64xf16>
    } {gc.kernel_name = "kernel"}
    return
  }
}
