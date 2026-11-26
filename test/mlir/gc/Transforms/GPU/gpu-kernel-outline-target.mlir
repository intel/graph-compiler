// RUN: gc-opt %s --gpu-dev-props=arch=bmg --gpu-kernel-outline=cmd-options=-ze-opt-large-register-file | FileCheck %s

// The outlined kernel module must get a single xevm target with the command
// options as flags. The chip is taken from the device attributes; "bmg" is the
// default chip of the xevm target and is therefore not printed.
//
// By default the outermost loop dimension (the rows) is mapped to block x. A
// kernel with grid_mapping = "innermost-first" maps its first three dimensions
// in reverse order: the third (the columns) to block x and the first to block
// z. Its fourth dimension stays sequential, and a nested parallel loop is still
// mapped to threads.

// CHECK-LABEL: func.func @entry
// CHECK:         %[[ROWS:.+]] = affine.apply #{{.+}}(%c64)
// CHECK:         %[[COLS:.+]] = affine.apply #{{.+}}(%c128)
// CHECK:         gpu.launch_func @kernel::@kernel blocks in (%[[ROWS]], %[[COLS]], %{{.+}})
// CHECK:         %[[EXPERTS:.+]] = affine.apply #{{.+}}(%c3
// CHECK:         %[[ROWS2:.+]] = affine.apply #{{.+}}(%c64
// CHECK:         %[[COLS2:.+]] = affine.apply #{{.+}}(%c128
// CHECK:         gpu.launch_func @kernel_inner::@kernel_inner blocks in (%[[COLS2]], %[[ROWS2]], %[[EXPERTS]])
// CHECK:       gpu.module @kernel [#xevm.target<O = 3, flags = {"cmd-options" = ["-ze-opt-large-register-file"]}>] {
// CHECK:         gpu.func @kernel
// CHECK:       gpu.module @kernel_inner
// CHECK:         gpu.func @kernel_inner
// CHECK:           %[[BX:.+]] = gpu.block_id x
// CHECK:           %[[BY:.+]] = gpu.block_id y
// CHECK:           %[[BZ:.+]] = gpu.block_id z
// CHECK:           %[[TX:.+]] = gpu.thread_id x
// CHECK:           %[[EXPERT:.+]] = affine.apply #{{.+}}(%[[BZ]])
// CHECK:           %[[ROW:.+]] = affine.apply #{{.+}}(%[[BY]])
// CHECK:           %[[COLUMN:.+]] = affine.apply #{{.+}}(%[[BX]])
// CHECK:           scf.for %[[EXTRA:[^ ]+]] =
// CHECK:           %[[LOCAL:.+]] = affine.apply #{{.+}}(%[[TX]])
// CHECK:           memref.store %{{.+}}, %{{.+}}[%[[EXPERT]], %[[ROW]], %[[COLUMN]], %[[EXTRA]], %[[LOCAL]]]
module attributes {gc.module = {kernels = {kernel = {threads = [16, 1, 1]}, kernel_inner = {grid_mapping = "innermost-first", threads = [16, 1, 1]}}}} {
  func.func @entry(%arg0: memref<64x128xf16>, %inner: memref<3x64x128x2x16xf16>) {
    %cst = arith.constant 0.0 : f16
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sixteen = arith.constant 16 : index
    scf.forall (%i, %j) = (0, 0) to (64, 128) step (16, 16) {
      memref.store %cst, %arg0[%i, %j] : memref<64x128xf16>
    } {gc.kernel_name = "kernel"}
    scf.forall (%expert, %row, %column, %extra) = (0, 0, 0, 0) to (3, 64, 128, 2) step (1, 16, 16, 1) {
      scf.parallel (%local) = (%zero) to (%sixteen) step (%one) {
        memref.store %cst, %inner[%expert, %row, %column, %extra, %local] : memref<3x64x128x2x16xf16>
        scf.reduce
      }
    } {gc.kernel_name = "kernel_inner"}
    return
  }
}
