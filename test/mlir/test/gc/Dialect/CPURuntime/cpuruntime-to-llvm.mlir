// RUN: gc-opt %s --convert-cpuruntime-to-llvm | FileCheck %s

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_init_1_32_32_32_32_4096_32_1024_3_3_ctor], priorities = [2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_init_1_32_32_32_32_4096_32_1024_3_3() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_init_1_32_32_32_32_4096_32_1024_3_3_ctor() -> i64 {
    %c3_i64 = arith.constant 3 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c1024_i64 = arith.constant 1024 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c32_i64 = arith.constant 32 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_1_32_32_32_32_4096_32_1024_3_3 : !llvm.ptr
    %1 = func.call @dnnl_brgemm_dispatch(%c1_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c4096_i64, %c32_i64, %c1024_i64, %cst, %c3_i64, %c3_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @main_entry(%arg0: memref<1x128x1x32xf32>, %arg1: memref<128x128x32x32xf32>, %arg2: memref<1x128x1x32xf32>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_1_32_32_32_32_4096_32_1024_3_3 : !llvm.ptr
    %c128_i64 = arith.constant 128 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %subview = memref.subview %arg0[0, 0, 0, 0] [1, 128, 1, 32] [1, 1, 1, 1] : memref<1x128x1x32xf32> to memref<128x1x32xf32, strided<[32, 32, 1]>>
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<128x1x32xf32, strided<[32, 32, 1]>> -> index
    %2 = arith.index_cast %intptr : index to i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
    %c0_0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1_1 = arith.constant 1 : index
    scf.parallel (%arg3) = (%c0_0) to (%c32) step (%c1_1) {
      %4 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg3)
      %subview_2 = memref.subview %arg2[0, %4, 0, 0] [1, 4, 1, 32] [1, 1, 1, 1] : memref<1x128x1x32xf32> to memref<1x4x1x32xf32, strided<[4096, 32, 32, 1], offset: ?>>
      func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %5 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + s0)>(%arg3)[%arg4]
        %subview_3 = memref.subview %arg1[%5, 0, 0, 0] [1, 128, 32, 32] [1, 1, 1, 1] : memref<128x128x32x32xf32> to memref<128x32x32xf32, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview_3 : memref<128x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
        %intptr_4 = memref.extract_aligned_pointer_as_index %subview_3 : memref<128x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_4 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        %subview_5 = memref.subview %subview_2[0, %arg4, 0, 0] [1, 1, 1, 32] [1, 1, 1, 1] : memref<1x4x1x32xf32, strided<[4096, 32, 32, 1], offset: ?>> to memref<1x32xf32, strided<[4096, 1], offset: ?>>
        %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %subview_5 : memref<1x32xf32, strided<[4096, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %intptr_10 = memref.extract_aligned_pointer_as_index %subview_5 : memref<1x32xf32, strided<[4096, 1], offset: ?>> -> index
        %8 = arith.index_cast %intptr_10 : index to i64
        %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
        func.call @dnnl_brgemm_execute(%1, %3, %c0, %7, %offset, %9, %offset_7, %c128_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
      }
      func.call @dnnl_brgemm_tilerelease() : () -> ()
      scf.reduce 
    }
    return
  }
}