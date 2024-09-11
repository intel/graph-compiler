// RUN: gc-opt %s -microkernel-invariant-code-motion -split-input-file | FileCheck %s

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @parallel_no_hoist() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.forall (%arg0, %arg1) in (4, 8) {
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
      %2 = arith.index_cast %intptr : index to i64
      %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
      linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
      %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
      %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
      %4 = arith.index_cast %intptr_2 : index to i64
      %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
      %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
      %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
      %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
      %6 = arith.index_cast %intptr_8 : index to i64
      %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
      func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
      func.call @dnnl_brgemm_execute(%1, %5, %offset, %7, %offset_5, %3, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
      func.call @dnnl_brgemm_tilerelease() : () -> ()
      memref.dealloc %alloc_1 : memref<32x32xf32>
    }
    return
  }
}

// CHECK-LABEL: parallel_no_hoist

// CHECK: scf.forall (%arg0, %arg1) in (4, 8)
// CHECK: call @dnnl_brgemm_tileconfig(%[[A:.+]]) : (i64) -> ()
// CHECK: call @dnnl_brgemm_execute([[B:.+]]) : ([[C:.+]]) -> ()
// CHECK-NEXT: call @dnnl_brgemm_tilerelease() : () -> () 

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @multi_level_conflict() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i64 = arith.constant 2 : i64
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %2 = arith.index_cast %intptr : index to i64
        %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %4 = arith.index_cast %intptr_2 : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_8 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
        func.call @dnnl_brgemm_execute(%1, %5, %offset, %7, %offset_5, %3, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
        func.call @dnnl_brgemm_tilerelease() : () -> ()
        %subview_9 = memref.subview %alloc[%arg0, 0, 0, 0] [1, 2, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer_10, %offset_11, %sizes_12:3, %strides_13:3 = memref.extract_strided_metadata %subview_9 : memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_14 = memref.extract_aligned_pointer_as_index %subview_9 : memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %8 = arith.index_cast %intptr_14 : index to i64
        %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
        %subview_15 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_16, %offset_17, %sizes_18:4, %strides_19:4 = memref.extract_strided_metadata %subview_15 : memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_20 = memref.extract_aligned_pointer_as_index %subview_15 : memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %10 = arith.index_cast %intptr_20 : index to i64
        %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
        func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
        func.call @dnnl_brgemm_execute(%1, %9, %offset_11, %11, %offset_17, %3, %c0, %c2_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
        func.call @dnnl_brgemm_tilerelease() : () -> ()
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: multi_level_conflict

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1

// CHECK: call @dnnl_brgemm_tileconfig(%[[A:.+]]) : (i64) -> ()
// CHECK: call @dnnl_brgemm_execute([[B:.+]]) : ([[C:.+]]) -> ()
// CHECK-NOT: call @dnnl_brgemm_tilerelease() : () -> ()

// CHECK: call @dnnl_brgemm_tileconfig(%[[D:.+]]) : (i64) -> ()
// CHECK: call @dnnl_brgemm_execute([[E:.+]]) : ([[F:.+]]) -> ()
// CHECK-NOT: call @dnnl_brgemm_tilerelease() : () -> ()
// CHECK: call @dnnl_brgemm_tilerelease() : () -> ()
// CHECK-NEXT: return

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @multi_level_partial_hoist() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i64 = arith.constant 2 : i64
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %2 = arith.index_cast %intptr : index to i64
        %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %4 = arith.index_cast %intptr_2 : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_8 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
        func.call @dnnl_brgemm_execute(%1, %5, %offset, %7, %offset_5, %3, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
        func.call @dnnl_brgemm_tilerelease() : () -> ()
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %2 = arith.index_cast %intptr : index to i64
        %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg1, 0, 0, 0] [1, 2, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %4 = arith.index_cast %intptr_2 : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_8 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
        func.call @dnnl_brgemm_execute(%1, %5, %offset, %7, %offset_5, %3, %c0, %c2_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
        func.call @dnnl_brgemm_tilerelease() : () -> ()
      }
    }
    return
  }
}

// CHECK-LABEL: multi_level_partial_hoist

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: call @dnnl_brgemm_tileconfig(%[[A:.+]]) : (i64) -> ()
// CHECK: scf.for %arg1 = %c0 to %c8 step %c1
// CHECK: call @dnnl_brgemm_execute([[B:.+]]) : ([[C:.+]]) -> ()

// CHECK: }
// CHECK-NEXT: call @dnnl_brgemm_tileconfig(%[[D:.+]]) : (i64) -> ()
// CHECK-NEXT: scf.for %arg1 = %c0 to %c4 step %c1
// CHECK: call @dnnl_brgemm_execute([[E:.+]]) : ([[F:.+]]) -> ()

// CHECK: call @dnnl_brgemm_tilerelease() : () -> ()
// CHECK-NEXT: return

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @multi_level_full_hoist() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_stride_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %2 = arith.index_cast %intptr : index to i64
        %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %4 = arith.index_cast %intptr_2 : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_8 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
        func.call @dnnl_brgemm_execute(%1, %5, %offset, %7, %offset_5, %3, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
        func.call @dnnl_brgemm_tilerelease() : () -> ()
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: multi_level_full_hoist

// CHECK: call @dnnl_brgemm_tileconfig(%[[A:.+]]) : (i64) -> ()
// CHECK-NEXT: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: call @dnnl_brgemm_execute([[B:.+]]) : ([[C:.+]]) -> ()
    
// CHECK: call @dnnl_brgemm_tilerelease() : () -> ()
// CHECK-NEXT: return

// -----
