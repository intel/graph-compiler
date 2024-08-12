// RUN: gc-opt %s -merge-branch-microkernel-context -split-input-file | FileCheck %s

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2_ctor, @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32, 2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
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
  func.func @if_branch_context_merge() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i64
    %3 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %4 = arith.index_cast %intptr : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_2 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %8 = arith.index_cast %intptr_8 : index to i64
        %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
        %10 = arith.cmpi eq, %arg0, %c0 : index
        scf.if %10 {
          func.call @dnnl_brgemm_tileconfig(%2) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%2, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        } else {
          func.call @dnnl_brgemm_tileconfig(%3) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%3, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        }
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: if_branch_context_merge

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK-NEXT: scf.if
// CHECK: } else {
// CHECK: }
// CHECK-NEXT: func.call @dnnl_brgemm_tilerelease() : () -> () 

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @if_only_branch_context_merge() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
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
        %8 = arith.cmpi eq, %arg0, %c0 : index
        scf.if %8 {
          func.call @dnnl_brgemm_tileconfig(%1) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%1, %5, %offset, %7, %offset_5, %3, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        }
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: if_only_branch_context_merge

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: scf.if
// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK: func.call @dnnl_brgemm_tilerelease() : () -> () 
// CHECK: }

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2_ctor, @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32, 2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c512_i64 = arith.constant 512 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c512_i64, %c512_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  func.func private @dnnl_brgemm_tilerelease()
  func.func private @dnnl_brgemm_execute(i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64)
  func.func private @dnnl_brgemm_tileconfig(i64)
  func.func private @dnnl_brgemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
  func.func @if_branch_context_no_merge() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i64
    %3 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %4 = arith.index_cast %intptr : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_2 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %8 = arith.index_cast %intptr_8 : index to i64
        %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
        %10 = arith.cmpi eq, %arg0, %c0 : index
        scf.if %10 {
          func.call @dnnl_brgemm_tileconfig(%2) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%2, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        } else {
          func.call @dnnl_brgemm_tileconfig(%3) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%3, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        }
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: if_branch_context_no_merge

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: scf.if
// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK: func.call @dnnl_brgemm_tilerelease() : () -> () 
// CHECK: } else {
// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK: func.call @dnnl_brgemm_tilerelease() : () -> () 
// CHECK: }

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2_ctor, @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor], priorities = [2147483647 : i32, 2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
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
  func.func @switch_branch_context_merge() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i64
    %3 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %4 = arith.index_cast %intptr : index to i64
        %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %6 = arith.index_cast %intptr_2 : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %8 = arith.index_cast %intptr_8 : index to i64
        %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
        scf.index_switch %arg0 
        case 0 {
          func.call @dnnl_brgemm_tileconfig(%3) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%3, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
          scf.yield
        }
        case 1 {
          func.call @dnnl_brgemm_tileconfig(%2) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%2, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
          scf.yield
        }
        default {
          func.call @dnnl_brgemm_tileconfig(%2) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%2, %7, %offset, %9, %offset_5, %5, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        }
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: switch_branch_context_merge

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK-NEXT: scf.index_switch
// CHECK: case 0 {
// CHECK: case 1 {
// CHECK: default {
// CHECK: }
// CHECK-NEXT: func.call @dnnl_brgemm_tilerelease() : () -> () 

// -----

module {
  llvm.mlir.global_ctors {ctors = [@g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2_ctor, @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor, @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2_ctor], priorities = [2147483647 : i32, 2147483647 : i32, 2147483647 : i32]}
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c512_i64 = arith.constant 512 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c512_i64, %c512_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %1 = func.call @dnnl_brgemm_dispatch(%c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c32_i64, %c1024_i64, %c1024_i64, %cst, %c2_i64, %c2_i64) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
    llvm.store %1, %0 : i64, !llvm.ptr
    llvm.return %1 : i64
  }
  llvm.mlir.global internal @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2() {addr_space = 0 : i32} : i64
  llvm.func @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2_ctor() -> i64 {
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
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
  func.func @switch_branch_context_no_merge() {
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_init_32_32_32_32_32_32_1024_1024_2_2 : !llvm.ptr
    %2 = llvm.mlir.addressof @g_dispatched_microkernel_brgemm_32_32_32_32_32_32_512_512_2_2 : !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> i64
    %4 = llvm.load %1 : !llvm.ptr -> i64
    %5 = llvm.load %0 : !llvm.ptr -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<32x32xf32> -> index
        %6 = arith.index_cast %intptr : index to i64
        %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
        linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<32x32xf32>)
        %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
        %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
        %intptr_2 = memref.extract_aligned_pointer_as_index %subview : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
        %8 = arith.index_cast %intptr_2 : index to i64
        %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
        %subview_3 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
        %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
        %intptr_8 = memref.extract_aligned_pointer_as_index %subview_3 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> -> index
        %10 = arith.index_cast %intptr_8 : index to i64
        %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
        scf.index_switch %arg0 
        case 0 {
          func.call @dnnl_brgemm_tileconfig(%4) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%4, %9, %offset, %11, %offset_5, %7, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
          scf.yield
        }
        case 1 {
          func.call @dnnl_brgemm_tileconfig(%5) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%5, %9, %offset, %11, %offset_5, %7, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
          scf.yield
        }
        default {
          func.call @dnnl_brgemm_tileconfig(%3) : (i64) -> ()
          func.call @dnnl_brgemm_execute(%3, %9, %offset, %11, %offset_5, %7, %c0, %c16_i64) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
          func.call @dnnl_brgemm_tilerelease() : () -> ()
        }
        memref.dealloc %alloc_1 : memref<32x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: switch_branch_context_no_merge

// CHECK: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK: func.call @dnnl_brgemm_tilerelease() : () -> () 
// CHECK: case 1 {
// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK: func.call @dnnl_brgemm_tilerelease() : () -> () 
// CHECK: default {
// CHECK: func.call @dnnl_brgemm_tileconfig
// CHECK: func.call @dnnl_brgemm_tilerelease() : () -> () 
// CHECK: }
