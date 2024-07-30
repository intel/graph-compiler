// RUN: gc-opt %s -convert-microkernel-to-dnnl-func -cse -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @basic_convert() {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    scf.forall (%arg0, %arg1) in (4, 8) {
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32x32xf32>)
      %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_4 = memref.subview %alloc_0[%arg1, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (f32, f32) 
      microkernel.brgemm.prologue(%0) : (i64) -> ()
      microkernel.brgemm(%0, %subview, %subview_4, %alloc_3, %c16_i64, %c0_i64) : (i64, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
      microkernel.brgemm.epilogue(%0) : (i64) -> ()
      %subview_5 = memref.subview %alloc_1[%arg0, %arg1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_3, %subview_5 : memref<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%alloc_3 : memref<32x32xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %1 = arith.addf %in, %in_7 : f32
        linalg.yield %1 : f32
      }
      %subview_6 = memref.subview %alloc_2[%arg0, %arg1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_3 : memref<32x32xf32>) outs(%subview_6 : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.maximumf %in, %cst : f32
        linalg.yield %1 : f32
      }
      memref.dealloc %alloc_3 : memref<32x32xf32>
    }
    return
  }
}

// CHECK-LABEL: dnnl_brgemm_execute
// CHECK-LABEL: dnnl_brgemm_dispatch
// CHECK-LABEL: basic_convert
// CHECK: %[[CST3:.+]] = arith.constant 3 : i64
// CHECK: %[[CST1F:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST1024:.+]] = arith.constant 1024 : i64
// CHECK: %[[CST32:.+]] = arith.constant 32 : i64
// CHECK: %[[CST0:.+]] = arith.constant 0 : index
// CHECK: %[[CST16:.+]] = arith.constant 16 : i64

// CHECK: %[[ptrC:.+]] = memref.extract_aligned_pointer_as_index %[[memrefC:.+]] : memref<32x32xf32> -> index
// CHECK-NEXT: %[[idxC:.+]] = arith.index_cast %[[ptrC]] : index to i64
// CHECK-NEXT: %[[llvmptrC:.+]] = llvm.inttoptr %[[idxC]] : i64 to !llvm.ptr

// CHECK: %[[bbA:.+]], %[[offA:.+]], %[[szA:.+]]:3, %[[strdA:.+]]:3 = memref.extract_strided_metadata %[[memrefA:.+]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
// CHECK-NEXT: %[[ptrA:.+]] = memref.extract_aligned_pointer_as_index %[[memrefA]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
// CHECK-NEXT: %[[idxA:.+]] = arith.index_cast %[[ptrA]] : index to i64
// CHECK-NEXT: %[[llvmptrA:.+]] = llvm.inttoptr %[[idxA]] : i64 to !llvm.ptr

// CHECK: %[[bbB:.+]], %[[offB:.+]], %[[szB:.+]]:3, %[[strdB:.+]]:3 = memref.extract_strided_metadata %[[memrefB:.+]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
// CHECK-NEXT: %[[ptrB:.+]] = memref.extract_aligned_pointer_as_index %[[memrefB]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
// CHECK-NEXT: %[[idxB:.+]] = arith.index_cast %[[ptrB]] : index to i64
// CHECK-NEXT: %[[llvmptrB:.+]] = llvm.inttoptr %[[idxB]] : i64 to !llvm.ptr

// CHECK: %[[KERNEL:.+]] = func.call @dnnl_brgemm_dispatch(%[[CST32]], %[[CST32]], %[[CST32]], %[[CST32]], %[[CST32]], %[[CST32]], %[[CST1024]], %[[CST1024]], %[[CST1F]], %[[CST3]], %[[CST3]]) : (i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i64
// CHECK-NOT: microkernel.brgemm.prologue(%[[TMP:.+]]) : (i64) -> ()

// CHECK: func.call @dnnl_brgemm_execute(%[[KERNEL]], %[[llvmptrA]], %[[offA]], %[[llvmptrB]], %[[offB]], %[[llvmptrC]], %[[CST0]], %[[CST16]]) : (i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
// CHECK-NOT: microkernel.brgemm.epilogue(%[[KERNEL]]) : (i64) -> ()

// -----
