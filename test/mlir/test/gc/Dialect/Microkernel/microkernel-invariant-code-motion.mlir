// RUN: gc-opt %s -early-dispatch-microkernel -convert-microkernel-to-dnnl-func -cse -microkernel-invariant-code-motion -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_brgemm() {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    scf.forall (%arg0, %arg1) in (4, 8) {
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32x32xf32>)
      %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      %subview_4 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
      %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(bf16, bf16) 
      microkernel.brgemm.prologue(%0) : (i64) -> ()
      microkernel.brgemm.execute(%0, %subview, %subview_4, %alloc_3, %c16_i64, %c0_i64) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
      microkernel.brgemm.epilogue(%0) : (i64) -> ()
      memref.dealloc %alloc_3 : memref<32x32xf32>
    }
    return
  }
}

// CHECK-LABEL: simple_brgemm

// CHECK: scf.forall (%arg0, %arg1) in (4, 8)
// CHECK: call @dnnl_brgemm_tileconfig(%[[A:.+]]) : (i64) -> ()
// CHECK: call @dnnl_brgemm_execute([[B:.+]]) : ([[C:.+]]) -> ()
// CHECK-NEXT: call @dnnl_brgemm_tilerelease() : () -> () 

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_brgemm() {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %c8_index = arith.constant 8 : index
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    scf.for %arg0 = %c0_index to %c4_index step %c1_index {
        scf.for %arg1 = %c0_index to %c8_index step %c1_index {
	      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
	      linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32x32xf32>)
	      %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	      %subview_4 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	      %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(bf16, bf16) 
	      microkernel.brgemm.prologue(%0) : (i64) -> ()
	      microkernel.brgemm.execute(%0, %subview, %subview_4, %alloc_3, %c16_i64, %c0_i64) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	      microkernel.brgemm.epilogue(%0) : (i64) -> ()
	      
	      %subview_7 = memref.subview %alloc[%arg0, 0, 0, 0] [1, 2, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	      %subview_8 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	      %10 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(bf16, bf16) 
	      microkernel.brgemm.prologue(%10) : (i64) -> ()
	      microkernel.brgemm.execute(%10, %subview_7, %subview_8, %alloc_3, %c2_i64, %c0_i64) : (i64, memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	      microkernel.brgemm.epilogue(%10) : (i64) -> ()
	      
              memref.dealloc %alloc_3 : memref<32x32xf32>
        }
    }
    return
  }
}

// CHECK-LABEL: simple_brgemm

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

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_brgemm() {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %c8_index = arith.constant 8 : index
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    scf.for %arg0 = %c0_index to %c4_index step %c1_index {
        scf.for %arg1 = %c0_index to %c8_index step %c1_index {
	    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
	    linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32x32xf32>)
	    %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	    %subview_4 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	    %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(bf16, bf16) 
	    microkernel.brgemm.prologue(%0) : (i64) -> ()
	    microkernel.brgemm.execute(%0, %subview, %subview_4, %alloc_3, %c16_i64, %c0_i64) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	    microkernel.brgemm.epilogue(%0) : (i64) -> ()
	    memref.dealloc %alloc_3 : memref<32x32xf32>
        }
        scf.for %arg2 = %c0_index to %c4_index step %c1_index {
	    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc_4 : memref<32x32xf32>)
	    %subview_7 = memref.subview %alloc[%arg2, 0, 0, 0] [1, 2, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	    %subview_8 = memref.subview %alloc_0[%arg2, 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	    %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(bf16, bf16) 
	    microkernel.brgemm.prologue(%0) : (i64) -> ()
            microkernel.brgemm.execute(%0, %subview_7, %subview_8, %alloc_4, %c2_i64, %c0_i64) : (i64, memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	    microkernel.brgemm.epilogue(%0) : (i64) -> ()
        }
    }
    return
  }
}

// CHECK-LABEL: simple_brgemm

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

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_brgemm() {
    %c0_i64 = arith.constant 0 : i64
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %c8_index = arith.constant 8 : index
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    scf.for %arg0 = %c0_index to %c4_index step %c1_index {
        scf.for %arg1 = %c0_index to %c8_index step %c1_index {
	      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
	      linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32x32xf32>)
	      %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	      %subview_4 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	      %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags(stride) data_type(bf16, bf16) 
	      microkernel.brgemm.prologue(%0) : (i64) -> ()
	      microkernel.brgemm.execute(%0, %subview, %subview_4, %alloc_3, %c16_i64, %c0_i64) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	      microkernel.brgemm.epilogue(%0) : (i64) -> ()
	      memref.dealloc %alloc_3 : memref<32x32xf32>
        }
    }
    return
  }
}

// CHECK-LABEL: simple_brgemm

// CHECK: call @dnnl_brgemm_tileconfig(%[[A:.+]]) : (i64) -> ()
// CHECK-NEXT: scf.for %arg0 = %c0 to %c4 step %c1
// CHECK-NEXT: scf.for %arg1 = %c0 to %c8 step %c1 

// CHECK: call @dnnl_brgemm_execute([[B:.+]]) : ([[C:.+]]) -> ()
    
// CHECK: call @dnnl_brgemm_tilerelease() : () -> ()
// CHECK-NEXT: return

// -----
