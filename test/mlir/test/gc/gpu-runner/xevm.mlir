// RUN: gc-gpu-runner --shared-libs=%mlir_runner_utils %s | FileCheck %s
  
module @gemm attributes {gpu.container_module} {

  gpu.module @kernels {
    gpu.func @load_store(%src: memref<8x16xf32>, %dst: memref<8x16xf32>) kernel {
      %constant = arith.constant 1.23 : f32
      %c0 = arith.constant 0 : index
      memref.store %constant, %dst[%c0, %c0] : memref<8x16xf32>
      gpu.return
    }
  }
  gpu.module @kernel {
    gpu.func @store_constant(%ptr: !llvm.ptr) kernel {
    %const_val = arith.constant 42.0 : f32
    llvm.store %const_val, %ptr : f32, !llvm.ptr
    gpu.return
    }
  }


  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %token0 = gpu.wait async
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 1 : index
    %memref_0 = gpu.alloc [%token0]  host_shared () : memref<8x16xf32>
    memref.copy %src, %memref_0 : memref<8x16xf32> to memref<8x16xf32>
    %0 = memref.extract_aligned_pointer_as_index %memref_0 : memref<8x16xf32> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %token1 = gpu.wait async
    %5 = gpu.launch_func async [%token1] @kernel::@store_constant blocks in (%c1, %c1, %c1) threads in (%c1, %c16, %c1) args(%2 : !llvm.ptr)
    gpu.wait [%5]
    // gpu.dealloc %memref_0 : memref<8x16xf32>
    return %memref_0 : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf32>
    %B = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    memref.dealloc %A : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
