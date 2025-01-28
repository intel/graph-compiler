// RUN: gc-opt %s --convert-xegpu-to-xevm --convert-xevm-to-llvm --xevm-attach-target --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-gpu-to-llvm-spv='use-64bit-index=true' --gpu-to-llvm --reconcile-unrealized-casts --cse --gpu-module-to-binary | gc-cpu-runner -e main -entry-point-result=void --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime | FileCheck %s

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<8x16xf32, 1> to memref<8x16xf32>
        %srcce3 = memref.collapse_shape %srcce [[0,1]] : memref<8x16xf32> into memref<128xf32>
          %dstte3 = memref.collapse_shape %dstte [[0,1]] : memref<8x16xf32> into memref<128xf32>

        %offsets = arith.constant dense<[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]> : vector<16xindex>
        %src_tdesc = xegpu.create_tdesc %srcce3, %offsets : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1>, #xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>>
        %dst_tdesc = xegpu.create_tdesc %dstte3, %offsets : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1>, #xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>>

        %mask = arith.constant dense<[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]> : vector<16xi1>
        %loaded = xegpu.load %src_tdesc, %mask <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1>, #xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>>, vector<16xi1> -> vector<1xf32>
        
        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32

        %c0 = arith.constant 0 : i32
        %loaded_modified = vector.insertelement %tid_x_f32, %loaded[%c0 : i32] : vector<1xf32>

        xegpu.store %loaded_modified, %dst_tdesc, %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}> : vector<1xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<chunk_size = 1>, #xegpu.sg_map<wi_layout = [16, 1], wi_data = [1, 1]>>, vector<16xi1> 
        gpu.return 
    }
  }

  func.func @test(%src : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_src = gpu.alloc host_shared () : memref<8x16xf32>
    memref.copy %src, %memref_src : memref<8x16xf32> to memref<8x16xf32>
    %memref_dst = gpu.alloc host_shared () : memref<8x16xf32>
    %srcc = memref.memory_space_cast %memref_src : memref<8x16xf32> to memref<8x16xf32, 1>
    %dstt = memref.memory_space_cast %memref_dst : memref<8x16xf32> to memref<8x16xf32, 1>

    gpu.launch_func @kernel::@load_store_2d blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%srcc : memref<8x16xf32, 1>, %dstt : memref<8x16xf32, 1>)
    return %memref_dst : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<8x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c11_f32, %A[%i, %j] : memref<8x16xf32>
      }
    }
    %B = call @test(%A) : (memref<8x16xf32>) -> memref<8x16xf32>
    %B_cast = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    %A_cast = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%A_cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [11.11{{.*}}]
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]
    
    // CHECK-NEXT: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // CHECK-COUNT-96: 11.11
    // CHECK-NEXT: [11.11{{.*}}]

    memref.dealloc %A : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
