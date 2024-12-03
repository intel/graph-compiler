// RUN: gc-gpu-runner --shared-libs=%mlir_runner_utils %s | FileCheck %s

module{

func.func @load_store(%src: memref<8x16xf32>, %dst: memref<8x16xf32>) -> memref<8x16xf32> {
  %constant = arith.constant 1.23 : f32
  %c0 = arith.constant 0 : index
  memref.store %constant, %dst[%c0, %c0] : memref<8x16xf32>

  %0 = memref.extract_aligned_pointer_as_index %src : memref<8x16xf32> -> index
  %1 = arith.index_cast %0 : index to i64
  %ptr_generic = llvm.inttoptr %1 : i64 to !llvm.ptr
  %ptr = llvm.addrspacecast %ptr_generic : !llvm.ptr to !llvm.ptr<1>


  %base_width = arith.constant 16 : i32
  %base_height = arith.constant 16 : i32
  %base_pitch = arith.constant 16 : i32
  %x = arith.constant 0 : i32
  %y = arith.constant 0 : i32

  %loaded = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>

  %dst_ptr_as_idx = memref.extract_aligned_pointer_as_index %dst : memref<8x16xf32> -> index
  %dst_ptr_as_i64 = arith.index_cast %dst_ptr_as_idx : index to i64
  %dst_ptr_generic = llvm.inttoptr %dst_ptr_as_i64 : i64 to !llvm.ptr
  %dst_ptr = llvm.addrspacecast %dst_ptr_generic : !llvm.ptr to !llvm.ptr<1>

  xevm.blockstore2d %dst_ptr, %base_width, %base_height, %base_pitch, %x, %y, %loaded {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)

  return %dst : memref<8x16xf32>
}

func.func @main() {
  %src = memref.alloc() : memref<8x16xf32>
  %dst = memref.alloc() : memref<8x16xf32>
  %gpu_res = call @load_store(%src, %dst) : (memref<8x16xf32>, memref<8x16xf32>) -> memref<8x16xf32>
  %cast = memref.cast %gpu_res : memref<8x16xf32> to memref<*xf32>
  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)

}
