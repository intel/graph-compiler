// RUN: gc-opt -split-input-file -verify-diagnostics %s

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op transpose and vnni_transform are mutually exclusive}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=16, v_blocks=1, transpose=true, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op expecting tile_height to be between 1 and 32}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=64, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op expecting tile_width to be between 4 and 64}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=128, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op expecting v_blocks to be 1, 2, or 4}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=6, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<48xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op tile_width * v_blocks should be less than or equal to 64 for 8 bit elements}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=4, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op transpose is only supported for 32 and 64 bit elements}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=true, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op vnni_transform is only supported for 8 and 16 bit elements}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'xevm.blockload2d' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64, tile_width=4, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op expecting tile_height to be 1, 2, 4, 8, 16, or 32}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=24, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<24xi16>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op tile_width when vnni_transform is true should be equal to subgroup size (16 elements)}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=true, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<1xi32>
  llvm.return
}

// -----

llvm.func @blockload2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'xevm.blockload2d' op expecting result element type to be 32 bits}}
  %0 = xevm.blockload2d %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<64xi8>) {
  // expected-error @+1 {{'xevm.blockstore2d' op expecting tile_height to be between 1 and 8}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=64, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<64xi8>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi8>) {
  // expected-error @+1 {{'xevm.blockstore2d' op expecting tile_width to be between 4 and 64}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=2, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi8>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  // expected-error @+1 {{'xevm.blockstore2d' op expecting v_blocks to be 1}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'xevm.blockstore2d' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<32xi16>) {
  // expected-error @+1 {{'xevm.blockstore2d' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=64, tile_width=4, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<32xi16>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<64xi8>) {
  // expected-error @+1 {{'xevm.blockstore2d' op expecting tile_height to be 1, 2, 4, 8, 16, or 32}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=6, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<64xi8>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi8>) {
  // expected-error @+1 {{'xevm.blockstore2d' op tile_width for 8 bit elements should be equal to 16 or 32}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=8, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi8>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // expected-error @+1 {{'xevm.blockstore2d' op tile_width for 16 bit elements should be equal to 16}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16, tile_width=32, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}

// -----

llvm.func @blockstore2d(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'xevm.blockstore2d' op tile_width for 32 bit elements should be equal to 16}}
  xevm.blockstore2d %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, l1_cache_control=Default, l3_cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}
