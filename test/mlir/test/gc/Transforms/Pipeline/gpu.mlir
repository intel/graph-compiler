// RUN: gc-opt --gc-gpu-pipeline %s | FileCheck %s

func.func @mlp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x16xf32>) 
                                 outs(%1 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %3 = tensor.empty() : tensor<8x16xf32>
  %4 = linalg.add ins(%arg2, %2 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %4 : tensor<8x16xf32> 
}

// func.func @mlp(%arg0: memref<8x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<8x16xf32>, %arg3: memref<8x16xf32>) {
//   %c0 = arith.constant 0 : index
//   %cst = arith.constant 0.000000e+00 : f32
//   %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
//   linalg.fill ins(%cst : f32) outs(%alloc : memref<8x16xf32>)
//   linalg.matmul_transpose_b ins(%arg0, %arg1 : memref<8x16xf32>, memref<16x16xf32>) outs(%alloc : memref<8x16xf32>)
//   %0 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %1 = xegpu.update_nd_offset %0, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %2 = xegpu.load_nd %1  : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16xf32>
//   %3 = xegpu.create_nd_tdesc %alloc[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %4 = xegpu.update_nd_offset %3, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %5 = xegpu.load_nd %4  : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16xf32>
//   %6 = arith.addf %2, %5 : vector<8x16xf32>
//   %7 = xegpu.create_nd_tdesc %arg3[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %8 = xegpu.update_nd_offset %7, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   xegpu.store_nd %6, %8 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   memref.dealloc %alloc : memref<8x16xf32>
//   return
// }