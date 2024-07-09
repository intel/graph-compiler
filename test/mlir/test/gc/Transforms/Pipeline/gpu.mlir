// RUN: gc-opt %s -o=/dev/null 2>&1
// gc-opt --gc-gpu-pipeline="dpas-tile=8,16,16 k-tile=16" -canonicalize %s | FileCheck %s

func.func @matmul(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf32>)
  return
}

// func.func @matmul(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
//   %c1024 = arith.constant 1024 : index
//   %c16 = arith.constant 16 : index
//   %c0 = arith.constant 0 : index
//   %0 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %1 = xegpu.update_nd_offset %0, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16xf32>
//   %3 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %4 = xegpu.update_nd_offset %3, [0, 0] : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %5 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %6 = xegpu.update_nd_offset %5, [0, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %7:3 = scf.for %arg3 = %c0 to %c16 step %c16 iter_args(%arg4 = %2, %arg5 = %4, %arg6 = %6) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>, !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>) {
//     %8 = arith.remui %arg3, %c1024 : index
//     %9 = arith.cmpi eq, %8, %c0 : index
//     scf.if %9 {
//       gpu.barrier
//     }
//     %10 = xegpu.load_nd %arg5 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x8x2xf16>
//     %11 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16x2xf16>
//     %12 = xegpu.update_nd_offset %arg5, [0, 16] : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     %13 = xegpu.update_nd_offset %arg6, [16, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     xegpu.prefetch_nd %12 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     xegpu.prefetch_nd %13 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     %14 = xegpu.dpas %10, %11, %arg4 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
//     scf.yield %14, %12, %13 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>, !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   }
//   xegpu.store_nd %7#0, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   return
// }

func.func @mlp(%arg0: tensor<8x16xf16>, %arg1: tensor<16x16xf16>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<8x16xf16>, tensor<16x16xf16>) 
                                 outs(%1 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %3 = tensor.empty() : tensor<8x16xf32>
  %4 = linalg.add ins(%arg2, %2 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %4 : tensor<8x16xf32> 
}

// func.func @mlp(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>, %arg3: memref<8x16xf32>) {
//   %c1024 = arith.constant 1024 : index
//   %c16 = arith.constant 16 : index
//   %c0 = arith.constant 0 : index
//   %cst = arith.constant 0.000000e+00 : f32
//   %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
//   linalg.fill ins(%cst : f32) outs(%alloc : memref<8x16xf32>)
//   %0 = xegpu.create_nd_tdesc %alloc[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %1 = xegpu.update_nd_offset %0, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %2 = xegpu.load_nd %1 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16xf32>
//   %3 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %4 = xegpu.update_nd_offset %3, [0, 0] : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %5 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %6 = xegpu.update_nd_offset %5, [0, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %7:3 = scf.for %arg4 = %c0 to %c16 step %c16 iter_args(%arg5 = %2, %arg6 = %4, %arg7 = %6) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>, !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>) {
//     %17 = arith.remui %arg4, %c1024 : index
//     %18 = arith.cmpi eq, %17, %c0 : index
//     scf.if %18 {
//       gpu.barrier
//     }
//     %19 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, vnni_axis = 1 : i64}> : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x8x2xf16>
//     %20 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, vnni_axis = 0 : i64}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16x2xf16>
//     %21 = xegpu.update_nd_offset %arg6, [0, 16] : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     %22 = xegpu.update_nd_offset %arg7, [16, 0] : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     xegpu.prefetch_nd %21 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     xegpu.prefetch_nd %22 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//     %23 = xegpu.dpas %19, %20, %arg5 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
//     scf.yield %23, %21, %22 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>, !xegpu.tensor_desc<16x16xf16, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   }
//   xegpu.store_nd %7#0, %1 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %8 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %9 = xegpu.update_nd_offset %8, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %10 = xegpu.load_nd %9  : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16xf32>
//   %11 = xegpu.create_nd_tdesc %alloc[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %12 = xegpu.update_nd_offset %11, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %13 = xegpu.load_nd %12  : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>> -> vector<8x16xf32>
//   %14 = arith.addf %10, %13 : vector<8x16xf32>
//   %15 = xegpu.create_nd_tdesc %arg3[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   %16 = xegpu.update_nd_offset %15, [0, 0] : !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   xegpu.store_nd %14, %16 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true, scattered = false>>
//   memref.dealloc %alloc : memref<8x16xf32>
//   return
// }