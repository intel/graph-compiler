import numpy
import sys
import argparse

numpy.set_printoptions(threshold=sys.maxsize)


def generate_single_matmul_mlir(M, N, K):
    mat_A = numpy.random.rand(M, K)
    mat_B = numpy.random.rand(K, N)
    mat_C = numpy.dot(mat_A, mat_B)
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry() attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  // Initialize various matrices.
  %da = arith.constant dense<{numpy.array2string(mat_A, separator=", ", max_line_width=100000)}> : tensor<{M}x{K}xf32>
  %db = arith.constant dense<{numpy.array2string(mat_B, separator=", ", max_line_width=100000)}> : tensor<{K}x{N}xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  // Call kernel.
  %C_init = tensor.empty():tensor<{M}x{N}xf32>
  %C = linalg.fill ins(%cst_0 : f32) outs(%C_init : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %0 = linalg.matmul ins(%da, %db : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%C : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %result = arith.constant dense<{numpy.array2string(mat_C, separator=", ", max_line_width=100000)}> : tensor<{M}x{N}xf32>
  %threshold = arith.constant 0.001: f32
  check.expect_almost_eq(%result, %0, %threshold): tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>, f32
  return
{block_end}
    '''
    return mlir_code


def generate_mlir_bf16_2dx4d(M, N, K, tile_m = 32, tile_n = 32, tile_k = 32, dtype_size=2):
    M_block = (M-1) // tile_m + 1
    K_block = (K-1) // tile_k + 1
    N_block = (N-1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M}x{K}xbf16>, %cst: tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16> ) -> tensor<{M}x{N}xbf16> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<{M}x{N}xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<{M}x{N}xbf16>) -> tensor<{M}x{N}xbf16>
    %2 = linalgx.mm2d_vnni ins(%arg0, %cst : tensor<{M}x{K}xbf16>, tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16>) outs(%1 : tensor<{M}x{N}xbf16>) -> tensor<{M}x{N}xbf16>
    return %2 : tensor<{M}x{N}xbf16>
{block_end}
    '''
    return mlir_code

def generate_mlir_bf16_4dx4d(M, N, K, tile_m = 32, tile_n = 32, tile_k = 32, dtype_size=2):
    M_block = (M-1) // tile_m + 1
    K_block = (K-1) // tile_k + 1
    N_block = (N-1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xbf16>, %cst: tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16> ) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    %2 = linalg.generic {block_start}
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d1, d5 * 2 + d6)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d4, d5, d3, d6)>, 
                           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d1, d3)>], 
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
          {block_end} 
          ins(%arg0, %cst : tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xbf16>, tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16>) 
          outs(%1 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>) {block_start}
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %c1 = arith.mulf %in, %in_0 : bf16
      %c2 = arith.addf %out, %c1 : bf16
      linalg.yield %c2 : bf16
    {block_end} -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    return %2 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
{block_end}
    '''
    return mlir_code

def generate_mlir_bf16_4dx4d_generic(M, N, K, tile_m = 32, tile_n = 32, tile_k = 32, dtype_size=2):
    M_block = (M-1) // tile_m + 1
    K_block = (K-1) // tile_k + 1
    N_block = (N-1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xbf16>, %cst: tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16> ) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16> attributes {block_start}llvm.emit_c_interface{block_end} {block_start}
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    %2 = linalg.generic {block_start}indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]{block_end} ins(%arg0, %cst : tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xbf16>, tensor<{N_block}x{K_block}x{tile_k // dtype_size}x{tile_n}x{dtype_size}xbf16>) outs(%1 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>) {block_start}
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_1 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    {block_end} -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
    return %2 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xbf16>
{block_end}
    '''
    return mlir_code

def generate_mlir_f32_4dx4d_generic(M, N, K, tile_m = 32, tile_n = 32, tile_k = 32):
    M_block = (M-1) // tile_m + 1
    K_block = (K-1) // tile_k + 1
    N_block = (N-1) // tile_n + 1
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @main_entry(%arg0: tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xf32>, %cst: tensor<{N_block}x{K_block}x{tile_k}x{tile_n}xf32> ) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32> attributes {block_start}llvm.emit_c_interface{block_end}
 {block_start}
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>) -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
    %2 = linalg.generic {block_start}indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]{block_end} ins(%arg0, %cst : tensor<{M_block}x{K_block}x{tile_m}x{tile_k}xf32>, tensor<{N_block}x{K_block}x{tile_k}x{tile_n}xf32>) outs(%1 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>) {block_start}
        ^bb0(%in: f32, %in_1: f32, %out: f32):
        %3 = arith.mulf %in, %in_1 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
    {block_end} -> tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
    return %2 : tensor<{M_block}x{N_block}x{tile_m}x{tile_n}xf32>
{block_end}
    '''
    return mlir_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLIR Correctness Check")
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--tile_m', type=int, default=4)
    parser.add_argument('--tile_n', type=int, default=4)
    parser.add_argument('--tile_k', type=int, default=4)
    parser.add_argument('--mode', type=str, default="correctness")
    args = parser.parse_args()
    if args.mode == "correctness":
        code = generate_single_matmul_mlir(args.M, args.N, args.K)
    elif args.mode == "bf16_2dx4d":
        code = generate_mlir_bf16_2dx4d(args.M, args.N, args.K, args.tile_m, args.tile_n, args.tile_k)
    elif args.mode == "bf16_4dx4d":
        code = generate_mlir_bf16_4dx4d(args.M, args.N, args.K, args.tile_m, args.tile_n, args.tile_k)
    elif args.mode == "bf16_4dx4d_generic":
        code = generate_mlir_bf16_4dx4d_generic(args.M, args.N, args.K, args.tile_m, args.tile_n, args.tile_k)
    elif args.mode == "f32_4dx4d_generic":
        code = generate_mlir_f32_4dx4d_generic(args.M, args.N, args.K, args.tile_m, args.tile_n, args.tile_k)
    print(code)