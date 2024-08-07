// RUN: gc-opt %s --pass-pipeline='builtin.module(func.func(iterative-tiling-and-fusion{use-cost-model=0 default-tile-size=matmul:{8,16,16}}),eliminate-empty-tensors,empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},drop-equivalent-buffer-results,func.func(finalizing-bufferize),canonicalize,cse,drop-equivalent-buffer-results,expand-realloc,canonicalize,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref,func.func(scf-forall-to-parallel),func.func(linalg-to-xegpu{stages=1 dpas-tile=8,16,16 k-tile=16}),xegpu-fold-alias-ops,func.func(convert-linalg-to-parallel-loops),func.func(gpu-map-parallel-loops),func.func(convert-parallel-loops-to-gpu),func.func(insert-gpu-allocs),gpu-kernel-outlining,canonicalize,set-spirv-capabilities{client-api=opencl},gpu.module(set-spirv-abi-attrs{client-api=opencl}),lower-affine,imex-vector-linearize,gpu.module(convert-xegpu-to-vc),reconcile-unrealized-casts,bf16-to-gpu,gpu.module(convert-func-to-spirv),gpu.module(convert-vector-to-spirv),imex-convert-gpu-to-spirv,spirv.module(spirv-lower-abi-attrs,spirv-update-vce),func.func(llvm-request-c-wrappers),serialize-spirv,convert-vector-to-scf,convert-gpu-to-gpux,convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,convert-math-to-llvm,convert-gpux-to-llvm,convert-index-to-llvm,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | gc-cpu-runner -e main --entry-point-result=void \
// RUN:   --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime | FileCheck %s

module {
  func.func @linalg_mlp(%arg0: tensor<32x4096xf16>, %arg1: tensor<4096x4096xf16>, %arg2 : tensor<32x4096xf16>, 
                        %arg3: tensor<4096x4096xf16>, %arg4 : tensor<32x4096xf16>) -> tensor<32x4096xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<32x4096xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x4096xf16>, tensor<4096x4096xf16>)
                       outs(%1 : tensor<32x4096xf16>) -> (tensor<32x4096xf16>)
    %3 = tensor.empty() : tensor<32x4096xf16>
    %4 = linalg.add ins(%arg2, %2 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                    outs(%3 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %5 = arith.constant dense<0.000000e+00> : tensor<32x4096xf16>
    %6 = tensor.empty() : tensor<32x4096xf16>
    %7 = linalg.max ins(%5, %4 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                    outs(%6 : tensor<32x4096xf16>) -> tensor<32x4096xf16>

    %8 = tensor.empty() : tensor<32x4096xf16>
    %9 = linalg.fill ins(%cst : f16) outs(%0 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %10 = linalg.matmul ins(%7, %arg3 : tensor<32x4096xf16>, tensor<4096x4096xf16>)
                        outs(%9 : tensor<32x4096xf16>) -> (tensor<32x4096xf16>)
    %11 = tensor.empty() : tensor<32x4096xf16>
    %12 = linalg.add ins(%arg4, %10 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                     outs(%11 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %13 = arith.constant dense<0.000000e+00> : tensor<32x4096xf16>
    %14 = tensor.empty() : tensor<32x4096xf16>
    %15 = linalg.max ins(%13, %12 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                     outs(%14 : tensor<32x4096xf16>) -> tensor<32x4096xf16>

    return %15 : tensor<32x4096xf16>
  }

  func.func @main() {
    %cst0 = arith.constant 0.01 : f16
    %cst1 = arith.constant 0.02 : f16

    %0 = tensor.generate {
      ^bb0(%i : index, %j : index):
        tensor.yield %cst0 : f16
    } : tensor<32x4096xf16>
    %1 = tensor.generate {
      ^bb0(%i : index, %j : index):
        tensor.yield %cst0 : f16
    } : tensor<4096x4096xf16>
    %2 = tensor.generate {
      ^bb0(%i : index, %j : index):
        tensor.yield %cst1 : f16
    } : tensor<32x4096xf16>
    %3 = tensor.generate {
      ^bb0(%i : index, %j : index):
        tensor.yield %cst0 : f16
    } : tensor<4096x4096xf16>
    %4 = tensor.generate {
      ^bb0(%i : index, %j : index):
        tensor.yield %cst1 : f16
    } : tensor<32x4096xf16>

    %5 = func.call @linalg_mlp(%0, %1, %2, %3, %4) : (tensor<32x4096xf16>, tensor<4096x4096xf16>, tensor<32x4096xf16>, 
                                                      tensor<4096x4096xf16>, tensor<32x4096xf16>) -> (tensor<32x4096xf16>)

    %6 = tensor.extract_slice %5[0, 0][32, 1][1, 1] : tensor<32x4096xf16> to tensor<32xf16>

    %cast = tensor.cast %6 : tensor<32xf16> to tensor<*xf16>
    call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()

    return
  }

  func.func private @printMemrefF16(%ptr : tensor<*xf16>) attributes { llvm.emit_c_interface }
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 1 offset = 0 sizes = [32] strides = [4096] data =
// CHECK-NEXT: [8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344,  8.02344]
