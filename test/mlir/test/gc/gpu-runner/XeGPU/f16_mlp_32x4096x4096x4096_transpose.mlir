// RUN: gc-opt %s --gc-gpu-pipeline="is-usm-args=false" \
// RUN: | gc-cpu-runner -e main --entry-point-result=void \
// RUN:   --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime | FileCheck %s

module {
  func.func @linalg_mlp(%arg0: tensor<32x4096xf16>, %arg1: tensor<4096x4096xf16>, %arg2 : tensor<32x4096xf16>, 
                        %arg3: tensor<4096x4096xf16>, %arg4 : tensor<32x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<32x4096xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<32x4096xf16>, tensor<4096x4096xf16>)
                       outs(%1 : tensor<32x4096xf16>) -> (tensor<32x4096xf16>)
    %3 = tensor.empty() : tensor<32x4096xf16>
    %4 = linalg.add ins(%arg2, %2 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                    outs(%3 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %5 = arith.constant dense<0.000000e+00> : tensor<32x4096xf16>
    %6 = tensor.empty() : tensor<32x4096xf16>
    %7 = linalg.max ins(%5, %4 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                    outs(%6 : tensor<32x4096xf16>) -> tensor<32x4096xf16>

    %8 = tensor.empty() : tensor<32x4096xf16>
    %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %t = tensor.empty() : tensor<4096x4096xf16>
    %transposed = linalg.transpose ins(%arg3 : tensor<4096x4096xf16>) outs(%t : tensor<4096x4096xf16>) permutation = [1, 0]

    %10 = linalg.matmul ins(%7, %transposed : tensor<32x4096xf16>, tensor<4096x4096xf16>)
                        outs(%9 : tensor<32x4096xf16>) -> (tensor<32x4096xf16>)
    %11 = tensor.empty() : tensor<32x4096xf16>
    %12 = linalg.add ins(%arg4, %10 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                     outs(%11 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %13 = arith.constant dense<0.000000e+00> : tensor<32x4096xf16>
    %14 = tensor.empty() : tensor<32x4096xf16>
    %15 = linalg.max ins(%13, %12 : tensor<32x4096xf16>, tensor<32x4096xf16>) 
                     outs(%14 : tensor<32x4096xf16>) -> tensor<32x4096xf16>

    %slice = tensor.extract_slice %15[0, 0][32, 2][1, 1] : tensor<32x4096xf16> to tensor<32x2xf16>
    %cast = tensor.cast %slice : tensor<32x2xf16> to tensor<*xf16>
    call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()

    return
  }

  // generates asymmetric tensor
  func.func @generate_t(%even_val : f16, %odd_val : f16) -> tensor<4096x4096xf16> {
    %0 = tensor.generate {
    ^bb0(%i : index, %j : index):
        %int0 = arith.index_cast %i : index to i32
        %int1 = arith.index_cast %j : index to i32

        %c2 = arith.constant 2 : i32
        %c0 = arith.constant 0 : i32
        %remeinder = arith.remui %int0, %c2 : i32
        %is_even = arith.cmpi eq, %remeinder, %c0 : i32

        %val = scf.if %is_even -> (f16) {
           scf.yield %even_val : f16
        } else {
           scf.yield %odd_val : f16
        }

        tensor.yield %val : f16
    } : tensor<4096x4096xf16>
    return %0 : tensor<4096x4096xf16>
  }

  func.func @main() {
    %0 = arith.constant dense<0.01> : tensor<32x4096xf16>

    %even_v1 = arith.constant 0.02 : f16
    %odd_v1 = arith.constant 0.01 : f16
    %1 = call @generate_t(%even_v1, %odd_v1) : (f16, f16) -> tensor<4096x4096xf16>

    %2 = arith.constant dense<0.02> : tensor<32x4096xf16>
 
    %even_v2 = arith.constant 0.06 : f16
    %odd_v2 = arith.constant 0.03 : f16
    %3 = call @generate_t(%even_v2, %odd_v2) : (f16, f16) -> tensor<4096x4096xf16>

    %4 = arith.constant dense<0.02> : tensor<32x4096xf16>

    func.call @linalg_mlp(%0, %1, %2, %3, %4) : (tensor<32x4096xf16>, tensor<4096x4096xf16>, tensor<32x4096xf16>, 
                                                 tensor<4096x4096xf16>, tensor<32x4096xf16>) -> ()
    return
  }

  func.func private @printMemrefF16(%ptr : tensor<*xf16>) attributes { llvm.emit_c_interface }
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [32, 2] strides = [4096, 1] data = 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375], 
// CHECK-NEXT:  [155.875,   77.9375]
