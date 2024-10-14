// RUN: gc-gpu-runner --shared-libs=%mlir_runner_utils %s | FileCheck %s

module {
  func.func @linalg_mlp(%arg0: tensor<32x4096xf16>, %arg1: tensor<4096x4096xf16>, %arg2 : tensor<32x4096xf16>,
                        %arg3: tensor<4096x4096xf16>, %arg4 : tensor<32x4096xf16>) {
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
    %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %10 = linalg.matmul ins(%7, %arg3 : tensor<32x4096xf16>, tensor<4096x4096xf16>)
                        outs(%9 : tensor<32x4096xf16>) -> (tensor<32x4096xf16>)
    %11 = tensor.empty() : tensor<32x4096xf16>
    %12 = linalg.add ins(%arg4, %10 : tensor<32x4096xf16>, tensor<32x4096xf16>)
                     outs(%11 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %13 = arith.constant dense<0.000000e+00> : tensor<32x4096xf16>
    %14 = tensor.empty() : tensor<32x4096xf16>
    %15 = linalg.max ins(%13, %12 : tensor<32x4096xf16>, tensor<32x4096xf16>)
                     outs(%14 : tensor<32x4096xf16>) -> tensor<32x4096xf16>

    %slice = tensor.extract_slice %15[0, 0][32, 1][1, 1] : tensor<32x4096xf16> to tensor<32xf16>
    %cast = tensor.cast %slice : tensor<32xf16> to tensor<*xf16>
    call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()

    return
  }

  func.func @main() {
    %0 = arith.constant dense<0.01> : tensor<32x4096xf16>
    %1 = arith.constant dense<0.01> : tensor<4096x4096xf16>
    %2 = arith.constant dense<0.02> : tensor<32x4096xf16>
    %3 = arith.constant dense<0.01> : tensor<4096x4096xf16>
    %4 = arith.constant dense<0.02> : tensor<32x4096xf16>

    func.call @linalg_mlp(%0, %1, %2, %3, %4) : (tensor<32x4096xf16>, tensor<4096x4096xf16>, tensor<32x4096xf16>,
                                                 tensor<4096x4096xf16>, tensor<32x4096xf16>) -> ()
    return
  }

  func.func private @printMemrefF16(%ptr : tensor<*xf16>) attributes { llvm.emit_c_interface }
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 1 offset = 0 sizes = [32] strides = [4096] data =
// CHECK-NEXT: [17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625, 17.625]
