// RUN: gc-opt %s --gc-gpu-pipeline="is-usm-args=false" \
// RUN: | gc-cpu-runner -e main --entry-point-result=void \
// RUN:   --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime | FileCheck %s
module{

func.func @linalg_matmul(%arg0: tensor<128x256xf16>,
                 %arg1: tensor<64x256xf16>,
                 %arg2: tensor<128x64xf16>) -> tensor<128x64xf16> {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<128x256xf16>, tensor<64x256xf16>)
                     outs(%arg2 : tensor<128x64xf16>) -> tensor<128x64xf16>
  return %0 : tensor<128x64xf16>
}

func.func @main() {
  %0 = arith.constant dense<0.1> : tensor<128x256xf16>
  %1 = arith.constant dense<0.2> : tensor<64x256xf16>
  %2 = arith.constant dense<0.0> : tensor<128x64xf16>
  %gpu_res = call @linalg_matmul(%0, %1, %2) : (tensor<128x256xf16>, tensor<64x256xf16>, tensor<128x64xf16>) -> tensor<128x64xf16>

  %slice = tensor.extract_slice %gpu_res[0, 0][32, 1][1, 1] : tensor<128x64xf16> to tensor<32xf16>
  %cast = tensor.cast %slice : tensor<32xf16> to tensor<*xf16>
  call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()

  return
}

func.func private @printMemrefF16(%ptr : tensor<*xf16>)
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 1 offset = 0 sizes = [32] strides = [64] data = 
// Computed using numpy:
// CHECK-NEXT: [5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719,  5.11719]
