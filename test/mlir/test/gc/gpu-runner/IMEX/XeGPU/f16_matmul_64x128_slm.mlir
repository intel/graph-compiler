// RUN: gc-gpu-runner --shared-libs=%mlir_runner_utils %s | FileCheck %s

module @fragment_name {
  // This kernel requires using SLM
  func.func @entry(%0: tensor<64x128xf16>, %1: tensor<128x128xf16>, %2: tensor<64x128xf16>, %res: tensor<64x128xf16>) -> tensor<64x128xf16> {
    %3 = tensor.empty() : tensor<128x128xf16>
    %4 = tensor.empty() : tensor<64x128xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %6 = linalg.matmul ins(%0, %1 : tensor<64x128xf16>, tensor<128x128xf16>) outs(%5 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %7 = tensor.empty() : tensor<64x128xf16>
    %8 = linalg.add ins(%6, %2 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%7 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %9 = tensor.empty() : tensor<64x128xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %10 = linalg.fill ins(%cst_0 : f16) outs(%9 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %11 = linalg.max ins(%8, %10 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%res : tensor<64x128xf16>) -> tensor<64x128xf16>
    return %11 : tensor<64x128xf16>
  }

  func.func @get_value(%i : index, %j : index, %even_val : f16, %odd_val : f16) -> f16 {
    %int0 = arith.index_cast %i : index to i32

    %c2 = arith.constant 2 : i32
    %remeinder = arith.remui %int0, %c2 : i32
    %c0i = arith.constant 0 : i32
    %is_even = arith.cmpi eq, %remeinder, %c0i : i32

    %val = scf.if %is_even -> (f16) {
        scf.yield %even_val : f16
    } else {
        scf.yield %odd_val : f16
    }
    return %val : f16
  }

  // generates asymmetric tensor
  func.func @generate_t(%even_val : f16, %odd_val : f16) -> tensor<64x128xf16> {
    %0 = tensor.generate {
    ^bb0(%i : index, %j : index):
        %val = func.call @get_value(%i, %j, %even_val, %odd_val) : (index, index, f16, f16) -> f16
        tensor.yield %val : f16
    } : tensor<64x128xf16>
    return %0 : tensor<64x128xf16>
  }

  func.func @generate_t_wide(%even_val : f16, %odd_val : f16) -> tensor<128x128xf16> {
    %0 = tensor.generate {
    ^bb0(%i : index, %j : index):
        %val = func.call @get_value(%i, %j, %even_val, %odd_val) : (index, index, f16, f16) -> f16
        tensor.yield %val : f16
    } : tensor<128x128xf16>
    return %0 : tensor<128x128xf16>
  }

  func.func @main() {
    %a0 = arith.constant 0.1 : f16
    %b0 = arith.constant 0.2 : f16
    %0 = call @generate_t(%a0, %b0) : (f16, f16) -> tensor<64x128xf16>

    %a1 = arith.constant 0.3 : f16
    %b1 = arith.constant 0.4 : f16
    %1 = call @generate_t_wide(%a1, %b1) : (f16, f16) -> tensor<128x128xf16>

    %a2 = arith.constant 0.5 : f16
    %b2 = arith.constant 0.6 : f16
    %2 = call @generate_t(%a2, %b2) : (f16, f16) -> tensor<64x128xf16>

    %3 = arith.constant dense<0.0> : tensor<64x128xf16>
    %gpu_res = call @entry(%0, %1, %2, %3) : (tensor<64x128xf16>, tensor<128x128xf16>, tensor<64x128xf16>, tensor<64x128xf16>) -> (tensor<64x128xf16>)
    %slice = tensor.extract_slice %gpu_res[0, 0][16, 16][1, 1] : tensor<64x128xf16> to tensor<16x16xf16>
    %cast = tensor.cast %slice : tensor<16x16xf16> to tensor<*xf16>
    call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()
    return
}

func.func private @printMemrefF16(%ptr : tensor<*xf16>)
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [16, 16] strides = [128, 1] data =
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625], 
// CHECK-NEXT: [4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047,   4.98047], 
// CHECK-NEXT: [9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625,   9.5625]
