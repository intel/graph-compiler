// RUN: gc-opt %s --split-compute-intensive-patterns | FileCheck %s

func.func @mlp(%arg0: tensor<128x512xbf16>, %arg1: tensor<512x64xbf16>, %arg2: tensor<64xbf16>, %arg3: tensor<64x256xbf16>, %arg4: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x64xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<512x64xbf16>) outs(%1 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %3 = tensor.empty() : tensor<128x64xbf16>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<64xbf16>) outs(%3 : tensor<128x64xbf16>) dimensions = [0] 
  %4 = tensor.empty() : tensor<128x64xbf16>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%4 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
  %6 = tensor.empty() : tensor<128x64xbf16>
  %7 = linalg.max ins(%5, %cst_0 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%6 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
  %cst_1 = arith.constant 0.000000e+00 : bf16
  %8 = tensor.empty() : tensor<128x256xbf16>
  %9 = linalg.fill ins(%cst_1 : bf16) outs(%8 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %10 = linalg.matmul ins(%7, %arg3 : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%9 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %11 = tensor.empty() : tensor<128x256xbf16>
  %broadcasted_2 = linalg.broadcast ins(%arg4 : tensor<256xbf16>) outs(%11 : tensor<128x256xbf16>) dimensions = [0] 
  %12 = tensor.empty() : tensor<128x256xbf16>
  %13 = linalg.add ins(%10, %broadcasted_2 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%12 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
  %14 = tensor.empty() : tensor<128x256xbf16>
  %15 = linalg.max ins(%13, %cst_3 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%14 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %15 : tensor<128x256xbf16>
}

func.func @mlp_transpose_a(%arg0: tensor<512x128xbf16>, %arg1: tensor<512x256xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x256xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %2 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<512x128xbf16>, tensor<512x256xbf16>) outs(%1 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %3 = tensor.empty() : tensor<128x256xbf16>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<256xbf16>) outs(%3 : tensor<128x256xbf16>) dimensions = [0] 
  %4 = tensor.empty() : tensor<128x256xbf16>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%4 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
  %6 = tensor.empty() : tensor<128x256xbf16>
  %7 = linalg.max ins(%5, %cst_0 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%6 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %7 : tensor<128x256xbf16>
}

func.func @mlp_transpose_b(%arg0: tensor<128x512xbf16>, %arg1: tensor<256x512xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<128x256xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<128x512xbf16>, tensor<256x512xbf16>) outs(%1 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %3 = tensor.empty() : tensor<128x256xbf16>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<256xbf16>) outs(%3 : tensor<128x256xbf16>) dimensions = [0] 
  %4 = tensor.empty() : tensor<128x256xbf16>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%4 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
  %6 = tensor.empty() : tensor<128x256xbf16>
  %7 = linalg.max ins(%5, %cst_0 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%6 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %7 : tensor<128x256xbf16>
}

func.func @mlp_transpose_a_b(%arg0: tensor<512x128xbf16>, %arg1: tensor<256x512xbf16>, %arg2: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<256x128xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<256x128xbf16>) -> tensor<256x128xbf16>
  %2 = linalg.matmul ins(%arg1, %arg0 : tensor<256x512xbf16>, tensor<512x128xbf16>) outs(%1 : tensor<256x128xbf16>) -> tensor<256x128xbf16>
  %cst_0 = arith.constant 0.000000e+00 : bf16
  %3 = tensor.empty() : tensor<128x256xbf16>
  %4 = linalg.fill ins(%cst_0 : bf16) outs(%3 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %transposed = linalg.transpose ins(%2 : tensor<256x128xbf16>) outs(%4 : tensor<128x256xbf16>) permutation = [1, 0] 
  %5 = tensor.empty() : tensor<128x256xbf16>
  %broadcasted = linalg.broadcast ins(%arg2 : tensor<256xbf16>) outs(%5 : tensor<128x256xbf16>) dimensions = [0] 
  %6 = tensor.empty() : tensor<128x256xbf16>
  %7 = linalg.add ins(%transposed, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%6 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
  %8 = tensor.empty() : tensor<128x256xbf16>
  %9 = linalg.max ins(%7, %cst_1 : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%8 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %9 : tensor<128x256xbf16>
}

func.func @llama2_mlp(%arg0: tensor<1x32x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: tensor<1x32x4096xbf16>, %arg3: tensor<1xf32>, %arg4: tensor<4096xbf16>, %arg5: tensor<11008x4096xbf16>, %arg6: tensor<11008x4096xbf16>, %arg7: tensor<4096x11008xbf16>, %arg8: tensor<1xf32>, %arg9: tensor<4096xbf16>) -> (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x32x4096xbf16> into tensor<32x4096xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<32x4096xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %2 = linalg.matmul_transpose_b ins(%collapsed, %arg1 : tensor<32x4096xbf16>, tensor<4096x4096xbf16>) outs(%1 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %expanded = tensor.expand_shape %2 [[0, 1], [2]] output_shape [1, 32, 4096] : tensor<32x4096xbf16> into tensor<1x32x4096xbf16>
  %3 = tensor.empty() : tensor<1x32x4096xbf16>
  %4 = linalg.add ins(%arg2, %expanded : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%3 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  %5 = tensor.empty() : tensor<1x32x4096xf32>
  %6 = linalg.copy ins(%4 : tensor<1x32x4096xbf16>) outs(%5 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x32x4096xf32>
  %7 = tensor.empty() : tensor<1x32x4096xf32>
  %8 = linalg.powf ins(%6, %cst_0 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%7 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %9 = tensor.empty() : tensor<1x32xf32>
  %10 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<1x32xf32>) -> tensor<1x32xf32>
  %reduced = linalg.reduce ins(%8 : tensor<1x32x4096xf32>) outs(%10 : tensor<1x32xf32>) dimensions = [2] 
    (%in: f32, %init: f32) {
      %64 = arith.addf %in, %init : f32
      linalg.yield %64 : f32
    }
  %cst_2 = arith.constant dense<4.096000e+03> : tensor<1x32xf32>
  %11 = tensor.empty() : tensor<1x32xf32>
  %12 = linalg.div ins(%reduced, %cst_2 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%11 : tensor<1x32xf32>) -> tensor<1x32xf32>
  %expanded_3 = tensor.expand_shape %12 [[0], [1, 2]] output_shape [1, 32, 1] : tensor<1x32xf32> into tensor<1x32x1xf32>
  %13 = tensor.empty() : tensor<1x32x1xf32>
  %broadcasted = linalg.broadcast ins(%arg8 : tensor<1xf32>) outs(%13 : tensor<1x32x1xf32>) dimensions = [0, 1] 
  %14 = tensor.empty() : tensor<1x32x1xf32>
  %15 = linalg.add ins(%expanded_3, %broadcasted : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%14 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %cst_4 = arith.constant dense<-5.000000e-01> : tensor<1x32x1xf32>
  %16 = tensor.empty() : tensor<1x32x1xf32>
  %17 = linalg.powf ins(%15, %cst_4 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%16 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %collapsed_5 = tensor.collapse_shape %17 [[0], [1, 2]] : tensor<1x32x1xf32> into tensor<1x32xf32>
  %18 = tensor.empty() : tensor<1x32x4096xf32>
  %broadcasted_6 = linalg.broadcast ins(%collapsed_5 : tensor<1x32xf32>) outs(%18 : tensor<1x32x4096xf32>) dimensions = [2] 
  %19 = tensor.empty() : tensor<1x32x4096xf32>
  %20 = linalg.mul ins(%6, %broadcasted_6 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%19 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %21 = tensor.empty() : tensor<1x32x4096xbf16>
  %22 = linalg.copy ins(%20 : tensor<1x32x4096xf32>) outs(%21 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  %23 = tensor.empty() : tensor<1x32x4096xbf16>
  %broadcasted_7 = linalg.broadcast ins(%arg4 : tensor<4096xbf16>) outs(%23 : tensor<1x32x4096xbf16>) dimensions = [0, 1] 
  %24 = tensor.empty() : tensor<1x32x4096xbf16>
  %25 = linalg.mul ins(%broadcasted_7, %22 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%24 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  %collapsed_8 = tensor.collapse_shape %25 [[0, 1], [2]] : tensor<1x32x4096xbf16> into tensor<32x4096xbf16>
  %cst_9 = arith.constant 0.000000e+00 : bf16
  %26 = tensor.empty() : tensor<32x11008xbf16>
  %27 = linalg.fill ins(%cst_9 : bf16) outs(%26 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %28 = linalg.matmul_transpose_b ins(%collapsed_8, %arg5 : tensor<32x4096xbf16>, tensor<11008x4096xbf16>) outs(%27 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %expanded_10 = tensor.expand_shape %28 [[0, 1], [2]] output_shape [1, 32, 11008] : tensor<32x11008xbf16> into tensor<1x32x11008xbf16>
  %29 = tensor.empty() : tensor<1x32x11008xbf16>
  %30 = linalgx.sigmoid ins(%expanded_10 : tensor<1x32x11008xbf16>) outs(%29 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
  %31 = tensor.empty() : tensor<1x32x11008xbf16>
  %32 = linalg.mul ins(%30, %expanded_10 : tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) outs(%31 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
  %collapsed_11 = tensor.collapse_shape %25 [[0, 1], [2]] : tensor<1x32x4096xbf16> into tensor<32x4096xbf16>
  %cst_12 = arith.constant 0.000000e+00 : bf16
  %33 = tensor.empty() : tensor<32x11008xbf16>
  %34 = linalg.fill ins(%cst_12 : bf16) outs(%33 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %35 = linalg.matmul_transpose_b ins(%collapsed_11, %arg6 : tensor<32x4096xbf16>, tensor<11008x4096xbf16>) outs(%34 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
  %expanded_13 = tensor.expand_shape %35 [[0, 1], [2]] output_shape [1, 32, 11008] : tensor<32x11008xbf16> into tensor<1x32x11008xbf16>
  %36 = tensor.empty() : tensor<1x32x11008xbf16>
  %37 = linalg.mul ins(%32, %expanded_13 : tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) outs(%36 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
  %collapsed_14 = tensor.collapse_shape %37 [[0, 1], [2]] : tensor<1x32x11008xbf16> into tensor<32x11008xbf16>
  %cst_15 = arith.constant 0.000000e+00 : bf16
  %38 = tensor.empty() : tensor<32x4096xbf16>
  %39 = linalg.fill ins(%cst_15 : bf16) outs(%38 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %40 = linalg.matmul_transpose_b ins(%collapsed_14, %arg7 : tensor<32x11008xbf16>, tensor<4096x11008xbf16>) outs(%39 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
  %expanded_16 = tensor.expand_shape %40 [[0, 1], [2]] output_shape [1, 32, 4096] : tensor<32x4096xbf16> into tensor<1x32x4096xbf16>
  %41 = tensor.empty() : tensor<1x32x4096xbf16>
  %42 = linalg.add ins(%4, %expanded_16 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%41 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  %43 = tensor.empty() : tensor<1x32x4096xf32>
  %44 = linalg.copy ins(%42 : tensor<1x32x4096xbf16>) outs(%43 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %cst_17 = arith.constant dense<2.000000e+00> : tensor<1x32x4096xf32>
  %45 = tensor.empty() : tensor<1x32x4096xf32>
  %46 = linalg.powf ins(%44, %cst_17 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%45 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %cst_18 = arith.constant 0.000000e+00 : f32
  %47 = tensor.empty() : tensor<1x32xf32>
  %48 = linalg.fill ins(%cst_18 : f32) outs(%47 : tensor<1x32xf32>) -> tensor<1x32xf32>
  %reduced_19 = linalg.reduce ins(%46 : tensor<1x32x4096xf32>) outs(%48 : tensor<1x32xf32>) dimensions = [2] 
    (%in: f32, %init: f32) {
      %64 = arith.addf %in, %init : f32
      linalg.yield %64 : f32
    }
  %cst_20 = arith.constant dense<4.096000e+03> : tensor<1x32xf32>
  %49 = tensor.empty() : tensor<1x32xf32>
  %50 = linalg.div ins(%reduced_19, %cst_20 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%49 : tensor<1x32xf32>) -> tensor<1x32xf32>
  %expanded_21 = tensor.expand_shape %50 [[0], [1, 2]] output_shape [1, 32, 1] : tensor<1x32xf32> into tensor<1x32x1xf32>
  %51 = tensor.empty() : tensor<1x32x1xf32>
  %broadcasted_22 = linalg.broadcast ins(%arg8 : tensor<1xf32>) outs(%51 : tensor<1x32x1xf32>) dimensions = [0, 1] 
  %52 = tensor.empty() : tensor<1x32x1xf32>
  %53 = linalg.add ins(%expanded_21, %broadcasted_22 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%52 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %cst_23 = arith.constant dense<-5.000000e-01> : tensor<1x32x1xf32>
  %54 = tensor.empty() : tensor<1x32x1xf32>
  %55 = linalg.powf ins(%53, %cst_23 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%54 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %collapsed_24 = tensor.collapse_shape %55 [[0], [1, 2]] : tensor<1x32x1xf32> into tensor<1x32xf32>
  %56 = tensor.empty() : tensor<1x32x4096xf32>
  %broadcasted_25 = linalg.broadcast ins(%collapsed_24 : tensor<1x32xf32>) outs(%56 : tensor<1x32x4096xf32>) dimensions = [2] 
  %57 = tensor.empty() : tensor<1x32x4096xf32>
  %58 = linalg.mul ins(%44, %broadcasted_25 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%57 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %59 = tensor.empty() : tensor<1x32x4096xbf16>
  %60 = linalg.copy ins(%58 : tensor<1x32x4096xf32>) outs(%59 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  %61 = tensor.empty() : tensor<1x32x4096xbf16>
  %broadcasted_26 = linalg.broadcast ins(%arg9 : tensor<4096xbf16>) outs(%61 : tensor<1x32x4096xbf16>) dimensions = [0, 1] 
  %62 = tensor.empty() : tensor<1x32x4096xbf16>
  %63 = linalg.mul ins(%broadcasted_26, %60 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%62 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  return %63, %42 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>
}
