module {
  func.func @entry(%arg0: tensor<1x32x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: tensor<1x32x4096xbf16>, %arg3: tensor<1xf32>, %arg4: tensor<4096xbf16>, %arg5: tensor<11008x4096xbf16>, %arg6: tensor<11008x4096xbf16>, %arg7: tensor<4096x11008xbf16>, %arg8: tensor<1xf32>, %arg9: tensor<4096xbf16>) -> (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) {
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
        %67 = arith.addf %in, %init : f32
        linalg.yield %67 : f32
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
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<1x32x11008xbf16>
    %30 = linalg.negf ins(%expanded_10 : tensor<1x32x11008xbf16>) outs(%29 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
    %31 = linalg.exp ins(%30 : tensor<1x32x11008xbf16>) outs(%29 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
    %32 = linalg.add ins(%cst_11, %31 : tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) outs(%29 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
    %33 = linalg.div ins(%cst_11, %32 : tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) outs(%29 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
    %34 = tensor.empty() : tensor<1x32x11008xbf16>
    %35 = linalg.mul ins(%33, %expanded_10 : tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) outs(%34 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
    %collapsed_12 = tensor.collapse_shape %25 [[0, 1], [2]] : tensor<1x32x4096xbf16> into tensor<32x4096xbf16>
    %cst_13 = arith.constant 0.000000e+00 : bf16
    %36 = tensor.empty() : tensor<32x11008xbf16>
    %37 = linalg.fill ins(%cst_13 : bf16) outs(%36 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
    %38 = linalg.matmul_transpose_b ins(%collapsed_12, %arg6 : tensor<32x4096xbf16>, tensor<11008x4096xbf16>) outs(%37 : tensor<32x11008xbf16>) -> tensor<32x11008xbf16>
    %expanded_14 = tensor.expand_shape %38 [[0, 1], [2]] output_shape [1, 32, 11008] : tensor<32x11008xbf16> into tensor<1x32x11008xbf16>
    %39 = tensor.empty() : tensor<1x32x11008xbf16>
    %40 = linalg.mul ins(%35, %expanded_14 : tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) outs(%39 : tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
    %collapsed_15 = tensor.collapse_shape %40 [[0, 1], [2]] : tensor<1x32x11008xbf16> into tensor<32x11008xbf16>
    %cst_16 = arith.constant 0.000000e+00 : bf16
    %41 = tensor.empty() : tensor<32x4096xbf16>
    %42 = linalg.fill ins(%cst_16 : bf16) outs(%41 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %43 = linalg.matmul_transpose_b ins(%collapsed_15, %arg7 : tensor<32x11008xbf16>, tensor<4096x11008xbf16>) outs(%42 : tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %expanded_17 = tensor.expand_shape %43 [[0, 1], [2]] output_shape [1, 32, 4096] : tensor<32x4096xbf16> into tensor<1x32x4096xbf16>
    %44 = tensor.empty() : tensor<1x32x4096xbf16>
    %45 = linalg.add ins(%4, %expanded_17 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%44 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
    %46 = tensor.empty() : tensor<1x32x4096xf32>
    %47 = linalg.copy ins(%45 : tensor<1x32x4096xbf16>) outs(%46 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %cst_18 = arith.constant dense<2.000000e+00> : tensor<1x32x4096xf32>
    %48 = tensor.empty() : tensor<1x32x4096xf32>
    %49 = linalg.powf ins(%47, %cst_18 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%48 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %cst_19 = arith.constant 0.000000e+00 : f32
    %50 = tensor.empty() : tensor<1x32xf32>
    %51 = linalg.fill ins(%cst_19 : f32) outs(%50 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %reduced_20 = linalg.reduce ins(%49 : tensor<1x32x4096xf32>) outs(%51 : tensor<1x32xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %67 = arith.addf %in, %init : f32
        linalg.yield %67 : f32
      }
    %cst_21 = arith.constant dense<4.096000e+03> : tensor<1x32xf32>
    %52 = tensor.empty() : tensor<1x32xf32>
    %53 = linalg.div ins(%reduced_20, %cst_21 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%52 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %expanded_22 = tensor.expand_shape %53 [[0], [1, 2]] output_shape [1, 32, 1] : tensor<1x32xf32> into tensor<1x32x1xf32>
    %54 = tensor.empty() : tensor<1x32x1xf32>
    %broadcasted_23 = linalg.broadcast ins(%arg8 : tensor<1xf32>) outs(%54 : tensor<1x32x1xf32>) dimensions = [0, 1] 
    %55 = tensor.empty() : tensor<1x32x1xf32>
    %56 = linalg.add ins(%expanded_22, %broadcasted_23 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%55 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %cst_24 = arith.constant dense<-5.000000e-01> : tensor<1x32x1xf32>
    %57 = tensor.empty() : tensor<1x32x1xf32>
    %58 = linalg.powf ins(%56, %cst_24 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%57 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %collapsed_25 = tensor.collapse_shape %58 [[0], [1, 2]] : tensor<1x32x1xf32> into tensor<1x32xf32>
    %59 = tensor.empty() : tensor<1x32x4096xf32>
    %broadcasted_26 = linalg.broadcast ins(%collapsed_25 : tensor<1x32xf32>) outs(%59 : tensor<1x32x4096xf32>) dimensions = [2] 
    %60 = tensor.empty() : tensor<1x32x4096xf32>
    %61 = linalg.mul ins(%47, %broadcasted_26 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%60 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %62 = tensor.empty() : tensor<1x32x4096xbf16>
    %63 = linalg.copy ins(%61 : tensor<1x32x4096xf32>) outs(%62 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
    %64 = tensor.empty() : tensor<1x32x4096xbf16>
    %broadcasted_27 = linalg.broadcast ins(%arg9 : tensor<4096xbf16>) outs(%64 : tensor<1x32x4096xbf16>) dimensions = [0, 1] 
    %65 = tensor.empty() : tensor<1x32x4096xbf16>
    %66 = linalg.mul ins(%broadcasted_27, %63 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) outs(%65 : tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
    return %66, %45 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>
  }
}