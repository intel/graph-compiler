// RUN: gc-opt %s -o=/dev/null 2>&1
// gc-opt %s --gc-cpu-pipeline | gc-cpu-runner -e main -entry-point-result=void | FileCheck --allow-empty %s
// Disabled for now because CI does not have bf16 lib support, once we add our bf16 legalizer, we can enable

func.func @llama2_mlp(%2: tensor<1x32x4096xbf16>, %3: tensor<4096x4096xbf16>, %1 : tensor<1x32x4096xbf16>, %00: tensor<1xf32>, 
                      %26: tensor<4096xbf16>, %28: tensor<11008x4096xbf16>, %33: tensor<11008x4096xbf16>, %37: tensor<4096x11008xbf16>, 
                      %0: tensor<1xf32>, %60: tensor<4096xbf16>) -> (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) {
  %5 = onednn_graph.matmul %2, %3 {transpose_b = true} 
       : (tensor<1x32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x32x4096xbf16>
  %7 = onednn_graph.add %1, %5 : (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>

  %11 = onednn_graph.type_cast %7 : (tensor<1x32x4096xbf16>) -> tensor<1x32x4096xf32>
  %13 = onednn_graph.pow %11 {beta = 2.0 : f32} : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %17 = onednn_graph.reduce_mean %13 {axes = array<i64: -1>, keep_dims = true} 
        : (tensor<1x32x4096xf32>) -> tensor<1x32x1xf32>
  %19 = onednn_graph.add %17, %0 : (tensor<1x32x1xf32>, tensor<1xf32>) -> tensor<1x32x1xf32>
  %20 = onednn_graph.pow %19 {beta = -0.5 : f32} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %21 = onednn_graph.mul %11, %20 : (tensor<1x32x4096xf32>, tensor<1x32x1xf32>) -> tensor<1x32x4096xf32>
  %25 = onednn_graph.type_cast %21 : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xbf16>
  %27 = onednn_graph.mul %26, %25 : (tensor<4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>

  %30 = onednn_graph.matmul %27, %28 {transpose_b = true} 
       : (tensor<1x32x4096xbf16>, tensor<11008x4096xbf16>) -> tensor<1x32x11008xbf16>
  %31 = onednn_graph.sigmoid %30 : (tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
  %32 = onednn_graph.mul %31, %30 : (tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
  %35 = onednn_graph.matmul %27, %33 {transpose_b = true} 
       : (tensor<1x32x4096xbf16>, tensor<11008x4096xbf16>) -> tensor<1x32x11008xbf16>
  %36 = onednn_graph.mul %32, %35 : (tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
  %39 = onednn_graph.matmul %36, %37 {transpose_b = true} 
       : (tensor<1x32x11008xbf16>, tensor<4096x11008xbf16>) -> tensor<1x32x4096xbf16>
  %41 = onednn_graph.add %7, %39 : (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
  %45 = onednn_graph.type_cast %41 : (tensor<1x32x4096xbf16>) -> tensor<1x32x4096xf32>

  %47 = onednn_graph.pow %45 {beta = 2.0 : f32} : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
  %51 = onednn_graph.reduce_mean %47 {axes = array<i64: -1>, keep_dims = true} 
        : (tensor<1x32x4096xf32>) -> tensor<1x32x1xf32>
  %53 = onednn_graph.add %51, %0 : (tensor<1x32x1xf32>, tensor<1xf32>) -> tensor<1x32x1xf32>
  %54 = onednn_graph.pow %53 {beta = -0.5 : f32} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>

  %55 = onednn_graph.mul %45, %54 : (tensor<1x32x4096xf32>, tensor<1x32x1xf32>) -> tensor<1x32x4096xf32>
  %59 = onednn_graph.type_cast %55 : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xbf16>
  %61 = onednn_graph.mul %60, %59 : (tensor<4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>

  return %61, %41 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>
}

func.func @main() {
  %cst0 = arith.constant 1.000000e+00 : bf16
  %cst1 = arith.constant 1.000000e+00 : f32

  %e2  = tensor.empty() : tensor<1x32x4096xbf16>
  %e3  = tensor.empty() : tensor<4096x4096xbf16>
  %e1  = tensor.empty() : tensor<1x32x4096xbf16>
  %e00 = tensor.empty() : tensor<1xf32>
  %e26 = tensor.empty() : tensor<4096xbf16>
  %e28 = tensor.empty() : tensor<11008x4096xbf16>
  %e33 = tensor.empty() : tensor<11008x4096xbf16>
  %e37 = tensor.empty() : tensor<4096x11008xbf16>
  %e0  = tensor.empty() : tensor<1xf32>
  %e60 = tensor.empty() : tensor<4096xbf16>

  %2  = linalg.fill ins(%cst0 : bf16) outs(%e2  : tensor<1x32x4096xbf16> ) -> tensor<1x32x4096xbf16>
  %3  = linalg.fill ins(%cst0 : bf16) outs(%e3  : tensor<4096x4096xbf16> ) -> tensor<4096x4096xbf16>
  %1  = linalg.fill ins(%cst0 : bf16) outs(%e1  : tensor<1x32x4096xbf16> ) -> tensor<1x32x4096xbf16>
  %00 = linalg.fill ins(%cst1 : f32)  outs(%e00 : tensor<1xf32> ) -> tensor<1xf32>
  %26 = linalg.fill ins(%cst0 : bf16) outs(%e26 : tensor<4096xbf16> ) -> tensor<4096xbf16>
  %28 = linalg.fill ins(%cst0 : bf16) outs(%e28 : tensor<11008x4096xbf16> ) -> tensor<11008x4096xbf16>
  %33 = linalg.fill ins(%cst0 : bf16) outs(%e33 : tensor<11008x4096xbf16> ) -> tensor<11008x4096xbf16>
  %37 = linalg.fill ins(%cst0 : bf16) outs(%e37 : tensor<4096x11008xbf16> ) -> tensor<4096x11008xbf16>
  %0  = linalg.fill ins(%cst1 : f32)  outs(%e0  : tensor<1xf32> ) -> tensor<1xf32>
  %60 = linalg.fill ins(%cst0 : bf16) outs(%e60 : tensor<4096xbf16> ) -> tensor<4096xbf16>

  %61, %41 = func.call @llama2_mlp(%2, %3, %1, %00, %26, %28, %33, %37, %0, %60) : 
  (tensor<1x32x4096xbf16>, tensor<4096x4096xbf16>, tensor<1x32x4096xbf16>, tensor<1xf32>, tensor<4096xbf16>, 
  tensor<11008x4096xbf16>, tensor<11008x4096xbf16>, tensor<4096x11008xbf16>, tensor<1xf32>, tensor<4096xbf16>) 
    -> (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>)

  %idx = arith.constant 0 : index
  %ex = tensor.extract %61[%idx, %idx, %idx] : tensor<1x32x4096xbf16>
  %ext = arith.extf %ex : bf16 to f32
  cpuruntime.printf "output[0, 0, 0]: %f\n" %ext : f32

  return
}
// CHECK: output[0, 0, 0]: 1.0
