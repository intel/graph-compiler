// RUN: gc-opt %s --gc-cpu-pipeline | gc-cpu-runner -e main -entry-point-result=void | FileCheck --allow-empty %s

// bin/gc-opt ../test/gc/Dialect/OneDNNGraph/onednn-graph-llama2.mlir --gc-cpu-pipeline | bin/gc-cpu-runner -e main -entry-point-result=void
// func.func @llama2_mlp(%2: tensor<1x32x4096xbf16>, %3: tensor<4096x4096xbf16>, %1 : tensor<1x32x4096xbf16>, %00: tensor<1xf32>, 
//                       %26: tensor<4096xbf16>, %28: tensor<11008x4096xbf16>, %33: tensor<11008x4096xbf16>, %37: tensor<4096x11008xbf16>, 
//                       %0: tensor<1xf32>, %60: tensor<4096xbf16>) -> (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) {
//   %5 = onednn_graph.matmul %2, %3 {transpose_b = true} 
//        : (tensor<1x32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x32x4096xbf16>
//   %7 = onednn_graph.add %1, %5 : (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>

//   %11 = onednn_graph.type_cast %7 : (tensor<1x32x4096xbf16>) -> tensor<1x32x4096xf32>
//   %13 = onednn_graph.pow %11 {beta = 2.0 : f32} : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
//   %17 = onednn_graph.reduce_mean %13 {axes = array<i64: -1>, keep_dims = true} 
//         : (tensor<1x32x4096xf32>) -> tensor<1x32x1xf32>
//   %19 = onednn_graph.add %17, %0 : (tensor<1x32x1xf32>, tensor<1xf32>) -> tensor<1x32x1xf32>
//   %20 = onednn_graph.pow %19 {beta = -0.5 : f32} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
//   %21 = onednn_graph.mul %11, %20 : (tensor<1x32x4096xf32>, tensor<1x32x1xf32>) -> tensor<1x32x4096xf32>
//   %25 = onednn_graph.type_cast %21 : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xbf16>
//   %27 = onednn_graph.mul %26, %25 : (tensor<4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>

//   %30 = onednn_graph.matmul %27, %28 {transpose_b = true} 
//        : (tensor<1x32x4096xbf16>, tensor<11008x4096xbf16>) -> tensor<1x32x11008xbf16>
//   %31 = onednn_graph.sigmoid %30 : (tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
//   %32 = onednn_graph.mul %31, %30 : (tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
//   %35 = onednn_graph.matmul %27, %33 {transpose_b = true} 
//        : (tensor<1x32x4096xbf16>, tensor<11008x4096xbf16>) -> tensor<1x32x11008xbf16>
//   %36 = onednn_graph.mul %32, %35 : (tensor<1x32x11008xbf16>, tensor<1x32x11008xbf16>) -> tensor<1x32x11008xbf16>
//   %39 = onednn_graph.matmul %36, %37 {transpose_b = true} 
//        : (tensor<1x32x11008xbf16>, tensor<4096x11008xbf16>) -> tensor<1x32x4096xbf16>
//   %41 = onednn_graph.add %7, %39 : (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>
//   %45 = onednn_graph.type_cast %41 : (tensor<1x32x4096xbf16>) -> tensor<1x32x4096xf32>

//   %47 = onednn_graph.pow %45 {beta = 2.0 : f32} : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
//   %51 = onednn_graph.reduce_mean %47 {axes = array<i64: -1>, keep_dims = true} 
//         : (tensor<1x32x4096xf32>) -> tensor<1x32x1xf32>
//   %53 = onednn_graph.add %51, %0 : (tensor<1x32x1xf32>, tensor<1xf32>) -> tensor<1x32x1xf32>
//   %54 = onednn_graph.pow %53 {beta = -0.5 : f32} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>

//   %55 = onednn_graph.mul %45, %54 : (tensor<1x32x4096xf32>, tensor<1x32x1xf32>) -> tensor<1x32x4096xf32>
//   %59 = onednn_graph.type_cast %55 : (tensor<1x32x4096xf32>) -> tensor<1x32x4096xbf16>
//   %61 = onednn_graph.mul %60, %59 : (tensor<4096xbf16>, tensor<1x32x4096xbf16>) -> tensor<1x32x4096xbf16>

//   return %61, %41 : tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>
// }

// func.func @main() {
//   %2 = tensor.empty() : tensor<1x32x4096xbf16>
//   %3 = tensor.empty() : tensor<4096x4096xbf16>
//   %1  = tensor.empty() : tensor<1x32x4096xbf16>
//   %00 = tensor.empty() : tensor<1xf32>
//   %26 = tensor.empty() : tensor<4096xbf16>
//   %28 = tensor.empty() : tensor<11008x4096xbf16>
//   %33 = tensor.empty() : tensor<11008x4096xbf16>
//   %37 = tensor.empty() : tensor<4096x11008xbf16>
//   %0 = tensor.empty() : tensor<1xf32>
//   %60 = tensor.empty() : tensor<4096xbf16>

//   %61, %41 = func.call @llama2_mlp(%2, %3, %1, %00, %26, %28, %33, %37, %0, %60) : 
//   (tensor<1x32x4096xbf16>, tensor<4096x4096xbf16>, tensor<1x32x4096xbf16>, tensor<1xf32>, tensor<4096xbf16>, 
//   tensor<11008x4096xbf16>, tensor<11008x4096xbf16>, tensor<4096x11008xbf16>, tensor<1xf32>, tensor<4096xbf16>) 
//     -> (tensor<1x32x4096xbf16>, tensor<1x32x4096xbf16>)
//   return
// }
// CHECK-NOT: any


// func.func @llama2_mlp_test(%2: tensor<1x32x4096xbf16>, %3: tensor<4096x4096xbf16>) -> tensor<1x32x4096xbf16> {
//   %5 = onednn_graph.matmul %2, %3 {transpose_b = true} 
//        : (tensor<1x32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x32x4096xbf16>

//   return %5 : tensor<1x32x4096xbf16>
// }

// func.func @main() {
//   %2 = tensor.empty() : tensor<1x32x4096xbf16>
//   %3 = tensor.empty() : tensor<4096x4096xbf16>

//   %61 = func.call @llama2_mlp_test(%2, %3) : (tensor<1x32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x32x4096xbf16>
//   return
// }

func.func @llama2_mlp_test(%arg0: tensor<128x512xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x256xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<128x512xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x256xf32>) -> tensor<128x256xf32>
    return %2 : tensor<128x256xf32>
}

func.func @main() {
  %cst = arith.constant 0.000000e+00 : f32

  %0 = tensor.empty() : tensor<128x512xf32>
  %1 = tensor.empty() : tensor<256x512xf32>

  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<256x512xf32>) -> tensor<256x512xf32>

  %61 = func.call @llama2_mlp_test(%2, %3) : (tensor<128x512xf32>, tensor<256x512xf32>) -> tensor<128x256xf32>
  return
}
