// RUN: gc-opt --split-input-file -pass-pipeline='builtin.module(func.func(lower-to-tile-vector))' --mlir-print-ir-after-all -- %s

// CHECK-LABEL: func @add_tensor
func.func @add_tensor_test0(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

func.func @add_tensor_test1(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<1x8x8xf32> {
  %0 = tensor.empty() : tensor<1x8x8xf32>
  %1 = tensor.extract_slice %arg0[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x8x8xf32>
  %2 = tensor.extract_slice %arg1[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x8x8xf32>
  %3 = linalg.add ins(%1, %2 : tensor<1x8x8xf32>, tensor<1x8x8xf32>) outs(%0: tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
  return %3 : tensor<1x8x8xf32>
}

func.func @add_tensor_pack_test2(%arg0: tensor<4x16x16xf32>, %arg1: tensor<4x16x16xf32>) -> tensor<4x4x4x4x4xf32> {
  %0 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %1 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %2 = tensor.pack %arg0 outer_dims_perm = [1, 0, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %0 : tensor<4x16x16xf32> -> tensor<4x4x4x4x4xf32>
  %3 = tensor.pack %arg1 outer_dims_perm = [1, 0, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %1 : tensor<4x16x16xf32> -> tensor<4x4x4x4x4xf32>
  %4 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %6 = linalg.add ins(%2, %3 : tensor<4x4x4x4x4xf32>, tensor<4x4x4x4x4xf32>) outs(%4: tensor<4x4x4x4x4xf32>) -> tensor<4x4x4x4x4xf32>
  return %6 : tensor<4x4x4x4x4xf32>
}

func.func @add_tensor_pad_test3(%arg0: tensor<4x16x15xf32>, %arg1: tensor<4x16x15xf32>) -> tensor<4x16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0, 0, 0] high[0, 0, 1]  {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<4x16x15xf32> to tensor<4x16x16xf32>
  %1 = tensor.pad %arg1 low[0, 0, 0] high[0, 0, 1]  {
  ^bb0(%arg5: index, %arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<4x16x15xf32> to tensor<4x16x16xf32>
  %2 = tensor.empty() : tensor<4x16x16xf32>
  %3 = linalg.add ins(%0, %1 : tensor<4x16x16xf32>, tensor<4x16x16xf32>) outs(%2: tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
  return %3 : tensor<4x16x16xf32>
}

func.func @add_tensor_test4(%arg0: tensor<12x2x56x56x32xf32>, %arg1: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %2 = tensor.empty() : tensor<12x56x56x64xf32>
  %3 = tensor.unpack %arg1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %4 = tensor.empty() : tensor<12x56x56x64xf32>
  %5 = linalg.add ins(%1, %3 : tensor<12x56x56x64xf32>, tensor<12x56x56x64xf32>) outs(%4: tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
  return %5 : tensor<12x56x56x64xf32>
}

func.func @add_tensor_test5() -> tensor<1x1x1x8xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %init = tensor.empty() : tensor<1x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x8xf32>) -> tensor<1x8xf32>
  %slice = tensor.extract_slice %fill[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
  %expand = tensor.expand_shape %slice [[0, 1, 2, 3]] : tensor<8xf32> into tensor<1x1x1x8xf32>
  return %expand : tensor<1x1x1x8xf32>
}

func.func @tensor_collapse_shape_test0(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

func.func @tensor_bitcast_test0(%input: tensor<2xi32>) -> tensor<2xf32> {
  %0 = tensor.bitcast %input : tensor<2xi32> to tensor<2xui32>
  %1 = tensor.bitcast %0 : tensor<2xui32> to tensor<2xf32>
  return %1 : tensor<2xf32>
}

func.func @tensor_static_concat_test0(%arg0 : tensor<1x1x64xf32>,
                               %arg1: tensor<1x1x64xf32>) -> tensor<1x1x128xf32> {
  %0 = tensor.concat dim(2) %arg0, %arg1
             : (tensor<1x1x64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x128xf32>
  return %0 : tensor<1x1x128xf32>
}

func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  // expected-remark @below {{elementwise binary}}
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
