// RUN: gc-opt --split-input-file --lower-to-tile-vector --mlir-print-ir-after-all -- %s | FileCheck %s

// CHECK-LABEL:   @add_tensor_test0
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4x8x16xf32>
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x8x16xf32>, vector<4x8x16xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x8x16xf32>, vector<4x8x16xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<4x8x16xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true, true]} : vector<4x8x16xf32>, tensor<4x8x16xf32>
func.func @add_tensor_test0(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// CHECK-LABEL:   @add_tensor_test1
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x8x8xf32>
// CHECK:           %[[EXTRACT0:.*]] = tensor.extract_slice %arg0[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x8x8xf32>
// CHECK:           %[[EXTRACT1:.*]] = tensor.extract_slice %arg1[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x8x8xf32>
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<1x8x8xf32>, vector<1x8x8xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<1x8x8xf32>, vector<1x8x8xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<1x8x8xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true, true]} : vector<1x8x8xf32>, tensor<1x8x8xf32>
func.func @add_tensor_test1(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<1x8x8xf32> {
  %0 = tensor.empty() : tensor<1x8x8xf32>
  %1 = tensor.extract_slice %arg0[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x8x8xf32>
  %2 = tensor.extract_slice %arg1[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<4x8x16xf32> to tensor<1x8x8xf32>
  %3 = linalg.add ins(%1, %2 : tensor<1x8x8xf32>, tensor<1x8x8xf32>) outs(%0: tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
  return %3 : tensor<1x8x8xf32>
}

// CHECK-LABEL:   @add_tensor_pack_test2
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x16x16xf32>, vector<4x16x16xf32>
// CHECK:           %[[SHAPECAST0:.*]] = vector.shape_cast %[[READ0]] : vector<4x16x16xf32> to vector<4x4x4x4x4xf32>
// CHECK:           %[[TRANSPOSE0:.*]] = vector.transpose %[[SHAPECAST0]], [1, 0, 3, 2, 4] : vector<4x4x4x4x4xf32> to vector<4x4x4x4x4xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x16x16xf32>, vector<4x16x16xf32>
// CHECK:           %[[SHAPECAST1:.*]] = vector.shape_cast %[[READ1]] : vector<4x16x16xf32> to vector<4x4x4x4x4xf32>
// CHECK:           %[[TRANSPOSE1:.*]] = vector.transpose %[[SHAPECAST1]], [1, 0, 3, 2, 4] : vector<4x4x4x4x4xf32> to vector<4x4x4x4x4xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4x4x4x4x4xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[TRANSPOSE0]], %[[TRANSPOSE1]] : vector<4x4x4x4x4xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true, true, true, true]} : vector<4x4x4x4x4xf32>, tensor<4x4x4x4x4xf32>
func.func @add_tensor_pack_test2(%arg0: tensor<4x16x16xf32>, %arg1: tensor<4x16x16xf32>) -> tensor<4x4x4x4x4xf32> {
  %0 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %1 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %2 = tensor.pack %arg0 outer_dims_perm = [1, 0, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %0 : tensor<4x16x16xf32> -> tensor<4x4x4x4x4xf32>
  %3 = tensor.pack %arg1 outer_dims_perm = [1, 0, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %1 : tensor<4x16x16xf32> -> tensor<4x4x4x4x4xf32>
  %4 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %6 = linalg.add ins(%2, %3 : tensor<4x4x4x4x4xf32>, tensor<4x4x4x4x4xf32>) outs(%4: tensor<4x4x4x4x4xf32>) -> tensor<4x4x4x4x4xf32>
  return %6 : tensor<4x4x4x4x4xf32>
}

// CHECK-LABEL:   @add_tensor_pad_test3
// CHECK:           %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4x16x16xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY0:.*]] = tensor.empty() : tensor<4x16x16xf32>
// CHECK:           %[[FILL0:.*]] = vector.transfer_write %[[cst]], %[[EMPTY0]][%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x16x16xf32>, tensor<4x16x16xf32>
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x16x15xf32>, vector<4x16x15xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %[[READ0]], %[[FILL0]][%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x16x15xf32>, tensor<4x16x16xf32>
// CHECK:           %[[EMPTY1:.*]] = tensor.empty() : tensor<4x16x16xf32>
// CHECK:           %[[FILL1:.*]] = vector.transfer_write %[[cst]], %[[EMPTY1]][%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x16x16xf32>, tensor<4x16x16xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x16x15xf32>, vector<4x16x15xf32>
// CHECK:           %[[WRITE1:.*]] = vector.transfer_write %[[READ1]], %[[FILL1]][%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x16x15xf32>, tensor<4x16x16xf32>
// CHECK:           %[[EMPTY2:.*]] = tensor.empty() : tensor<4x16x16xf32>
// CHECK:           %[[READ2:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x16x16xf32>, vector<4x16x16xf32>
// CHECK:           %[[READ3:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true]} : tensor<4x16x16xf32>, vector<4x16x16xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[READ2]], %[[READ3]] : vector<4x16x16xf32>
// CHECK:           %[[WRITE2:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true, true]} : vector<4x16x16xf32>, tensor<4x16x16xf32>
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

// CHECK-LABEL:   @add_tensor_test4
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true, true, true]} : tensor<12x2x56x56x32xf32>, vector<12x2x56x56x32xf32>
// CHECK:           %[[TRANSPOSE0:.*]] = vector.transpose %[[READ0]], [0, 2, 3, 1, 4] : vector<12x2x56x56x32xf32> to vector<12x56x56x2x32xf32>
// CHECK:           %[[SHAPECAST0:.*]] = vector.shape_cast %[[TRANSPOSE0]] : vector<12x56x56x2x32xf32> to vector<12x56x56x64xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true, true, true, true]} : tensor<12x2x56x56x32xf32>, vector<12x2x56x56x32xf32>
// CHECK:           %[[TRANSPOSE1:.*]] = vector.transpose %[[READ1]], [0, 2, 3, 1, 4] : vector<12x2x56x56x32xf32> to vector<12x56x56x2x32xf32>
// CHECK:           %[[SHAPECAST1:.*]] = vector.shape_cast %[[TRANSPOSE1]] : vector<12x56x56x2x32xf32> to vector<12x56x56x64xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[SHAPECAST0]], %[[SHAPECAST1]] : vector<12x56x56x64xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true, true, true]} : vector<12x56x56x64xf32>, tensor<12x56x56x64xf32>
func.func @add_tensor_test4(%arg0: tensor<12x2x56x56x32xf32>, %arg1: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %2 = tensor.empty() : tensor<12x56x56x64xf32>
  %3 = tensor.unpack %arg1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %4 = tensor.empty() : tensor<12x56x56x64xf32>
  %5 = linalg.add ins(%1, %3 : tensor<12x56x56x64xf32>, tensor<12x56x56x64xf32>) outs(%4: tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
  return %5 : tensor<12x56x56x64xf32>
}

// CHECK-LABEL:   @add_tensor_test5
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CST_0:.*]] = arith.constant dense<1.000000e+00> : vector<1x8xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY0:.*]] tensor.empty() : tensor<1x8xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<1x8xf32>, tensor<1x8xf32>
// CHECK:           %[[extracted_slice:.*]] = tensor.extract_slice %1[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}}  {in_bounds = [true]} : tensor<8xf32>, vector<8xf32>
// CHECK:           %[[SHAPECAST1:.*]] = vector.shape_cast %[[READ1]] : vector<8xf32> to vector<1x1x1x8xf32>
// CHECK:           %[[EMPTY1:.*]] = tensor.empty() : tensor<1x1x1x8xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf32>, tensor<1x1x1x8xf32>
func.func @add_tensor_test5() -> tensor<1x1x1x8xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %init = tensor.empty() : tensor<1x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x8xf32>) -> tensor<1x8xf32>
  %slice = tensor.extract_slice %fill[0, 0] [1, 8] [1, 1] : tensor<1x8xf32> to tensor<8xf32>
  %expand = tensor.expand_shape %slice [[0, 1, 2, 3]] output_shape [1, 1, 1, 8] : tensor<8xf32> into tensor<1x1x1x8xf32>
  return %expand : tensor<1x1x1x8xf32>
}

// CHECK-LABEL:   @tensor_collapse_shape_test6
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}}  {in_bounds = [true, true]} : tensor<2x3xf32>, vector<2x3xf32>
// CHECK:           %[[SHAPECAST1:.*]] = vector.shape_cast %[[READ1]] :  vector<2x3xf32> to vector<6xf32>
// CHECK:           %[[EMPTY1:.*]] = tensor.empty() : tensor<6xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<6xf32>, tensor<6xf32>
func.func @tensor_collapse_shape_test6(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL:   @tensor_static_concat_test7
// CHECK:           %[[C64:.*]] = arith.constant 64 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY0:.*]] = tensor.empty() : tensor<1x1x128xf32>
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}}  {in_bounds = [true, true, true]} : tensor<1x1x64xf32>, vector<1x1x64xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %[[READ0]], %[[EMPTY0]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x1x64xf32>, tensor<1x1x128xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}}  {in_bounds = [true, true, true]} : tensor<1x1x64xf32>, vector<1x1x64xf32>
// CHECK:           %[[WRITE1:.*]] = vector.transfer_write %[[READ1]], %[[WRITE0]][%[[C0]], %[[C0]], %[[C64]]] {in_bounds = [true, true, true]} : vector<1x1x64xf32>, tensor<1x1x128xf32>
func.func @tensor_static_concat_test7(%arg0 : tensor<1x1x64xf32>,
                               %arg1: tensor<1x1x64xf32>) -> tensor<1x1x128xf32> {
  %0 = tensor.concat dim(2) %arg0, %arg1
             : (tensor<1x1x64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x128xf32>
  return %0 : tensor<1x1x128xf32>
}

// CHECK-LABEL:   @fc_relu_test8
// CHECK:           %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<512x512xf32>
// CHECK:           %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[MATMUL:.*]] = linalg.matmul
// CHECK:           %[[READ0:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<512x512xf32>, vector<512x512xf32>
// CHECK:           %[[READ1:.*]] = vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<512x512xf32>, vector<512x512xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[READ0]], %[[READ1]] : vector<512x512xf32>
// CHECK:           %[[MAX0:.*]] = arith.maximumf %[[ADD0]], %[[CST]] : vector<512x512xf32>
// CHECK:           %[[WRITE0:.*]] = vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<512x512xf32>, tensor<512x512xf32>
func.func @fc_relu_test8(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}


func.func @test_pad_dynamic_shape_test9(%arg0: tensor<?x?xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %f0 = arith.constant 0.0 : f32
  // expected-error @+1 {{Attempted to vectorize, but failed}}
 %pad = tensor.pad %arg0 low[0, 0] high[1,1] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %f0 : f32
  } : tensor<?x?xf32> to tensor<4x16xf32>
  return %pad : tensor<4x16xf32>
}

func.func @test_add_dynamic_shape_test10(%arg0: tensor<?x?xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %0 = tensor.empty() : tensor<4x16xf32>
  // expected-error @+1 {{Fail to vectorize.}}
  %1 = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<4x16xf32>) outs(%0: tensor<4x16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}
