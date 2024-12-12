// RUN: gc-gpu-runner --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils %s | FileCheck %s

!dtype=f16
!input_memref_type=memref<2x7x32x128x!dtype>
!input_tensor_type=tensor<2x7x32x128x!dtype>
!output_memref_type=memref<2x32x7x128x!dtype>
!output_tensor_type=tensor<2x32x7x128x!dtype>
!cos_sin_cache_memref_type=memref<1x1x2048x128x!dtype>
!cos_sin_cache_tensor_type=tensor<1x1x2048x128x!dtype>
!cos_sin_cache_tensor_shrink_type=tensor<1x1x7x128x!dtype>
!pos_ids_memref_type=memref<1x7xindex>
!pos_ids_tensor_type=tensor<1x7xindex>
#map = affine_map<(xi, yi, zi) -> ((xi * 3 * 4 + yi * 4 + zi) * 2)>
module @fragment_name {
memref.global "private" constant @_cos_cache : !cos_sin_cache_memref_type = dense<3.000000e+00>
memref.global "private" constant @_sin_cache : !cos_sin_cache_memref_type = dense<2.000000e+00>
memref.global "private" constant @_iinput_const : !input_memref_type = dense<3.000000e+00>
memref.global "private" constant @_ipos_ids_const : !pos_ids_memref_type = dense<1>
memref.global "private" constant @_ipos_id_end_const : memref<1xindex> = dense<1>
func.func @RoPE(%iinput: !input_memref_type, %ipos_ids: !pos_ids_memref_type, %ipos_id_end: memref<1xindex>, %out: !output_memref_type) {
    %input = bufferization.to_tensor %iinput restrict : !input_memref_type
    %cos_cache = memref.get_global @_cos_cache : !cos_sin_cache_memref_type
    %sin_cache = memref.get_global @_sin_cache : !cos_sin_cache_memref_type
    %cos_cache_tensor = bufferization.to_tensor %cos_cache restrict : !cos_sin_cache_memref_type
    %sin_cache_tensor = bufferization.to_tensor %sin_cache restrict : !cos_sin_cache_memref_type
    %pos_ids = bufferization.to_tensor %ipos_ids restrict : !pos_ids_memref_type
    %pos_id_end = bufferization.to_tensor %ipos_id_end restrict : memref<1xindex>
    %3 = tensor.empty(): !output_tensor_type

    %transpose_in =  linalg.transpose ins(%input: !input_tensor_type) outs(%3:!output_tensor_type)  permutation = [0, 2, 1, 3]
    
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %cos_cache_slice = tensor.extract_slice %cos_cache_tensor[0, 0, 0, 0] [1, 1, 7, 128] [1, 1, 1, 1] : !cos_sin_cache_tensor_type to !cos_sin_cache_tensor_shrink_type
    %cos_cache_slice2 = tensor.collapse_shape %cos_cache_slice [[0, 1], [2],[3]] : tensor<1x1x7x128x!dtype> into tensor<1x7x128x!dtype>
    %cos_cache_slice3 = tensor.collapse_shape %cos_cache_slice2 [[0, 1], [2]] : tensor<1x7x128x!dtype> into tensor<7x128x!dtype>
    %pos_ids_index=tensor.expand_shape %pos_ids [[0],[1,2]] output_shape [1, 7, 1] : tensor<1x7xindex> into tensor<1x7x1xindex>

    %cos_cache_slice4 = tensor.gather %cos_cache_slice3[%pos_ids_index] gather_dims([0]) : (tensor<7x128x!dtype>, tensor<1x7x1xindex>) -> tensor<1x7x128x!dtype>

    %cos_cache_slice5 = tensor.expand_shape %cos_cache_slice4 [[0,1],[2],[3]] output_shape [1,1,7,128] : tensor<1x7x128x!dtype> into tensor<1x1x7x128x!dtype>
    %cos_cache_slice6 = tensor.collapse_shape %cos_cache_slice5 [[0,1,2],[3]] : tensor<1x1x7x128x!dtype> into tensor<7x128x!dtype>

    %cos_cache_slice7 = linalg.broadcast ins(%cos_cache_slice6: tensor<7x128x!dtype>) outs(%3: !output_tensor_type) dimensions = [0, 1]
    %input_apply_cos_cache = linalg.mul ins(%transpose_in, %cos_cache_slice7:  !output_tensor_type, !output_tensor_type) outs(%3: !output_tensor_type) -> !output_tensor_type
    
    %head_dim = tensor.dim  %transpose_in, %c3 : !output_tensor_type
    %c2 = arith.constant 2 : index
    %half_head_dim = arith.floordivsi %head_dim, %c2 : index
    %transpose_input_first_half = tensor.extract_slice %transpose_in[0, 0, 0, 0][2, 32, 7, 64][1,1,1,1] : !output_tensor_type to tensor<2x32x7x64x!dtype>
    %transpose_input_second_half = tensor.extract_slice %transpose_in[0, 0, 0, %half_head_dim][2, 32, 7, 64][1,1,1,1] : !output_tensor_type to tensor<2x32x7x64x!dtype>
    %cnegative1 = arith.constant dense<-1.000000e+00> : tensor<2x32x7x64x!dtype>
    %empty_tensor = tensor.empty() : tensor<2x32x7x64x!dtype>
    %transpose_input_second_half_opposite = linalg.mul ins(%transpose_input_second_half, %cnegative1:  tensor<2x32x7x64x!dtype>, tensor<2x32x7x64x!dtype>) outs(%empty_tensor: tensor<2x32x7x64x!dtype>) -> tensor<2x32x7x64x!dtype>

    %transformed_input = tensor.concat dim(3) %transpose_input_second_half_opposite, %transpose_input_first_half : (tensor<2x32x7x64x!dtype>, tensor<2x32x7x64x!dtype>) -> !output_tensor_type

    %sin_cache_slice = tensor.extract_slice %sin_cache_tensor[0, 0, 0, 0] [1, 1, 7, 128] [1, 1, 1, 1] : !cos_sin_cache_tensor_type to !cos_sin_cache_tensor_shrink_type
    %sin_cache_slice2 = tensor.collapse_shape %sin_cache_slice [[0, 1], [2],[3]] : tensor<1x1x7x128x!dtype> into tensor<1x7x128x!dtype>
    %sin_cache_slice3 = tensor.collapse_shape %sin_cache_slice2 [[0, 1], [2]] : tensor<1x7x128x!dtype> into tensor<7x128x!dtype>
    %sin_cache_slice4 = tensor.gather %sin_cache_slice3[%pos_ids_index] gather_dims([0]) : (tensor<7x128x!dtype>, tensor<1x7x1xindex>) -> tensor<1x7x128x!dtype>
    
    %sin_cache_slice5 = tensor.expand_shape %sin_cache_slice4 [[0,1],[2],[3]] output_shape [1,1,7,128] : tensor<1x7x128x!dtype> into tensor<1x1x7x128x!dtype>
    %sin_cache_slice6 = tensor.collapse_shape %sin_cache_slice5 [[0,1,2],[3]] : tensor<1x1x7x128x!dtype> into tensor<7x128x!dtype>

    %sin_cache_slice7 = linalg.broadcast ins(%sin_cache_slice6: tensor<7x128x!dtype>) outs(%3: !output_tensor_type) dimensions = [0, 1]
    %input_apply_sin_cache = linalg.mul ins(%transformed_input, %sin_cache_slice7:  !output_tensor_type, !output_tensor_type) outs(%3: !output_tensor_type) -> !output_tensor_type
    
    %result = linalg.add ins(%input_apply_cos_cache, %input_apply_sin_cache:  !output_tensor_type, !output_tensor_type) outs(%3: !output_tensor_type) -> !output_tensor_type
    bufferization.materialize_in_destination %result in restrict writable %out : (!output_tensor_type, !output_memref_type) -> ()
    return
}

func.func @main() {
  %inp = memref.get_global @_iinput_const : !input_memref_type
  %ipos_ids = memref.get_global @_ipos_ids_const : !pos_ids_memref_type
  %ipos_id_end = memref.get_global @_ipos_id_end_const : memref<1xindex>

  %out = memref.alloc() {alignment = 64 : i64} : !output_memref_type

  func.call @RoPE(%inp, %ipos_ids, %ipos_id_end, %out) : (!input_memref_type, !pos_ids_memref_type, memref<1xindex>, !output_memref_type) -> ()

  %out_subview = memref.subview %out[0, 0, 0, 0] [2, 1, 1, 1] [1, 1, 1, 1] : !output_memref_type to memref<2xf16, strided<[28672]>>
  %cast = memref.cast %out_subview : memref<2xf16, strided<[28672]>> to memref<*xf16>
  call @printMemrefF16(%cast) : (memref<*xf16>) -> ()

  return
}

func.func private @printMemrefF16(%ptr : memref<*xf16>)
}

// CHECK: Unranked Memref base@{{(0x)?[-0-9a-fA-F]*}}
// CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [28672] data =
// CHECK-NEXT: [3, 3]
