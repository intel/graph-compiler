// RUN: gc-gpu-runner --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils %s | FileCheck %s

!dtype=i16
!input_memref_type=memref<2x7x32x128x!dtype>
!input_tensor_type=tensor<2x7x32x128x!dtype>
!output_memref_type=memref<2x32x7x128x!dtype>
!output_tensor_type=tensor<2x32x7x128x!dtype>
!cos_sin_cache_memref_type=memref<1x1x7x128x!dtype>
!cos_sin_cache_tensor_type=tensor<1x1x7x128x!dtype>
!cos_sin_cache_tensor_shrink_type=tensor<1x1x7x128x!dtype>
!pos_ids_memref_type=memref<1x7xindex>
module @fragment_name {
memref.global "private" constant @_all_zeroes : !output_memref_type = dense<0>
  func.func @rope1(%iinput: !input_memref_type, %ipos_ids: !pos_ids_memref_type, %out: !output_memref_type,
                  %cos_cache : !cos_sin_cache_memref_type, %sin_cache : !cos_sin_cache_memref_type) {
      %input = bufferization.to_tensor %iinput restrict : !input_memref_type
      %cos_cache_tensor = bufferization.to_tensor %cos_cache restrict : !cos_sin_cache_memref_type
      %sin_cache_tensor = bufferization.to_tensor %sin_cache restrict : !cos_sin_cache_memref_type
      %pos_ids = bufferization.to_tensor %ipos_ids restrict : !pos_ids_memref_type
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
      %cnegative1 = arith.constant dense<-1> : tensor<2x32x7x64x!dtype>
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

  func.func @rope2(%iinput: !input_memref_type, %ipos_ids: !pos_ids_memref_type, %out: !output_memref_type,
                  %cos_cache: !cos_sin_cache_memref_type, %sin_cache: !cos_sin_cache_memref_type) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %cm1 = arith.constant -1 : !dtype

      %input = bufferization.to_tensor %iinput restrict : !input_memref_type
      %cos_cache_tensor = bufferization.to_tensor %cos_cache restrict : !cos_sin_cache_memref_type
      %sin_cache_tensor = bufferization.to_tensor %sin_cache restrict : !cos_sin_cache_memref_type
      %pos_ids = bufferization.to_tensor %ipos_ids restrict : !pos_ids_memref_type
      %tmp = tensor.empty(): !output_tensor_type

      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%tmp : !output_tensor_type) {
        ^bb0(%ignore: !dtype):
          %i0 = linalg.index 0 : index
          %i1 = linalg.index 1 : index
          %i2 = linalg.index 2 : index
          %i3 = linalg.index 3 : index
          %pos = tensor.extract %pos_ids[%c0, %i2] : tensor<1x7xindex>
          %cos = tensor.extract %cos_cache_tensor[%c0, %c0, %pos, %i3] : !cos_sin_cache_tensor_type
          %sin = tensor.extract %sin_cache_tensor[%c0, %c0, %pos, %i3] : !cos_sin_cache_tensor_type
          %in = tensor.extract %input[%i0, %i2, %i1, %i3] : !input_tensor_type
          %cos_val = arith.muli %cos, %in : !dtype

          %cond = arith.cmpi slt, %i3, %c64 : index
          %sin_val = scf.if %cond -> (!dtype) {
            %i3_plus_64 = arith.addi %i3, %c64 : index
            %v = tensor.extract %input[%i0, %i2, %i1, %i3_plus_64] : !input_tensor_type
            %minusv = arith.muli %cm1, %v : !dtype
            %mul = arith.muli %sin, %minusv : !dtype
            scf.yield %mul : !dtype
          } else {
            %i3_minus_64 = arith.addi %i3, %c64 : index
            %v = tensor.extract %input[%i0, %i2, %i1, %i3_minus_64] : !input_tensor_type
            %mul = arith.muli %sin, %v : !dtype
            scf.yield %mul : !dtype
          }

          %sum = arith.addi %cos_val, %sin_val : !dtype
          linalg.yield %sum : !dtype
      } -> !output_tensor_type

      bufferization.materialize_in_destination %result in restrict writable %out : (!output_tensor_type, !output_memref_type) -> ()
      return
  }

    func.func @main() {
      %in_tmp = tensor.empty(): !input_tensor_type
      %input_values = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%in_tmp : !input_tensor_type) {
        ^bb0(%ignore: !dtype):
          %i3 = linalg.index 3 : index
          %val = arith.index_cast %i3 : index to !dtype
          linalg.yield %val : !dtype
      } -> !input_tensor_type
      %inp = memref.alloc() {alignment = 64 : i64} : !input_memref_type
      bufferization.materialize_in_destination %input_values in restrict writable %inp : (!input_tensor_type, !input_memref_type) -> ()

      %ipos_ids_tmp = tensor.empty() : tensor<1x7xindex>
      %ipos_ids_values = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } outs(%ipos_ids_tmp : tensor<1x7xindex>) {
        ^bb0(%ignore: index):
          %c6 = arith.constant 6 : index
          %i1 = linalg.index 1 : index
          %val = arith.subi %c6, %i1 : index
          linalg.yield %i1 : index
      } -> tensor<1x7xindex>
      %ipos_ids = memref.alloc() {alignment = 64 : i64} : !pos_ids_memref_type
      bufferization.materialize_in_destination %ipos_ids_values in restrict writable %ipos_ids : (tensor<1x7xindex>, !pos_ids_memref_type) -> ()

      %cos_cache_tmp = tensor.empty() : !cos_sin_cache_tensor_type
      %sin_cache_tmp = tensor.empty() : !cos_sin_cache_tensor_type
      %cos_cache_values, %sin_cache_values = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%cos_cache_tmp, %sin_cache_tmp : !cos_sin_cache_tensor_type, !cos_sin_cache_tensor_type) {
        ^bb0(%ignore_cos: !dtype, %ignore_sin: !dtype):
          %c3 = arith.constant 3 : !dtype
          %c2 = arith.constant 2 : !dtype
          %i3 = linalg.index 3 : index
          %val = arith.index_cast %i3 : index to !dtype
          %cos = arith.addi %c3, %val : !dtype
          %sin = arith.addi %c2, %val : !dtype
          linalg.yield %cos, %sin : !dtype, !dtype
      } -> (!cos_sin_cache_tensor_type, !cos_sin_cache_tensor_type)
      %cos_cache = memref.alloc() {alignment = 64 : i64} : !cos_sin_cache_memref_type
      %sin_cache = memref.alloc() {alignment = 64 : i64} : !cos_sin_cache_memref_type
      bufferization.materialize_in_destination %cos_cache_values in restrict writable %cos_cache : (!cos_sin_cache_tensor_type, !cos_sin_cache_memref_type) -> ()
      bufferization.materialize_in_destination %sin_cache_values in restrict writable %sin_cache : (!cos_sin_cache_tensor_type, !cos_sin_cache_memref_type) -> ()

      %out1 = memref.alloc() {alignment = 64 : i64} : !output_memref_type
      %start1 = call @nanoTime() : () -> i64
      func.call @rope1(%inp, %ipos_ids, %out1, %cos_cache, %cos_cache) : (!input_memref_type, !pos_ids_memref_type, !output_memref_type, !cos_sin_cache_memref_type, !cos_sin_cache_memref_type) -> ()
      %end1 = call @nanoTime() : () -> i64
      %time1 = arith.subi %end1, %start1 : i64

      %out2 = memref.alloc() {alignment = 64 : i64} : !output_memref_type
      %start2 = call @nanoTime() : () -> i64
      func.call @rope2(%inp, %ipos_ids, %out2, %cos_cache, %cos_cache) : (!input_memref_type, !pos_ids_memref_type, !output_memref_type, !cos_sin_cache_memref_type, !cos_sin_cache_memref_type) -> ()
      %end2 = call @nanoTime() : () -> i64
      %time2 = arith.subi %end2, %start2 : i64

      %out1_tensor = bufferization.to_tensor %out1 restrict : !output_memref_type
      %out2_tensor = bufferization.to_tensor %out2 restrict : !output_memref_type
      %out_buf = tensor.empty(): !output_tensor_type
      %out_tensor = linalg.sub ins(%out1_tensor, %out2_tensor : !output_tensor_type, !output_tensor_type)
                 outs(%out_buf : !output_tensor_type) -> !output_tensor_type
      %out = memref.alloc() {alignment = 64 : i64} : !output_memref_type
      bufferization.materialize_in_destination %out_tensor in restrict writable %out : (!output_tensor_type, !output_memref_type) -> ()

      // %cast = memref.cast %out : !output_memref_type to memref<*xi16>
      // call @printMemrefI16(%cast) : (memref<*xi16>) -> ()

// CHECK: [[TIME1:[0-9]+]]
      llvm.call @printI64(%time1) : (i64) -> ()
      llvm.call @printNewline() : () -> ()
// CHECK: [[TIME2:[0-9]+]]
      llvm.call @printI64(%time2) : (i64) -> ()
      llvm.call @printNewline() : () -> ()

      %all_zeroes = memref.get_global @_all_zeroes : !output_memref_type
      %cast_all_zeroes = memref.cast %all_zeroes : !output_memref_type to memref<*xi16>
      %cast_out = memref.cast %out : !output_memref_type to memref<*xi16>
      %cmp = call @verifyMemRefI16(%cast_all_zeroes, %cast_out) : (memref<*xi16>, memref<*xi16>) -> (i64)
// CHECK: 0
      llvm.call @printI64(%cmp) : (i64) -> ()
      llvm.call @printNewline() : () -> ()

      return
    }

func.func private @printMemrefI16(%ptr : memref<*xi16>)
func.func private @verifyMemRefI16(%a : memref<*xi16>, %b : memref<*xi16>) -> i64 attributes { llvm.emit_c_interface }
func.func private @nanoTime() -> i64
llvm.func @printI64(i64)
llvm.func @printNewline()

}
