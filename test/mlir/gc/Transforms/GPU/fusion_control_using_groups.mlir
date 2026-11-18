// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-parallel --split-input-file | FileCheck %s

// Group membership compares IDs, not tile sizes. These describe the iteration
// domains of exp(A), matmul, and the broadcast bias addition respectively.
#pre0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>

#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// exp(A) @ B + bias: the pre-op, fill, matmul, and post-op all fuse into one
// workgroup. The exp is computed inside the reduction loop.
// CHECK-LABEL: func.func @all_same_group(
// CHECK-SAME:  gc.num_kernels = 1 :
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 :
// CHECK-DAG:     %[[K:.*]] = arith.constant 256 :
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 64 :
// CHECK:         %[[RESULT:.*]] = scf.forall
// CHECK:           %[[INIT:.*]] = linalg.fill
// CHECK-NEXT:      %[[REDUCE:.*]] = scf.for {{.*}} = %[[ZERO]] to %[[K]] step %[[STEP]] iter_args(%[[ACC:.*]] = %[[INIT]])
// CHECK:             %[[PRE:.*]] = linalg.generic
// CHECK-NEXT:        ^bb0(%[[X:[^:]+]]:
// CHECK-NEXT:          %[[EXP:.*]] = math.exp %[[X]]
// CHECK-NEXT:          linalg.yield %[[EXP]]
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[RHS:.*]] = tensor.extract_slice
// CHECK-NEXT:        %[[MM:.*]] = linalg.matmul {{.*}}ins(%[[PRE]], %[[RHS]] {{.*}}outs(%[[ACC]]
// CHECK-NEXT:        scf.yield %[[MM]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK:           %[[POST:.*]] = linalg.generic {{.*}}ins(%[[REDUCE]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "all_same_group_kernel", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT]]
func.func @all_same_group(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                          %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {gc.fusion_group = #pre0} {
  ^bb0(%x: f16, %out: f16):
    %exp = math.exp %x : f16
    linalg.yield %exp : f16
  } -> tensor<64x256xf16>
  %mm_empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%mm_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm = linalg.matmul {gc.fusion_group = #mm0}
    ins(%pre, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_empty = tensor.empty() : tensor<64x64xf16>
  %post = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post : tensor<64x64xf16>
}

// -----

#pre1 = #gc.fusion_group<1, tile_sizes = [32, 64]>
#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// The pre-op belongs to another group and must stay outside the matmul loop.
// The post-op still fuses. TileParallel subsequently creates a separate kernel
// for the pre-op without crossing the boundary.
// CHECK-LABEL: func.func @pre_other_group(
// CHECK-SAME:  %[[A:[^:]+]]: {{[^,]+}}, %[[B:[^:]+]]:
// CHECK-SAME:  gc.num_kernels = 2 :
// CHECK:         %[[PRE_RESULT:.*]] = scf.forall
// CHECK:           %[[PRE:.*]] = linalg.generic
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]:
// CHECK-NEXT:        %[[EXP:.*]] = math.exp %[[X]]
// CHECK-NEXT:        linalg.yield %[[EXP]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[PRE]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : {{[^,}]+}}}
// CHECK:         %[[RESULT:.*]] = scf.forall
// CHECK:           %[[REDUCE:.*]] = scf.for
// CHECK-DAG:         %[[LHS:.*]] = tensor.extract_slice %[[PRE_RESULT]][
// CHECK-DAG:         %[[RHS:.*]] = tensor.extract_slice %[[B]][
// CHECK-NEXT:        %[[MM:.*]] = linalg.matmul {{.*}}ins(%[[LHS]], %[[RHS]]
// CHECK-NEXT:        scf.yield %[[MM]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK:           %[[POST:.*]] = linalg.generic {{.*}}ins(%[[REDUCE]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT]]
func.func @pre_other_group(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                           %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {gc.fusion_group = #pre1} {
  ^bb0(%x: f16, %out: f16):
    %exp = math.exp %x : f16
    linalg.yield %exp : f16
  } -> tensor<64x256xf16>
  %mm_empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%mm_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm = linalg.matmul {gc.fusion_group = #mm0}
    ins(%pre, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_empty = tensor.empty() : tensor<64x64xf16>
  %post = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post : tensor<64x64xf16>
}

// -----

#pre0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#post1 = #gc.fusion_group<1, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// The post-op is in another group: it must not be moved into the matmul loop.
// CHECK-LABEL: func.func @post_other_group(
// CHECK-SAME:  gc.num_kernels = 2 :
// CHECK:         %[[MM_RESULT:.*]] = scf.forall
// CHECK:           %[[REDUCE:.*]] = scf.for
// CHECK:             %[[PRE:.*]] = linalg.generic
// CHECK-NEXT:        ^bb0(%[[X:[^:]+]]:
// CHECK-NEXT:          %[[EXP:.*]] = math.exp %[[X]]
// CHECK-NEXT:          linalg.yield %[[EXP]]
// CHECK-NEXT:        }
// CHECK:             %[[MM:.*]] = linalg.matmul {{.*}}ins(%[[PRE]],
// CHECK-NEXT:        scf.yield %[[MM]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK-NEXT:      scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[REDUCE]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK:         %[[RESULT:.*]] = scf.forall
// CHECK:           %[[MM_TILE:.*]] = tensor.extract_slice %[[MM_RESULT]][
// CHECK:           %[[POST:.*]] = linalg.generic {{.*}}ins(%[[MM_TILE]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT]]
func.func @post_other_group(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                            %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {gc.fusion_group = #pre0} {
  ^bb0(%x: f16, %out: f16):
    %exp = math.exp %x : f16
    linalg.yield %exp : f16
  } -> tensor<64x256xf16>
  %mm_empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%mm_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm = linalg.matmul {gc.fusion_group = #mm0}
    ins(%pre, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_empty = tensor.empty() : tensor<64x64xf16>
  %post = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post1} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post : tensor<64x64xf16>
}

// -----

#pre1 = #gc.fusion_group<1, tile_sizes = [32, 64]>
#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#post2 = #gc.fusion_group<2, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// Three different groups become three kernels after both tiling passes.
// CHECK-LABEL: func.func @all_different_groups(
// CHECK-SAME:  %[[A:[^:]+]]: {{[^,]+}}, %[[B:[^:]+]]:
// CHECK-SAME:  gc.num_kernels = 3 :
// CHECK:         %[[PRE_RESULT:.*]] = scf.forall
// CHECK:           %[[PRE:.*]] = linalg.generic
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]:
// CHECK-NEXT:        %[[EXP:.*]] = math.exp %[[X]]
// CHECK-NEXT:        linalg.yield %[[EXP]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[PRE]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : {{[^,}]+}}}
// CHECK:         %[[MM_RESULT:.*]] = scf.forall
// CHECK:           %[[REDUCE:.*]] = scf.for
// CHECK-DAG:         %[[LHS:.*]] = tensor.extract_slice %[[PRE_RESULT]][
// CHECK-DAG:         %[[RHS:.*]] = tensor.extract_slice %[[B]][
// CHECK-NEXT:        %[[MM:.*]] = linalg.matmul {{.*}}ins(%[[LHS]], %[[RHS]]
// CHECK-NEXT:        scf.yield %[[MM]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK-NEXT:      scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[REDUCE]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK:         %[[RESULT:.*]] = scf.forall
// CHECK:           %[[MM_TILE:.*]] = tensor.extract_slice %[[MM_RESULT]][
// CHECK:           %[[POST:.*]] = linalg.generic {{.*}}ins(%[[MM_TILE]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 2 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT]]
func.func @all_different_groups(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                                %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {gc.fusion_group = #pre1} {
  ^bb0(%x: f16, %out: f16):
    %exp = math.exp %x : f16
    linalg.yield %exp : f16
  } -> tensor<64x256xf16>
  %mm_empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%mm_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm = linalg.matmul {gc.fusion_group = #mm0}
    ins(%pre, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_empty = tensor.empty() : tensor<64x64xf16>
  %post = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post2} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post : tensor<64x64xf16>
}

// -----

#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// An unannotated producer or consumer cannot join an annotated kernel.
// CHECK-LABEL: func.func @mixed_annotations(
// CHECK-SAME:  %[[A:[^:]+]]: {{[^,]+}}, %[[B:[^:]+]]:
// CHECK-SAME:  gc.num_kernels = 3 :
// CHECK:         %[[PRE_RESULT:.*]] = scf.forall
// CHECK:           %[[PRE:.*]] = linalg.generic
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]:
// CHECK-NEXT:        %[[EXP:.*]] = math.exp %[[X]]
// CHECK-NEXT:        linalg.yield %[[EXP]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[PRE]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}"}
// CHECK:         %[[MM_RESULT:.*]] = scf.forall
// CHECK:           %[[REDUCE:.*]] = scf.for
// CHECK-DAG:         %[[LHS:.*]] = tensor.extract_slice %[[PRE_RESULT]][
// CHECK-DAG:         %[[RHS:.*]] = tensor.extract_slice %[[B]][
// CHECK-NEXT:        %[[MM:.*]] = linalg.matmul {{.*}}ins(%[[LHS]], %[[RHS]]
// CHECK-NEXT:        scf.yield %[[MM]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK-NEXT:      scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[REDUCE]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK:         %[[RESULT:.*]] = scf.forall
// CHECK:           %[[MM_TILE:.*]] = tensor.extract_slice %[[MM_RESULT]][
// CHECK:           %[[POST:.*]] = linalg.generic {{.*}}ins(%[[MM_TILE]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}"}
// CHECK-NEXT:    return %[[RESULT]]
func.func @mixed_annotations(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                             %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>) {
  ^bb0(%x: f16, %out: f16):
    %exp = math.exp %x : f16
    linalg.yield %exp : f16
  } -> tensor<64x256xf16>
  %mm_empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%mm_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm = linalg.matmul {gc.fusion_group = #mm0}
    ins(%pre, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_empty = tensor.empty() : tensor<64x64xf16>
  %post = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty : tensor<64x64xf16>) {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post : tensor<64x64xf16>
}

// -----

#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>
#vector = affine_map<(n) -> (n)>

// Consumer fusion exposes another producer (the bias exp). The subsequent
// fuseProducers() path must enforce the group boundary too.
// CHECK-LABEL: func.func @post_input_other_group(
// CHECK-SAME:  %[[A:[^:]+]]: {{[^,]+}}, %[[B:[^:]+]]:
// CHECK-SAME:  gc.num_kernels = 2 :
// CHECK:         %[[BIAS_PRE:.*]] = linalg.generic
// CHECK-NEXT:    ^bb0(%[[X:[^:]+]]:
// CHECK-NEXT:      %[[EXP:.*]] = math.exp %[[X]]
// CHECK-NEXT:      linalg.yield %[[EXP]]
// CHECK-NEXT:    }
// CHECK:         %[[RESULT:.*]] = scf.forall
// CHECK:           %[[REDUCE:.*]] = scf.for
// CHECK-DAG:         %[[LHS:.*]] = tensor.extract_slice %[[A]][
// CHECK-DAG:         %[[RHS:.*]] = tensor.extract_slice %[[B]][
// CHECK-NEXT:        %[[MM:.*]] = linalg.matmul {{.*}}ins(%[[LHS]], %[[RHS]]
// CHECK-NEXT:        scf.yield %[[MM]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK:           %[[POST:.*]] = linalg.generic {{.*}}ins(%[[REDUCE]], %[[BIAS_PRE]]
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "post_input_other_group_kernel", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT]]
func.func @post_input_other_group(%a: tensor<64x256xf16>,
                                 %b: tensor<256x64xf16>,
                                 %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %bias_empty = tensor.empty() : tensor<64xf16>
  %bias_pre = linalg.generic {
    indexing_maps = [#vector, #vector], iterator_types = ["parallel"]
  } ins(%bias : tensor<64xf16>) outs(%bias_empty : tensor<64xf16>)
    attrs = {gc.fusion_group = #gc.fusion_group<1, tile_sizes = [64]>} {
  ^bb0(%x: f16, %out: f16):
    %exp = math.exp %x : f16
    linalg.yield %exp : f16
  } -> tensor<64xf16>
  %mm_empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%mm_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm = linalg.matmul {gc.fusion_group = #mm0}
    ins(%a, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_empty = tensor.empty() : tensor<64x64xf16>
  %post = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias_pre : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post : tensor<64x64xf16>
}

// -----

#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// Independent matmuls with a shared input and the same group may merge their
// workgroup and K loops. Both post-ops remain in that workgroup.
// CHECK-LABEL: func.func @two_matmuls_same_group(
// CHECK-SAME:  gc.num_kernels = 1 :
// CHECK:         %[[RESULT:.*]]:2 = scf.forall
// CHECK:           %[[INIT_A:.*]] = linalg.fill
// CHECK:           %[[INIT_B:.*]] = linalg.fill
// CHECK-NEXT:      %[[REDUCE:.*]]:2 = scf.for {{.*}}iter_args(%[[ACC_A:.*]] = %[[INIT_A]], %[[ACC_B:.*]] = %[[INIT_B]])
// CHECK-DAG:         %[[MM_A:.*]] = linalg.matmul {{.*}}outs(%[[ACC_A]]
// CHECK-DAG:         %[[MM_B:.*]] = linalg.matmul {{.*}}outs(%[[ACC_B]]
// CHECK-NEXT:        scf.yield %[[MM_A]], %[[MM_B]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK:           %[[POST_A:.*]] = linalg.generic {{.*}}ins(%[[REDUCE]]#0,
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           %[[POST_B:.*]] = linalg.generic {{.*}}ins(%[[REDUCE]]#1,
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-DAG:         tensor.parallel_insert_slice %[[POST_A]] into
// CHECK-DAG:         tensor.parallel_insert_slice %[[POST_B]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "two_matmuls_same_group_kernel", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1
func.func @two_matmuls_same_group(%a: tensor<64x256xf16>,
                                 %b: tensor<256x64xf16>,
                                 %c: tensor<256x64xf16>,
                                 %bias: tensor<64xf16>)
    -> (tensor<64x64xf16>, tensor<64x64xf16>) {
  %zero = arith.constant 0.0 : f16
  %empty_a = tensor.empty() : tensor<64x64xf16>
  %empty_b = tensor.empty() : tensor<64x64xf16>
  %post_empty_a = tensor.empty() : tensor<64x64xf16>
  %post_empty_b = tensor.empty() : tensor<64x64xf16>
  %init_a = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%empty_a : tensor<64x64xf16>) -> tensor<64x64xf16>
  %init_b = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%empty_b : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm_a = linalg.matmul {gc.fusion_group = #mm0}
    ins(%a, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init_a : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm_b = linalg.matmul {gc.fusion_group = #mm0}
    ins(%a, %c : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init_b : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_a = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm_a, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty_a : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  %post_b = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm_b, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty_b : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post_a, %post_b : tensor<64x64xf16>, tensor<64x64xf16>
}

// -----

#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#mm1 = #gc.fusion_group<1, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#post1 = #gc.fusion_group<1, tile_sizes = [32, 64]>
#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// Identical launch configurations and a shared input must not merge different
// groups. Each deferred workgroup must restore its own ID before fusing its
// post-op; otherwise one post-op would be left outside its workgroup.
// CHECK-LABEL: func.func @two_matmuls_different_groups(
// CHECK-SAME:  gc.num_kernels = 2 :
// CHECK:         %[[RESULT_A:.*]] = scf.forall
// CHECK:           %[[REDUCE_A:.*]] = scf.for
// CHECK:             %[[MM_A:.*]] = linalg.matmul {{.*}}gc.fusion_group = #gc.fusion_group<0,
// CHECK-NEXT:        scf.yield %[[MM_A]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 0 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK:           %[[POST_A:.*]] = linalg.generic {{.*}}ins(%[[REDUCE_A]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST_A]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : {{[^,}]+}}}
// CHECK-NEXT:    %[[RESULT_B:.*]] = scf.forall
// CHECK:           %[[REDUCE_B:.*]] = scf.for
// CHECK:             %[[MM_B:.*]] = linalg.matmul {{.*}}gc.fusion_group = #gc.fusion_group<1,
// CHECK-NEXT:        scf.yield %[[MM_B]]
// CHECK-NEXT:      } {gc.tiling.fusion_group_id = 1 : {{[^,}]+}}, gc.tiling.level = 1 : i8}
// CHECK:           %[[POST_B:.*]] = linalg.generic {{.*}}ins(%[[REDUCE_B]],
// CHECK-NEXT:      ^bb0(%[[X:[^:]+]]: {{[^,]+}}, %[[Y:[^:]+]]:
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[X]], %[[Y]]
// CHECK-NEXT:        linalg.yield %[[SUM]]
// CHECK-NEXT:      }
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[POST_B]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : {{[^,}]+}}}
// CHECK-NEXT:    return %[[RESULT_A]], %[[RESULT_B]]
func.func @two_matmuls_different_groups(%a: tensor<64x256xf16>,
                                       %b: tensor<256x64xf16>,
                                       %c: tensor<256x64xf16>,
                                       %bias: tensor<64xf16>)
    -> (tensor<64x64xf16>, tensor<64x64xf16>) {
  %zero = arith.constant 0.0 : f16
  %empty_a = tensor.empty() : tensor<64x64xf16>
  %empty_b = tensor.empty() : tensor<64x64xf16>
  %post_empty_a = tensor.empty() : tensor<64x64xf16>
  %post_empty_b = tensor.empty() : tensor<64x64xf16>
  %init_a = linalg.fill {gc.fusion_group = #post0}
    ins(%zero : f16) outs(%empty_a : tensor<64x64xf16>) -> tensor<64x64xf16>
  %init_b = linalg.fill {gc.fusion_group = #post1}
    ins(%zero : f16) outs(%empty_b : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm_a = linalg.matmul {gc.fusion_group = #mm0}
    ins(%a, %b : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init_a : tensor<64x64xf16>) -> tensor<64x64xf16>
  %mm_b = linalg.matmul {gc.fusion_group = #mm1}
    ins(%a, %c : tensor<64x256xf16>, tensor<256x64xf16>)
    outs(%init_b : tensor<64x64xf16>) -> tensor<64x64xf16>
  %post_a = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm_a, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty_a : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  %post_b = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm_b, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty_b : tensor<64x64xf16>)
    attrs = {gc.fusion_group = #post1} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post_a, %post_b : tensor<64x64xf16>, tensor<64x64xf16>
}
