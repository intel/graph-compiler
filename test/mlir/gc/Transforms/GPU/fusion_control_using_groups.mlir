// RUN: gc-opt %s --gpu-dev-props --tile-contract --split-input-file | FileCheck %s --check-prefix=CONTRACT
// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-parallel --split-input-file | FileCheck %s --check-prefix=PARALLEL

// Group membership compares IDs, not tile sizes. These describe the iteration
// domains of exp(A), matmul, and the broadcast bias addition respectively.
#pre0 = #gc.fusion_group<0, tile_sizes = [32, 64]>
#mm0 = #gc.fusion_group<0, tile_sizes = [32, 64, 64]>
#post0 = #gc.fusion_group<0, tile_sizes = [32, 64]>

#identity = affine_map<(m, n) -> (m, n)>
#bias = affine_map<(m, n) -> (n)>

// exp(A) @ B + bias: the pre-op, fill, matmul, and post-op all fuse into one
// workgroup. The exp is computed inside the reduction loop.
// CONTRACT-LABEL: func.func @all_same_group(
// CONTRACT-SAME: gc.num_kernels = 1 : i32
// CONTRACT-NOT: __pre__
// CONTRACT: scf.forall
// CONTRACT: linalg.fill
// CONTRACT: scf.for
// CONTRACT: linalg.generic {{.*}}__pre__
// CONTRACT: linalg.matmul
// CONTRACT: linalg.generic {{.*}}__post__
// CONTRACT: } {gc.kernel_name = "all_same_group_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT-NOT: linalg.generic
// CONTRACT: return
// PARALLEL-LABEL: func.func @all_same_group(
// PARALLEL-SAME: gc.num_kernels = 1 : i32
// PARALLEL: scf.forall
// PARALLEL-NOT: = scf.forall
// PARALLEL: return
func.func @all_same_group(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                          %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {__pre__, gc.fusion_group = #pre0} {
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
    attrs = {__post__, gc.fusion_group = #post0} {
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
// CONTRACT-LABEL: func.func @pre_other_group(
// CONTRACT-SAME: gc.num_kernels = 1 : i32
// CONTRACT: linalg.generic {{.*}}__pre__
// CONTRACT: scf.forall
// CONTRACT-NOT: __pre__
// CONTRACT: linalg.matmul
// CONTRACT-NOT: __pre__
// CONTRACT: linalg.generic {{.*}}__post__
// CONTRACT: } {gc.kernel_name = "pre_other_group_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT: return
// PARALLEL-LABEL: func.func @pre_other_group(
// PARALLEL-SAME: gc.num_kernels = 2 : i32
// PARALLEL: scf.forall
// PARALLEL: linalg.generic {{.*}}__pre__
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : i64}
// PARALLEL: scf.forall
// PARALLEL-NOT: __pre__
// PARALLEL: linalg.matmul
// PARALLEL: linalg.generic {{.*}}__post__
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : i64}
// PARALLEL: return
func.func @pre_other_group(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                           %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {__pre__, gc.fusion_group = #pre1} {
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
    attrs = {__post__, gc.fusion_group = #post0} {
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
// CONTRACT-LABEL: func.func @post_other_group(
// CONTRACT-SAME: gc.num_kernels = 1 : i32
// CONTRACT: scf.forall
// CONTRACT: linalg.generic {{.*}}__pre__
// CONTRACT: linalg.matmul
// CONTRACT-NOT: __post__
// CONTRACT: } {gc.kernel_name = "post_other_group_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT: linalg.generic {{.*}}__post__
// CONTRACT: return
// PARALLEL-LABEL: func.func @post_other_group(
// PARALLEL-SAME: gc.num_kernels = 2 : i32
// PARALLEL: scf.forall
// PARALLEL: linalg.matmul
// PARALLEL-NOT: __post__
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : i64}
// PARALLEL: scf.forall
// PARALLEL: linalg.generic {{.*}}__post__
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : i64}
// PARALLEL: return
func.func @post_other_group(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                            %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {__pre__, gc.fusion_group = #pre0} {
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
    attrs = {__post__, gc.fusion_group = #post1} {
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
// CONTRACT-LABEL: func.func @all_different_groups(
// CONTRACT: linalg.generic {{.*}}__pre__
// CONTRACT: scf.forall
// CONTRACT-NOT: __pre__
// CONTRACT: linalg.matmul
// CONTRACT-NOT: __post__
// CONTRACT: } {gc.kernel_name = "all_different_groups_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT: linalg.generic {{.*}}__post__
// CONTRACT: return
// PARALLEL-LABEL: func.func @all_different_groups(
// PARALLEL-SAME: gc.num_kernels = 3 : i32
// PARALLEL: scf.forall
// PARALLEL: linalg.generic {{.*}}__pre__
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : i64}
// PARALLEL: scf.forall
// PARALLEL: linalg.matmul
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : i64}
// PARALLEL: scf.forall
// PARALLEL: linalg.generic {{.*}}__post__
// PARALLEL: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 2 : i64}
// PARALLEL: return
func.func @all_different_groups(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                                %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {__pre__, gc.fusion_group = #pre1} {
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
    attrs = {__post__, gc.fusion_group = #post2} {
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
// CONTRACT-LABEL: func.func @mixed_annotations(
// CONTRACT: linalg.generic {{.*}}__pre__
// CONTRACT: scf.forall
// CONTRACT-NOT: __pre__
// CONTRACT: linalg.matmul
// CONTRACT-NOT: __post__
// CONTRACT: } {gc.kernel_name = "mixed_annotations_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT: linalg.generic {{.*}}__post__
// CONTRACT: return
// PARALLEL-LABEL: func.func @mixed_annotations(
// PARALLEL-SAME: gc.num_kernels = 3 : i32
func.func @mixed_annotations(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                             %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %pre_empty = tensor.empty() : tensor<64x256xf16>
  %pre = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%a : tensor<64x256xf16>) outs(%pre_empty : tensor<64x256xf16>)
    attrs = {__pre__} {
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
    outs(%post_empty : tensor<64x64xf16>) attrs = {__post__} {
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
// CONTRACT-LABEL: func.func @post_input_other_group(
// CONTRACT: linalg.generic {{.*}}__bias_pre__
// CONTRACT: scf.forall
// CONTRACT-NOT: __bias_pre__
// CONTRACT: linalg.matmul
// CONTRACT-NOT: __bias_pre__
// CONTRACT: linalg.generic {{.*}}__post__
// CONTRACT: } {gc.kernel_name = "post_input_other_group_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT: return
// PARALLEL-LABEL: func.func @post_input_other_group(
// PARALLEL-SAME: gc.num_kernels = 2 : i32
func.func @post_input_other_group(%a: tensor<64x256xf16>,
                                 %b: tensor<256x64xf16>,
                                 %bias: tensor<64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %bias_empty = tensor.empty() : tensor<64xf16>
  %bias_pre = linalg.generic {
    indexing_maps = [#vector, #vector], iterator_types = ["parallel"]
  } ins(%bias : tensor<64xf16>) outs(%bias_empty : tensor<64xf16>)
    attrs = {__bias_pre__, gc.fusion_group = #gc.fusion_group<1, tile_sizes = [64]>} {
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
    attrs = {__post__, gc.fusion_group = #post0} {
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
// CONTRACT-LABEL: func.func @two_matmuls_same_group(
// CONTRACT-SAME: gc.num_kernels = 1 : i32
// CONTRACT: scf.forall
// CONTRACT: scf.for
// CONTRACT: linalg.matmul
// CONTRACT: linalg.matmul
// CONTRACT: scf.yield
// CONTRACT-NOT: scf.for
// CONTRACT: linalg.generic {{.*}}__post_a__
// CONTRACT: linalg.generic {{.*}}__post_b__
// CONTRACT: } {gc.kernel_name = "two_matmuls_same_group_kernel", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT-NOT: scf.forall
// CONTRACT: return
// PARALLEL-LABEL: func.func @two_matmuls_same_group(
// PARALLEL-SAME: gc.num_kernels = 1 : i32
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
    attrs = {__post_a__, gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  %post_b = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm_b, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty_b : tensor<64x64xf16>)
    attrs = {__post_b__, gc.fusion_group = #post0} {
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
// CONTRACT-LABEL: func.func @two_matmuls_different_groups(
// CONTRACT-SAME: gc.num_kernels = 2 : i32
// CONTRACT: scf.forall
// CONTRACT: linalg.matmul {{.*}}gc.fusion_group = #gc.fusion_group<0,
// CONTRACT: linalg.generic {{.*}}__post_a__
// CONTRACT: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 0 : i64}
// CONTRACT: scf.forall
// CONTRACT: linalg.matmul {{.*}}gc.fusion_group = #gc.fusion_group<1,
// CONTRACT: linalg.generic {{.*}}__post_b__
// CONTRACT: } {gc.kernel_name = "{{[^"]+}}", gc.tiling.fusion_group_id = 1 : i64}
// CONTRACT-NOT: linalg.generic
// CONTRACT: return
// PARALLEL-LABEL: func.func @two_matmuls_different_groups(
// PARALLEL-SAME: gc.num_kernels = 2 : i32
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
    attrs = {__post_a__, gc.fusion_group = #post0} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  %post_b = linalg.generic {
    indexing_maps = [#identity, #bias, #identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm_b, %bias : tensor<64x64xf16>, tensor<64xf16>)
    outs(%post_empty_b : tensor<64x64xf16>)
    attrs = {__post_b__, gc.fusion_group = #post1} {
  ^bb0(%x: f16, %bias_value: f16, %out: f16):
    %sum = arith.addf %x, %bias_value : f16
    linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %post_a, %post_b : tensor<64x64xf16>, tensor<64x64xf16>
}
