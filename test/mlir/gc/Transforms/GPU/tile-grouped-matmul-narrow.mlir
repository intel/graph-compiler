// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-grouped-matmul --split-input-file | FileCheck %s
// RUN: gc-opt %s --gpu-dev-props --tile-contract --tile-grouped-matmul --apply-padding-level --split-input-file | FileCheck %s --check-prefix=PAD

// The i32 expert offsets must not hide the i4 parameter's 64-wide N load.
// CHECK: n_contiguous_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @n_contiguous
// CHECK: scf.forall {{.*}}step (1, 32, 64)
// CHECK: linalg.matmul
// CHECK-SAME: gc.tiling.wg_tile_sizes = array<i64: 32, 64, 64>
func.func @n_contiguous(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %z: tensor<4x128xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// i4 weights loaded along K do not require wider N sub-group tiles.
// CHECK: k_contiguous_kernel = {{.*}}sg_layout = [1, 4, 4]
// CHECK-SAME: threads = [256, 1, 1]
// CHECK-LABEL: func.func @k_contiguous
// CHECK: scf.forall {{.*}}step (1, 32, 64)
func.func @k_contiguous(%a: tensor<?x128xf16>, %w: tensor<4x128x128xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w : tensor<4x128x128xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%q: i4, %unused: f16):
    %v = arith.uitofp %q : i4 to f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// Ordinary f16 grouped matmul preserves its existing layout.
// CHECK: wide_kernel = {{.*}}sg_layout = [1, 4, 4]
// CHECK-SAME: threads = [256, 1, 1]
// CHECK-LABEL: func.func @wide
// CHECK: scf.forall {{.*}}step (1, 32, 64)
func.func @wide(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %w, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// Strided [expert, N, group] storage is contiguous along N, not group.
// CHECK: strided_groups_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @strided_groups
// CHECK: scf.forall {{.*}}step (1, 32, 64)
func.func @strided_groups(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %zbuf: memref<4x128x2xi4, strided<[256, 1, 128]>>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %z = bufferization.to_tensor %zbuf restrict writable : memref<4x128x2xi4, strided<[256, 1, 128]>> to tensor<4x128x2xi4>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n, k floordiv 64)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128x2xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// A column gather in row-major [expert, N, group] storage is a different path.
// CHECK: row_major_groups_kernel = {{.*}}sg_layout = [1, 4, 4]
// CHECK-SAME: threads = [256, 1, 1]
// CHECK-LABEL: func.func @row_major_groups
func.func @row_major_groups(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %z: tensor<4x128x2xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n, k floordiv 64)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128x2xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// Producer indexing maps must preserve the source axis through a transpose.
// CHECK: transposed_parameter_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @transposed_parameter
func.func @transposed_parameter(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %z: tensor<4x128xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %ze = tensor.empty() : tensor<128x4xi4>
  %zt = linalg.transpose ins(%z : tensor<4x128xi4>) outs(%ze : tensor<128x4xi4>) permutation = [1, 0]
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (n, e)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %zt : tensor<4x128x128xf16>, tensor<128x4xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// Rank-reducing slices and tensor casts must not hide the N-contiguous source.
// CHECK: sliced_parameter_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @sliced_parameter
func.func @sliced_parameter(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %source: tensor<1x4x128xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %slice = tensor.extract_slice %source[0, 0, 0] [1, 4, 128] [1, 1, 1] : tensor<1x4x128xi4> to tensor<4x128xi4>
  %z = tensor.cast %slice : tensor<4x128xi4> to tensor<4x128xi4>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// Grow a 32-wide WG tile to 64 as well: the SG must never exceed its WG.
// CHECK: n_tail_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @n_tail
// CHECK: scf.forall {{.*}}to (4, %{{.*}}, 96) step (1, 32, 64)
// CHECK: return {{.*}} : tensor<?x96xf16>
// PAD-LABEL: func.func @n_tail
// PAD: linalg.matmul
// PAD-SAME: tensor<32x64xf16>, tensor<64x64xf16>
// PAD-SAME: tensor<32x64xf16>
// PAD: return {{.*}} : tensor<?x96xf16>
func.func @n_tail(%a: tensor<?x128xf16>, %w: tensor<4x96x128xf16>, %z: tensor<4x96xi4>, %offs: tensor<4xi32>) -> tensor<?x96xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %we = tensor.empty() : tensor<4x96x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x96x128xf16>, tensor<4x96xi4>) outs(%we : tensor<4x96x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x96x128xf16>
  %ce = tensor.empty(%m) : tensor<?x96xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x96xf16>) -> tensor<?x96xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x96x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x96xf16>) -> tensor<?x96xf16>
  return %r : tensor<?x96xf16>
}

// -----

// Reassociation and trailing unit dimensions preserve contiguous N storage.
// CHECK: reshaped_parameter_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @reshaped_parameter
func.func @reshaped_parameter(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %source: tensor<1x4x128xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %collapsed = tensor.collapse_shape %source [[0, 1], [2]] : tensor<1x4x128xi4> into tensor<4x128xi4>
  %z = tensor.expand_shape %collapsed [[0], [1, 2]] output_shape [4, 128, 1] : tensor<4x128xi4> into tensor<4x128x1xi4>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n, 0)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128x1xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// A strided slice gathers rather than loading a contiguous N block.
// CHECK: strided_parameter_kernel = {{.*}}sg_layout = [1, 4, 4]
// CHECK-SAME: threads = [256, 1, 1]
// CHECK-LABEL: func.func @strided_parameter
func.func @strided_parameter(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %source: tensor<4x256xi4>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %z = tensor.extract_slice %source[0, 0] [4, 128] [1, 2] : tensor<4x256xi4> to tensor<4x128xi4>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128xi4>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}

// -----

// Padding a unit N dimension must keep its subgroup-layout position.
// CHECK: unit_n_kernel = {{.*}}sg_layout = [1, 4, 1]
// CHECK-SAME: threads = [64, 1, 1]
// CHECK-LABEL: func.func @unit_n
// CHECK: scf.forall {{.*}}step (1, 64)
// CHECK: return {{.*}} : tensor<?x1xf16>
func.func @unit_n(%a: tensor<?x128xf16>, %w: tensor<4x1x128xf16>, %z: tensor<4x1xi4>, %offs: tensor<4xi32>) -> tensor<?x1xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %we = tensor.empty() : tensor<4x1x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x1x128xf16>, tensor<4x1xi4>) outs(%we : tensor<4x1x128xf16>) {
  ^bb0(%f: f16, %zi: i4, %unused: f16):
    %zf = arith.uitofp %zi : i4 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x1x128xf16>
  %ce = tensor.empty(%m) : tensor<?x1xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x1xf16>) -> tensor<?x1xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x1x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x1xf16>) -> tensor<?x1xf16>
  return %r : tensor<?x1xf16>
}

// -----

// N-contiguous i8 inputs require 32-element subgroup block-load widths.
// CHECK: n_contiguous_i8_kernel = {{.*}}sg_layout = [1, 4, 2]
// CHECK-SAME: threads = [128, 1, 1]
// CHECK-LABEL: func.func @n_contiguous_i8
// CHECK: scf.forall {{.*}}step (1, 32, 64)
func.func @n_contiguous_i8(%a: tensor<?x128xf16>, %w: tensor<4x128x128xf16>, %z: tensor<4x128xi8>, %offs: tensor<4xi32>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  %m = tensor.dim %a, %c0 : tensor<?x128xf16>
  %we = tensor.empty() : tensor<4x128x128xf16>
  %dw = linalg.generic {
    indexing_maps = [affine_map<(e, n, k) -> (e, n, k)>, affine_map<(e, n, k) -> (e, n)>, affine_map<(e, n, k) -> (e, n, k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%w, %z : tensor<4x128x128xf16>, tensor<4x128xi8>) outs(%we : tensor<4x128x128xf16>) {
  ^bb0(%f: f16, %zi: i8, %unused: f16):
    %zf = arith.uitofp %zi : i8 to f16
    %v = arith.subf %f, %zf : f16
    linalg.yield %v : f16
  } -> tensor<4x128x128xf16>
  %ce = tensor.empty(%m) : tensor<?x128xf16>
  %init = linalg.fill ins(%zero : f16) outs(%ce : tensor<?x128xf16>) -> tensor<?x128xf16>
  %r = linalgx.grouped_matmul <{indexing_maps = [affine_map<(e, m, n, k) -> (m, k)>, affine_map<(e, m, n, k) -> (e, n, k)>, affine_map<(e, m, n, k) -> (e)>, affine_map<(e, m, n, k) -> (m, n)>]}>
    ins(%a, %dw, %offs : tensor<?x128xf16>, tensor<4x128x128xf16>, tensor<4xi32>)
    outs(%init : tensor<?x128xf16>) -> tensor<?x128xf16>
  return %r : tensor<?x128xf16>
}
