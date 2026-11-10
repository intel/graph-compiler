// RUN: gc-opt %s --gpu-dev-props --tile-contract --split-input-file | FileCheck %s

// The gate and up contractions share the tokens input and have the same
// workgroup tile space. They use one workgroup loop and one K loop, with the
// downstream swish and multiply inside the workgroup loop.

#lhs_map = affine_map<(m, n, k) -> (m, k)>
#rhs_map = affine_map<(m, n, k) -> (n, k)>
#acc_map = affine_map<(m, n, k) -> (m, n)>
#elt = affine_map<(m, n) -> (m, n)>

// CHECK-LABEL: func.func @entry(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK-DAG:     bufferization.to_tensor %arg1 restrict writable : memref<512x2048xf16>
// CHECK-DAG:     bufferization.to_tensor %arg2 restrict writable : memref<512x2048xf16>
// CHECK:         %[[RESULT:.+]] = scf.forall
// CHECK:           linalg.fill {__gate_fill__, gc.tiling.wg_tile_sizes = array<i64: 32, 64>}
// CHECK:           linalg.fill {__up_fill__, gc.tiling.wg_tile_sizes = array<i64: 32, 64>}
// CHECK:           %[[K:[^: ]+]]:2 = scf.for
// CHECK:             linalg.matmul {{.*}}{__gate__, gc.tiling.level = 1 : i8, gc.tiling.wg_tile_sizes = array<i64: 32, 64, 64>}
// CHECK:             linalg.matmul {{.*}}{__up__, gc.tiling.level = 1 : i8, gc.tiling.wg_tile_sizes = array<i64: 32, 64, 64>}
// CHECK:             scf.yield {{.*}}, {{.*}} : tensor<?x64xf16>, tensor<?x64xf16>
// CHECK-NOT:       = scf.for
// CHECK:           %[[SWISH:.+]] = linalg.generic {{.*}} ins(%[[K]]#0 {{.*}} attrs = {{ *}}{__swish__
// CHECK:           %[[MUL:.+]] = linalg.generic {{.*}} ins(%[[SWISH]], %[[K]]#1 {{.*}} attrs = {{ *}}{__mul__,
// CHECK:           scf.forall.in_parallel {
// CHECK-NEXT:        tensor.parallel_insert_slice %[[MUL]] into
// CHECK-NEXT:      }
// CHECK-NEXT:    } {gc.kernel_name = "entry_kernel"}
// CHECK-NOT:     = scf.forall (
// CHECK:         bufferization.materialize_in_destination %[[RESULT]]
func.func @entry(%tokens_buf: memref<?x2048xf16>,
                 %gate_w_buf: memref<512x2048xf16>,
                 %up_w_buf: memref<512x2048xf16>,
                 %out_buf: memref<?x512xf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %one = arith.constant 1.000000e+00 : f16

  %tokens = bufferization.to_tensor %tokens_buf restrict writable : memref<?x2048xf16> to tensor<?x2048xf16>
  %total = tensor.dim %tokens, %c0 : tensor<?x2048xf16>
  %gate_w = bufferization.to_tensor %gate_w_buf restrict writable : memref<512x2048xf16> to tensor<512x2048xf16>
  %up_w = bufferization.to_tensor %up_w_buf restrict writable : memref<512x2048xf16> to tensor<512x2048xf16>

  // Keep the destinations before both contractions so the consumer operands
  // dominate the loops created during tiling.
  %gate_dst = tensor.empty(%total) : tensor<?x512xf16>
  %up_dst = tensor.empty(%total) : tensor<?x512xf16>
  %swish_dst = tensor.empty(%total) : tensor<?x512xf16>
  %mul_dst = tensor.empty(%total) : tensor<?x512xf16>

  %gate_init = linalg.fill {__gate_fill__} ins(%cst : f16) outs(%gate_dst : tensor<?x512xf16>) -> tensor<?x512xf16>
  %gate = linalg.matmul indexing_maps = [#lhs_map, #rhs_map, #acc_map] {__gate__}
    ins(%tokens, %gate_w : tensor<?x2048xf16>, tensor<512x2048xf16>)
    outs(%gate_init : tensor<?x512xf16>) -> tensor<?x512xf16>

  %up_init = linalg.fill {__up_fill__} ins(%cst : f16) outs(%up_dst : tensor<?x512xf16>) -> tensor<?x512xf16>
  %up = linalg.matmul indexing_maps = [#lhs_map, #rhs_map, #acc_map] {__up__}
    ins(%tokens, %up_w : tensor<?x2048xf16>, tensor<512x2048xf16>)
    outs(%up_init : tensor<?x512xf16>) -> tensor<?x512xf16>

  %swish = linalg.generic {
    indexing_maps = [#elt, #elt], iterator_types = ["parallel", "parallel"]
  } ins(%gate : tensor<?x512xf16>) outs(%swish_dst : tensor<?x512xf16>) attrs = {__swish__} {
  ^bb0(%in: f16, %out: f16):
    %n0 = arith.negf %in : f16
    %e0 = math.exp %n0 : f16
    %d0 = arith.addf %e0, %one : f16
    %s0 = arith.divf %in, %d0 : f16
    linalg.yield %s0 : f16
  } -> tensor<?x512xf16>

  %res = linalg.generic {
    indexing_maps = [#elt, #elt, #elt], iterator_types = ["parallel", "parallel"]
  } ins(%swish, %up : tensor<?x512xf16>, tensor<?x512xf16>) outs(%mul_dst : tensor<?x512xf16>) attrs = {__mul__} {
  ^bb0(%in: f16, %in_up: f16, %out: f16):
    %p = arith.mulf %in, %in_up : f16
    linalg.yield %p : f16
  } -> tensor<?x512xf16>

  bufferization.materialize_in_destination %res in restrict writable %out_buf
    : (tensor<?x512xf16>, memref<?x512xf16>) -> ()
  return
}

// -----

// All three contractions read %x and have the same tile space. Sharing the
// zero-filled init tensor does not make their SSA results dependent.

// CHECK-LABEL: func.func @three(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK: = scf.forall (
// CHECK: = scf.for
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: scf.yield
// CHECK-NOT: = scf.for
// CHECK: } {gc.kernel_name = "three_kernel"}
// CHECK-NOT: = scf.forall (
// CHECK: return
func.func @three(%x: tensor<64x128xf16>, %w0: tensor<128x32xf16>,
                 %w1: tensor<128x32xf16>, %w2: tensor<128x32xf16>)
    -> (tensor<64x32xf16>, tensor<64x32xf16>, tensor<64x32xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<64x32xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x32xf16>) -> tensor<64x32xf16>
  %a = linalg.matmul ins(%x, %w0 : tensor<64x128xf16>, tensor<128x32xf16>)
    outs(%init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %b = linalg.matmul ins(%x, %w1 : tensor<64x128xf16>, tensor<128x32xf16>)
    outs(%init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %c = linalg.matmul ins(%x, %w2 : tensor<64x128xf16>, tensor<128x32xf16>)
    outs(%init : tensor<64x32xf16>) -> tensor<64x32xf16>
  return %a, %b, %c : tensor<64x32xf16>, tensor<64x32xf16>, tensor<64x32xf16>
}

// -----

// Equal tile spaces alone are insufficient: these matmuls share no input.
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @no_shared_input(
// CHECK-SAME: attributes {gc.num_kernels = 2 : i32}
// CHECK: = scf.forall (
// CHECK: = scf.forall (
// CHECK-NOT: = scf.forall (
// CHECK: return
func.func @no_shared_input(%a: tensor<64x128xf16>, %b: tensor<64x128xf16>,
                           %wa: tensor<32x128xf16>, %wb: tensor<32x128xf16>)
    -> (tensor<64x32xf16>, tensor<64x32xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %a_dst = tensor.empty() : tensor<64x32xf16>
  %b_dst = tensor.empty() : tensor<64x32xf16>
  %a_init = linalg.fill ins(%zero : f16) outs(%a_dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %b_init = linalg.fill ins(%zero : f16) outs(%b_dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %ra = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%a, %wa : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%a_init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %rb = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%b, %wb : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%b_init : tensor<64x32xf16>) -> tensor<64x32xf16>
  return %ra, %rb : tensor<64x32xf16>, tensor<64x32xf16>
}

// -----

// The first matmul has a pure user before the second matmul. Defer that user
// until after the sibling loops merge so both matmuls use one workgroup loop.
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
#elt = affine_map<(m, n) -> (m, n)>

// CHECK-LABEL: func.func @use_between_matmuls(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK: = scf.forall (
// CHECK-NOT: = scf.forall (
// CHECK: return
func.func @use_between_matmuls(%a: tensor<64x128xf16>,
                               %wa: tensor<32x128xf16>,
                               %wb: tensor<32x128xf16>)
    -> (tensor<64x32xf16>, tensor<64x32xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %one = arith.constant 1.000000e+00 : f16
  %gate_dst = tensor.empty() : tensor<64x32xf16>
  %up_dst = tensor.empty() : tensor<64x32xf16>
  %mid_dst = tensor.empty() : tensor<64x32xf16>
  %gate_init = linalg.fill ins(%zero : f16) outs(%gate_dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %up_init = linalg.fill ins(%zero : f16) outs(%up_dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %gate = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%a, %wa : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%gate_init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %mid = linalg.generic {
    indexing_maps = [#elt, #elt], iterator_types = ["parallel", "parallel"]
  } ins(%gate : tensor<64x32xf16>) outs(%mid_dst : tensor<64x32xf16>) {
  ^bb0(%in: f16, %outval: f16):
    %sum = arith.addf %in, %one : f16
    linalg.yield %sum : f16
  } -> tensor<64x32xf16>
  %up = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%a, %wb : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%up_init : tensor<64x32xf16>) -> tensor<64x32xf16>
  return %mid, %up : tensor<64x32xf16>, tensor<64x32xf16>
}

// -----

// One pair can fuse while an unrelated third matmul remains a separate
// kernel. Kernel names and metadata must stay distinct after renumbering.
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>

// CHECK: kernels = {one_pair_one_other_kernel =
// CHECK-SAME: one_pair_one_other_kernel_1 =
// CHECK-LABEL: func.func @one_pair_one_other(
// CHECK-SAME: attributes {gc.num_kernels = 2 : i32}
// CHECK: = scf.forall (
// CHECK: = scf.forall (
// CHECK-NOT: = scf.forall (
// CHECK: return
func.func @one_pair_one_other(%x: tensor<64x128xf16>,
                              %y: tensor<64x128xf16>,
                              %w0: tensor<32x128xf16>,
                              %w1: tensor<32x128xf16>,
                              %w2: tensor<32x128xf16>)
    -> (tensor<64x32xf16>, tensor<64x32xf16>, tensor<64x32xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %dst = tensor.empty() : tensor<64x32xf16>
  %init = linalg.fill ins(%zero : f16) outs(%dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %a = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%x, %w0 : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %b = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%x, %w1 : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %c = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%y, %w2 : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%init : tensor<64x32xf16>) -> tensor<64x32xf16>
  return %a, %b, %c : tensor<64x32xf16>, tensor<64x32xf16>, tensor<64x32xf16>
}

// -----

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
#elt = affine_map<(m, n) -> (m, n)>

// CHECK-LABEL: func.func @interleaved_cast(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK: = scf.forall (
// CHECK: = scf.for
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: arith.truncf
// CHECK-NOT: = scf.for
// CHECK-NOT: = scf.forall (
// CHECK: return
func.func @interleaved_cast(%x: tensor<64x128xf16>,
                            %wa: tensor<32x128xf16>,
                            %wb: tensor<32x128xf16>)
    -> (tensor<64x32xf16>, tensor<64x32xf16>) {
  %zero = arith.constant 0.000000e+00 : f32
  %dst = tensor.empty() : tensor<64x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %gate_dst = tensor.empty() : tensor<64x32xf16>
  %up_dst = tensor.empty() : tensor<64x32xf16>
  %gate = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%x, %wa : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
  %gate_cast = linalg.generic {
    indexing_maps = [#elt, #elt], iterator_types = ["parallel", "parallel"]
  } ins(%gate : tensor<64x32xf32>) outs(%gate_dst : tensor<64x32xf16>) {
  ^bb0(%in: f32, %outval: f16):
    %converted = arith.truncf %in : f32 to f16
    linalg.yield %converted : f16
  } -> tensor<64x32xf16>
  %up = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%x, %wb : tensor<64x128xf16>, tensor<32x128xf16>)
    outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
  %up_cast = linalg.generic {
    indexing_maps = [#elt, #elt], iterator_types = ["parallel", "parallel"]
  } ins(%up : tensor<64x32xf32>) outs(%up_dst : tensor<64x32xf16>) {
  ^bb0(%in: f32, %outval: f16):
    %converted = arith.truncf %in : f32 to f16
    linalg.yield %converted : f16
  } -> tensor<64x32xf16>
  return %gate_cast, %up_cast : tensor<64x32xf16>, tensor<64x32xf16>
}

// -----

// Tiling the live seed can erase an unused earlier matmul during function-wide
// canonicalization. Do not save the unused operation as a sibling member.

// CHECK-LABEL: func.func @unused_sibling(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK-NOT: __unused__
// CHECK: = scf.forall (
// CHECK: linalg.matmul {{.*}}{__live__
// CHECK-NOT: __unused__
// CHECK: return
func.func @unused_sibling(%x: tensor<64x128xf16>,
                          %wa: tensor<128x32xf16>,
                          %wb: tensor<128x32xf16>) -> tensor<64x32xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %dead_dst = tensor.empty() : tensor<64x32xf16>
  %live_dst = tensor.empty() : tensor<64x32xf16>
  %dead_init = linalg.fill ins(%zero : f16) outs(%dead_dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %live_init = linalg.fill ins(%zero : f16) outs(%live_dst : tensor<64x32xf16>) -> tensor<64x32xf16>
  %dead = linalg.matmul {__unused__} ins(%x, %wa : tensor<64x128xf16>, tensor<128x32xf16>)
    outs(%dead_init : tensor<64x32xf16>) -> tensor<64x32xf16>
  %live = linalg.matmul {__live__} ins(%x, %wb : tensor<64x128xf16>, tensor<128x32xf16>)
    outs(%live_init : tensor<64x32xf16>) -> tensor<64x32xf16>
  return %live : tensor<64x32xf16>
}

// -----

// A matmul used only by a dead consumer is not initially use_empty. Its
// workgroup bound also differs from the live matmul's, so it cannot be merged
// into the live loop after tiling.
#elt = affine_map<(m, n) -> (m, n)>

// CHECK-LABEL: func.func @indirect_unused_incompatible(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK-NOT: __indirect_dead__
// CHECK: = scf.forall (
// CHECK: linalg.matmul {{.*}}{__live__
// CHECK-NOT: __indirect_dead__
// CHECK: return
func.func @indirect_unused_incompatible(
    %x0: tensor<?x128xf16>, %x1: tensor<?x128xf16>,
    %w: tensor<128x32xf16>) -> tensor<?x32xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f16
  %m0 = tensor.dim %x0, %c0 : tensor<?x128xf16>
  %m1 = tensor.dim %x1, %c0 : tensor<?x128xf16>
  %dead_dst = tensor.empty(%m0) : tensor<?x32xf16>
  %live_dst = tensor.empty(%m1) : tensor<?x32xf16>
  %intermediate_dst = tensor.empty(%m0) : tensor<?x32xf16>
  %dead_init = linalg.fill ins(%zero : f16) outs(%dead_dst : tensor<?x32xf16>) -> tensor<?x32xf16>
  %live_init = linalg.fill ins(%zero : f16) outs(%live_dst : tensor<?x32xf16>) -> tensor<?x32xf16>
  %dead = linalg.matmul {__indirect_dead__}
    ins(%x0, %w : tensor<?x128xf16>, tensor<128x32xf16>)
    outs(%dead_init : tensor<?x32xf16>) -> tensor<?x32xf16>
  %unused = linalg.generic {
    indexing_maps = [#elt, #elt], iterator_types = ["parallel", "parallel"]
  } ins(%dead : tensor<?x32xf16>) outs(%intermediate_dst : tensor<?x32xf16>) {
  ^bb0(%v: f16, %out: f16):
    linalg.yield %v : f16
  } -> tensor<?x32xf16>
  %live = linalg.matmul {__live__}
    ins(%x1, %w : tensor<?x128xf16>, tensor<128x32xf16>)
    outs(%live_init : tensor<?x32xf16>) -> tensor<?x32xf16>
  return %live : tensor<?x32xf16>
}

// -----

#weights = affine_map<(n, k) -> (n, k)>
#group = affine_map<(n, k) -> (n, k floordiv 64)>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @sibling_dequant(
// CHECK-SAME: attributes {gc.num_kernels = 1 : i32}
// CHECK-NOT: linalg.generic
// CHECK: %[[WG:[^: ]+]]:2 = scf.forall
// CHECK-NOT: linalg.generic
// CHECK: %[[K:[^: ]+]]:2 = scf.for
// CHECK-DAG: {{^        }}%[[GATE_W:.+]] = linalg.generic {{.*}} ins({{.*}} : tensor<64x64xi4>, tensor<64x1xf16>)
// CHECK-DAG: {{^        }}%[[GATE:.+]] = linalg.matmul {{.*}} ins({{.*}}, %[[GATE_W]] : tensor<32x64xf16>, tensor<64x64xf16>)
// CHECK-DAG: {{^        }}%[[UP_W:.+]] = linalg.generic {{.*}} ins({{.*}} : tensor<64x64xi4>, tensor<64x1xf16>)
// CHECK-DAG: {{^        }}%[[UP:.+]] = linalg.matmul {{.*}} ins({{.*}}, %[[UP_W]] : tensor<32x64xf16>, tensor<64x64xf16>)
// CHECK: {{^        }}scf.yield %[[GATE]], %[[UP]] : tensor<32x64xf16>, tensor<32x64xf16>
// CHECK-NEXT: {{^      }}}
// CHECK-NOT: linalg.generic
// CHECK-NOT: = scf.for
// CHECK: scf.forall.in_parallel {
// CHECK-NEXT: tensor.parallel_insert_slice %[[K]]#0 into
// CHECK-NEXT: tensor.parallel_insert_slice %[[K]]#1 into
// CHECK-NEXT: }
// CHECK-NEXT: } {gc.kernel_name = "sibling_dequant_kernel"}
// CHECK-NOT: = scf.forall
// CHECK: return %[[WG]]#0, %[[WG]]#1
func.func @sibling_dequant(%tokens: tensor<64x2048xf16>,
                           %gate_weights: tensor<512x2048xi4>,
                           %gate_scales: tensor<512x32xf16>,
                           %up_weights: tensor<512x2048xi4>,
                           %up_scales: tensor<512x32xf16>)
    -> (tensor<64x512xf16>, tensor<64x512xf16>) {
  %zero = arith.constant 0.0 : f16
  %gate_w_dst = tensor.empty() : tensor<512x2048xf16>
  %up_w_dst = tensor.empty() : tensor<512x2048xf16>
  %gate_dst = tensor.empty() : tensor<64x512xf16>
  %up_dst = tensor.empty() : tensor<64x512xf16>
  %gate_init = linalg.fill ins(%zero : f16) outs(%gate_dst : tensor<64x512xf16>) -> tensor<64x512xf16>
  %up_init = linalg.fill ins(%zero : f16) outs(%up_dst : tensor<64x512xf16>) -> tensor<64x512xf16>
  %gate_w = linalg.generic {
    indexing_maps = [#weights, #group, #weights],
    iterator_types = ["parallel", "parallel"]}
    ins(%gate_weights, %gate_scales : tensor<512x2048xi4>, tensor<512x32xf16>)
    outs(%gate_w_dst : tensor<512x2048xf16>) {
  ^bb0(%quantized: i4, %scale: f16, %unused: f16):
    %extended = arith.extui %quantized : i4 to i32
    %converted = arith.sitofp %extended : i32 to f16
    %scaled = arith.mulf %converted, %scale : f16
    linalg.yield %scaled : f16
  } -> tensor<512x2048xf16>
  %gate = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%tokens, %gate_w : tensor<64x2048xf16>, tensor<512x2048xf16>)
    outs(%gate_init : tensor<64x512xf16>) -> tensor<64x512xf16>
  %up_w = linalg.generic {
    indexing_maps = [#weights, #group, #weights],
    iterator_types = ["parallel", "parallel"]}
    ins(%up_weights, %up_scales : tensor<512x2048xi4>, tensor<512x32xf16>)
    outs(%up_w_dst : tensor<512x2048xf16>) {
  ^bb0(%quantized: i4, %scale: f16, %unused: f16):
    %extended = arith.extui %quantized : i4 to i32
    %converted = arith.sitofp %extended : i32 to f16
    %scaled = arith.mulf %converted, %scale : f16
    linalg.yield %scaled : f16
  } -> tensor<512x2048xf16>
  %up = linalg.matmul indexing_maps = [#lhs, #rhs, #out]
    ins(%tokens, %up_w : tensor<64x2048xf16>, tensor<512x2048xf16>)
    outs(%up_init : tensor<64x512xf16>) -> tensor<64x512xf16>
  return %gate, %up : tensor<64x512xf16>, tensor<64x512xf16>
}
