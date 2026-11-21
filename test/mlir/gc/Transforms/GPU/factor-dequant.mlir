// RUN: gc-opt %s --factor-dequant --split-input-file | FileCheck %s
// RUN: gc-opt %s --factor-dequant=fold-zero-point=false --split-input-file | FileCheck %s --check-prefix=NOZP

// The form that --vectorize makes of a grouped dequant matmul: the weights
// are dequantized as a 1x64x64 tile that goes through a tensor before the
// contraction. The scale and the zero point are broadcast along k.
// CHECK-LABEL: func.func @grouped_tile
// CHECK: %[[INIT:.+]] = arith.extf %{{.*}} : vector<32x64xf16> to vector<32x64xf32>
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[INIT]]) -> (vector<32x64xf32>)
// CHECK-NOT: arith.mulf
// CHECK-DAG: %[[X:.+]] = vector.transfer_read {{.*}} : tensor<32x2048xf16>, vector<32x64xf16>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK-DAG: %[[BIAS:.+]] = arith.constant dense<26112> : vector<1x64x64xi16>
// CHECK-DAG: %[[OFFSET:.+]] = arith.constant dense<1.536000e+03> : vector<1x64xf16>
// CHECK-DAG: %[[ZP:.+]] = vector.transfer_read {{.*}} : tensor<256x512x32xi4>, vector<1x64x1xi4>
// CHECK-DAG: %[[W:.+]] = vector.transfer_read {{.*}} : tensor<256x512x2048xi4>, vector<1x64x64xi4>
// CHECK-DAG: %[[W16:.+]] = arith.extui %[[W]] : vector<1x64x64xi4> to vector<1x64x64xi16>
// CHECK-DAG: %[[BITS:.+]] = arith.ori %[[W16]], %[[BIAS]] : vector<1x64x64xi16>
// CHECK-DAG: %[[MAGIC:.+]] = arith.bitcast %[[BITS]] : vector<1x64x64xi16> to vector<1x64x64xf16>
// CHECK-DAG: %[[B:.+]] = vector.shape_cast %[[MAGIC]] : vector<1x64x64xf16> to vector<64x64xf16>
// CHECK-DAG: %[[ZF:.+]] = arith.uitofp %{{.*}} : vector<1x64xi32> to vector<1x64xf16>
// CHECK-DAG: %[[ZO:.+]] = arith.addf %[[ZF]], %[[OFFSET]] : vector<1x64xf16>
// CHECK-DAG: %[[ZN:.+]] = arith.negf %[[ZO]] : vector<1x64xf16>
// CHECK-DAG: %[[ZK:.+]] = vector.broadcast %[[ZN]] : vector<1x64xf16> to vector<64x64xf16>
// CHECK-DAG: %[[ZB:.+]] = vector.transpose %[[ZK]], [1, 0] : vector<64x64xf16> to vector<64x64xf16>
// CHECK: %[[START:.+]] = vector.contract {{.*}} %[[X]], %[[ZB]], %[[ZERO]] : vector<32x64xf16>, vector<64x64xf16> into vector<32x64xf32>
// CHECK: %[[P:.+]] = vector.contract {{.*}} %[[X]], %[[B]], %[[START]] : vector<32x64xf16>, vector<64x64xf16> into vector<32x64xf32>
// CHECK: %[[S:.+]] = arith.extf %{{.*}} : vector<1x64xf16> to vector<1x64xf32>
// CHECK: %[[SB:.+]] = vector.broadcast %[[S]] : vector<1x64xf32> to vector<32x64xf32>
// CHECK: %[[OUT:.+]] = math.fma %[[SB]], %[[P]], %[[ACC]] : vector<32x64xf32>
// CHECK: scf.yield %[[OUT]] : vector<32x64xf32>
// CHECK: arith.truncf %{{.*}} : vector<32x64xf32> to vector<32x64xf16>
// NOZP-LABEL: func.func @grouped_tile
// NOZP: scf.for {{.*}} iter_args(%[[ACC:.+]] = %{{.*}}) -> (vector<32x64xf32>)
// NOZP-NOT: arith.mulf
// NOZP-NOT: arith.ori
// NOZP: arith.subi %{{.*}} : vector<1x64x64xi32>
// NOZP: %[[P:.+]] = vector.contract {{.*}} : vector<32x64xf16>, vector<64x64xf16> into vector<32x64xf32>
// NOZP: %[[OUT:.+]] = math.fma %{{.*}}, %[[P]], %[[ACC]] : vector<32x64xf32>
// NOZP: scf.yield %[[OUT]]
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @grouped_tile(%x: tensor<32x2048xf16>, %w: tensor<256x512x2048xi4>,
                        %s: tensor<256x512x32xf16>, %z: tensor<256x512x32xi4>,
                        %e: index, %n: index) -> vector<32x64xf16> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %pf16 = ub.poison : f16
  %init = arith.constant dense<0.0> : vector<32x64xf16>
  %tile = tensor.empty() : tensor<1x64x64xf16>
  %r = scf.for %k = %c0 to %c2048 step %c64 iter_args(%acc = %init) -> (vector<32x64xf16>) {
    %g = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%k)
    %sv = vector.transfer_read %s[%e, %n, %g], %pf16 {in_bounds = [true, true, true]} : tensor<256x512x32xf16>, vector<1x64x1xf16>
    %sb = vector.broadcast %sv : vector<1x64x1xf16> to vector<64x1x64x1xf16>
    %st = vector.transpose %sb, [1, 2, 3, 0] : vector<64x1x64x1xf16> to vector<1x64x1x64xf16>
    %zv = vector.transfer_read %z[%e, %n, %g], %pi4 {in_bounds = [true, true, true]} : tensor<256x512x32xi4>, vector<1x64x1xi4>
    %zb = vector.broadcast %zv : vector<1x64x1xi4> to vector<64x1x64x1xi4>
    %zt = vector.transpose %zb, [1, 2, 3, 0] : vector<64x1x64x1xi4> to vector<1x64x1x64xi4>
    %wv = vector.transfer_read %w[%e, %n, %k], %pi4 {in_bounds = [true, true, true]} : tensor<256x512x2048xi4>, vector<1x64x64xi4>
    %sc = vector.shape_cast %st : vector<1x64x1x64xf16> to vector<1x64x64xf16>
    %zc = vector.shape_cast %zt : vector<1x64x1x64xi4> to vector<1x64x64xi4>
    %wi = arith.extui %wv : vector<1x64x64xi4> to vector<1x64x64xi32>
    %zi = arith.extui %zc : vector<1x64x64xi4> to vector<1x64x64xi32>
    %d = arith.subi %wi, %zi : vector<1x64x64xi32>
    %q = arith.sitofp %d : vector<1x64x64xi32> to vector<1x64x64xf16>
    %deq = arith.mulf %q, %sc : vector<1x64x64xf16>
    %t = vector.transfer_write %deq, %tile[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x64xf16>, tensor<1x64x64xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<32x2048xf16>, vector<32x64xf16>
    %b = vector.transfer_read %t[%c0, %c0, %c0], %pf16 {in_bounds = [true, true]} : tensor<1x64x64xf16>, vector<64x64xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %b, %acc : vector<32x64xf16>, vector<64x64xf16> into vector<32x64xf16>
    scf.yield %c : vector<32x64xf16>
  }
  return %r : vector<32x64xf16>
}

// -----

// Without fold-zero-point, only the scale moves. A rank-2 tile with an f32
// accumulator keeps its loop as it is.
// With the fold, the start contraction has the same rectangular shape.
// CHECK-LABEL: func.func @scale_only
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %{{.*}}) -> (vector<16x32xf32>)
// CHECK-NOT: arith.mulf
// CHECK: arith.ori %{{.*}} : vector<32x64xi16>
// CHECK: %[[START:.+]] = vector.contract {{.*}} : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf32>
// CHECK: %[[P:.+]] = vector.contract {{.*}}, %[[START]] : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf32>
// CHECK: math.fma %{{.*}}, %[[P]], %[[ACC]] : vector<16x32xf32>
// NOZP-LABEL: func.func @scale_only
// NOZP: scf.for {{.*}} iter_args(%[[ACC:.+]] = %{{.*}}) -> (vector<16x32xf32>)
// NOZP-NOT: arith.mulf
// NOZP-NOT: arith.ori
// NOZP: %[[Q:.+]] = arith.sitofp %{{.*}} : vector<32x64xi32> to vector<32x64xf16>
// NOZP: %[[P:.+]] = vector.contract {{.*}} %{{.*}}, %[[Q]], %{{.*}} : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf32>
// NOZP: %[[OUT:.+]] = math.fma %{{.*}}, %[[P]], %[[ACC]] : vector<16x32xf32>
// NOZP: scf.yield %[[OUT]]
// NOZP-NOT: arith.truncf
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @scale_only(%x: tensor<16x2048xf16>, %w: tensor<32x2048xi4>,
                      %s: tensor<32x32xf16>, %z: tensor<32x32xi4>) -> vector<16x32xf32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %pf16 = ub.poison : f16
  %init = arith.constant dense<0.0> : vector<16x32xf32>
  %r = scf.for %k = %c0 to %c2048 step %c64 iter_args(%acc = %init) -> (vector<16x32xf32>) {
    %g = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%k)
    %sv = vector.transfer_read %s[%c0, %g], %pf16 {in_bounds = [true, true]} : tensor<32x32xf16>, vector<32x1xf16>
    %sb = vector.broadcast %sv : vector<32x1xf16> to vector<32x64xf16>
    %zv = vector.transfer_read %z[%c0, %g], %pi4 {in_bounds = [true, true]} : tensor<32x32xi4>, vector<32x1xi4>
    %zb = vector.broadcast %zv : vector<32x1xi4> to vector<32x64xi4>
    %wv = vector.transfer_read %w[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<32x2048xi4>, vector<32x64xi4>
    %wi = arith.extui %wv : vector<32x64xi4> to vector<32x64xi32>
    %zi = arith.extui %zb : vector<32x64xi4> to vector<32x64xi32>
    %d = arith.subi %wi, %zi : vector<32x64xi32>
    %q = arith.sitofp %d : vector<32x64xi32> to vector<32x64xf16>
    %b = arith.mulf %sb, %q : vector<32x64xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<16x2048xf16>, vector<16x64xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %b, %acc : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf32>
    scf.yield %c : vector<16x32xf32>
  }
  return %r : vector<16x32xf32>
}

// -----

// A scale that changes along k stays where it is.
// CHECK-LABEL: func.func @scale_per_k
// CHECK: arith.mulf
// CHECK-NOT: math.fma
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @scale_per_k(%x: tensor<16x2048xf16>, %w: tensor<32x2048xi4>,
                       %s: tensor<32x2048xf16>) -> vector<16x32xf32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %pf16 = ub.poison : f16
  %init = arith.constant dense<0.0> : vector<16x32xf32>
  %r = scf.for %k = %c0 to %c2048 step %c64 iter_args(%acc = %init) -> (vector<16x32xf32>) {
    %sv = vector.transfer_read %s[%c0, %k], %pf16 {in_bounds = [true, true]} : tensor<32x2048xf16>, vector<32x64xf16>
    %wv = vector.transfer_read %w[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<32x2048xi4>, vector<32x64xi4>
    %wi = arith.extui %wv : vector<32x64xi4> to vector<32x64xi32>
    %q = arith.uitofp %wi : vector<32x64xi32> to vector<32x64xf16>
    %b = arith.mulf %q, %sv : vector<32x64xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<16x2048xf16>, vector<16x64xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %b, %acc : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf32>
    scf.yield %c : vector<16x32xf32>
  }
  return %r : vector<16x32xf32>
}

// An f64 accumulator is not rewritten.
// CHECK-LABEL: func.func @f64_acc
// CHECK: arith.mulf
// CHECK-NOT: math.fma
func.func @f64_acc(%x: tensor<16x2048xf16>, %w: tensor<32x2048xi4>,
                   %s: tensor<32x32xf16>) -> vector<16x32xf64> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %init = arith.constant dense<0.0> : vector<16x32xf64>
  %r = scf.for %k = %c0 to %c2048 step %c64 iter_args(%acc = %init) -> (vector<16x32xf64>) {
    %g = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%k)
    %sv = vector.transfer_read %s[%c0, %g], %f0 {in_bounds = [true, true]} : tensor<32x32xf16>, vector<32x1xf16>
    %sb = vector.broadcast %sv : vector<32x1xf16> to vector<32x64xf16>
    %wv = vector.transfer_read %w[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<32x2048xi4>, vector<32x64xi4>
    %wi = arith.extui %wv : vector<32x64xi4> to vector<32x64xi32>
    %q = arith.uitofp %wi : vector<32x64xi32> to vector<32x64xf16>
    %b = arith.mulf %q, %sb : vector<32x64xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<16x2048xf16>, vector<16x64xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %b, %acc : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf64>
    scf.yield %c : vector<16x32xf64>
  }
  return %r : vector<16x32xf64>
}

// The trailing dim of this scale comes from the vector elements of the
// memref, not from the read, so it is not known to be constant.
// CHECK-LABEL: func.func @vector_elements
// CHECK: arith.mulf
// CHECK-NOT: math.fma
func.func @vector_elements(%x: tensor<16x2048xf16>, %w: tensor<3x2048xi4>,
                           %s: memref<3xvector<4xf16>>) -> vector<16x3xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %pad = arith.constant dense<0.0> : vector<4xf16>
  %init = arith.constant dense<0.0> : vector<16x3xf32>
  %r = scf.for %k = %c0 to %c2048 step %c4 iter_args(%acc = %init) -> (vector<16x3xf32>) {
    %sv = vector.transfer_read %s[%c0], %pad {in_bounds = [true]} : memref<3xvector<4xf16>>, vector<3x4xf16>
    %wv = vector.transfer_read %w[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<3x2048xi4>, vector<3x4xi4>
    %wi = arith.extui %wv : vector<3x4xi4> to vector<3x4xi32>
    %q = arith.uitofp %wi : vector<3x4xi32> to vector<3x4xf16>
    %b = arith.mulf %sv, %q : vector<3x4xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<16x2048xf16>, vector<16x4xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %b, %acc : vector<16x4xf16>, vector<3x4xf16> into vector<16x3xf32>
    scf.yield %c : vector<16x3xf32>
  }
  return %r : vector<16x3xf32>
}

// -----

// A zero point that changes along k stays where it is.
// CHECK-LABEL: func.func @zero_point_per_k
// CHECK: arith.mulf
// CHECK-NOT: math.fma
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @zero_point_per_k(%x: tensor<16x2048xf16>, %w: tensor<32x2048xi4>,
                            %s: tensor<32x32xf16>, %z: tensor<32x2048xi4>) -> vector<16x32xf32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %pf16 = ub.poison : f16
  %init = arith.constant dense<0.0> : vector<16x32xf32>
  %r = scf.for %k = %c0 to %c2048 step %c64 iter_args(%acc = %init) -> (vector<16x32xf32>) {
    %g = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%k)
    %sv = vector.transfer_read %s[%c0, %g], %pf16 {in_bounds = [true, true]} : tensor<32x32xf16>, vector<32x1xf16>
    %sb = vector.broadcast %sv : vector<32x1xf16> to vector<32x64xf16>
    %zv = vector.transfer_read %z[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<32x2048xi4>, vector<32x64xi4>
    %wv = vector.transfer_read %w[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<32x2048xi4>, vector<32x64xi4>
    %wi = arith.extui %wv : vector<32x64xi4> to vector<32x64xi32>
    %zi = arith.extui %zv : vector<32x64xi4> to vector<32x64xi32>
    %d = arith.subi %wi, %zi : vector<32x64xi32>
    %q = arith.sitofp %d : vector<32x64xi32> to vector<32x64xf16>
    %b = arith.mulf %sb, %q : vector<32x64xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<16x2048xf16>, vector<16x64xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %b, %acc : vector<16x64xf16>, vector<32x64xf16> into vector<16x32xf32>
    scf.yield %c : vector<16x32xf32>
  }
  return %r : vector<16x32xf32>
}

// -----

// A tensor round trip from a fixed-size write to a scalable read does not
// cover the same elements, so the loop is left as it is.
// CHECK-LABEL: func.func @scalable_round_trip
// CHECK: arith.mulf
// CHECK-NOT: math.fma
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @scalable_round_trip(%x: tensor<16x2048xf16>, %w: tensor<32x2048xi4>,
                               %s: tensor<32x32xf16>) -> vector<16x32xf32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %f0 = arith.constant 0.0 : f16
  %pi4 = ub.poison : i4
  %pf16 = ub.poison : f16
  %init = arith.constant dense<0.0> : vector<16x32xf32>
  %tile = tensor.empty() : tensor<32x64xf16>
  %r = scf.for %k = %c0 to %c2048 step %c64 iter_args(%acc = %init) -> (vector<16x32xf32>) {
    %g = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%k)
    %sv = vector.transfer_read %s[%c0, %g], %pf16 {in_bounds = [true, true]} : tensor<32x32xf16>, vector<32x1xf16>
    %sb = vector.broadcast %sv : vector<32x1xf16> to vector<32x64xf16>
    %wv = vector.transfer_read %w[%c0, %k], %pi4 {in_bounds = [true, true]} : tensor<32x2048xi4>, vector<32x64xi4>
    %wi = arith.extui %wv : vector<32x64xi4> to vector<32x64xi32>
    %q = arith.uitofp %wi : vector<32x64xi32> to vector<32x64xf16>
    %b = arith.mulf %sb, %q : vector<32x64xf16>
    %t = vector.transfer_write %b, %tile[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, tensor<32x64xf16>
    %bs = vector.transfer_read %t[%c0, %c0], %f0 {in_bounds = [true, true]} : tensor<32x64xf16>, vector<32x[64]xf16>
    %xv = vector.transfer_read %x[%c0, %k], %f0 {in_bounds = [true, true]} : tensor<16x2048xf16>, vector<16x[64]xf16>
    %c = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %xv, %bs, %acc : vector<16x[64]xf16>, vector<32x[64]xf16> into vector<16x32xf32>
    scf.yield %c : vector<16x32xf32>
  }
  return %r : vector<16x32xf32>
}
