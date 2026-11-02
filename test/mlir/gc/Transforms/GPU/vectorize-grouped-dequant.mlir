// RUN: gc-opt %s --simplify-indexing-maps --vectorize | FileCheck %s

// Broadcast and collapse_shape expansion must vectorize to transfer_read,
// broadcast, transpose and shape_cast without tensor intermediates.
// CHECK-LABEL: func.func @two_groups
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.collapse_shape
// CHECK: %[[S:.+]] = vector.transfer_read %{{.*}} : tensor<32x2xf16>, vector<32x2xf16>
// CHECK: %[[SB:.+]] = vector.broadcast %[[S]] : vector<32x2xf16> to vector<32x32x2xf16>
// CHECK: %[[ST:.+]] = vector.transpose %[[SB]], [1, 2, 0] : vector<32x32x2xf16> to vector<32x2x32xf16>
// CHECK: %[[Z:.+]] = vector.transfer_read %{{.*}} : tensor<32x2xi4>, vector<32x2xi4>
// CHECK: %[[ZB:.+]] = vector.broadcast %[[Z]] : vector<32x2xi4> to vector<32x32x2xi4>
// CHECK: %[[ZT:.+]] = vector.transpose %[[ZB]], [1, 2, 0] : vector<32x32x2xi4> to vector<32x2x32xi4>
// CHECK: %[[W:.+]] = vector.transfer_read %{{.*}} : tensor<32x64xi4>, vector<32x64xi4>
// CHECK-DAG: %[[SC:.+]] = vector.shape_cast %[[ST]] : vector<32x2x32xf16> to vector<32x64xf16>
// CHECK-DAG: %[[ZC:.+]] = vector.shape_cast %[[ZT]] : vector<32x2x32xi4> to vector<32x64xi4>
// CHECK-DAG: arith.extui %[[W]] : vector<32x64xi4> to vector<32x64xi32>
// CHECK-DAG: arith.extui %[[ZC]] : vector<32x64xi4> to vector<32x64xi32>
// CHECK: arith.mulf %{{.*}}, %[[SC]] : vector<32x64xf16>
// CHECK-NOT: tensor.collapse_shape
func.func @two_groups(%w: tensor<32x64xi4>, %s: tensor<32x2xf16>,
                      %z: tensor<32x2xi4>, %o: tensor<32x64xf16>) -> tensor<32x64xf16> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 32)>,
                     affine_map<(d0, d1) -> (d0, d1 floordiv 32)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%w, %s, %z : tensor<32x64xi4>, tensor<32x2xf16>, tensor<32x2xi4>) outs(%o : tensor<32x64xf16>) {
  ^bb0(%in: i4, %in_s: f16, %in_z: i4, %out: f16):
    %0 = arith.extui %in : i4 to i32
    %1 = arith.extui %in_z : i4 to i32
    %2 = arith.subi %0, %1 : i32
    %3 = arith.sitofp %2 : i32 to f16
    %4 = arith.mulf %3, %in_s : f16
    linalg.yield %4 : f16
  } -> tensor<32x64xf16>
  return %r : tensor<32x64xf16>
}
