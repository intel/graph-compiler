// RUN: gc-opt %s --buffer-nested-parallel-loop-hoisting \
// RUN:           --allow-unregistered-dialect -split-input-file | FileCheck %s

func.func @nested_forall(%arg0: memref<2xf32>) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<2xf32>
    memref.copy %arg0, %alloc : memref<2xf32> to memref<2xf32>
    scf.forall (%i) in (%c16) {
      %alloc_0 = memref.alloc() : memref<4xf32>
      %alloc_1 = memref.alloc() : memref<8xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc) : (memref<2xf32>) -> ()
        "test.source"(%alloc_1) : (memref<8xf32>) -> ()
        %alloc_2 = memref.alloc() : memref<16xf32>
        "test.source"(%alloc_2) : (memref<16xf32>) -> ()
      }
      "test.source"(%alloc) : (memref<2xf32>) -> ()
      "test.source"(%alloc_0) : (memref<4xf32>) -> ()
    }
    return
}
// CHECK-LABEL: func @nested_forall
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2xf32>
//  CHECK-NEXT: memref.copy %arg0, %[[ALLOC0]]
//  CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc() : memref<128xf32>
//  CHECK-NEXT: scf.forall
//  CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc() : memref<4xf32>
//       CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC1]][%{{.*}}] [{{.*}}] [{{.*}}]
//  CHECK-NEXT: scf.forall
//       CHECK: %[[ALLOC3:.*]] = memref.alloc() : memref<16xf32>

func.func @mixed_forall_and_for(
    %lb: index,
    %ub: index,
    %step: index,
    %arg0: memref<2xf32>) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %0 = memref.alloc() : memref<2xf32>
    scf.forall (%i) in (%c16) {
      %alloc = memref.alloc() : memref<4xf32>
      scf.forall (%j) in (%c4) {
        %alloc_0 = memref.alloc() : memref<8xf32>
        %1 = scf.for %k = %lb to %ub step %step
          iter_args(%iterBuf = %arg0) -> (memref<2xf32>) {
          "test.source"(%alloc) : (memref<4xf32>) -> ()
          "test.source"(%alloc_0) : (memref<8xf32>) -> ()
          scf.yield %0 : memref<2xf32>
        }
      }
    }
    return
}
// CHECK-LABEL: func @mixed_forall_and_for
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2xf32>
//       CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<64xf32>
//  CHECK-NEXT: scf.forall
//       CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[ALLOC1]][%{{.*}}] [{{.*}}] [{{.*}}]
//  CHECK-NEXT:   scf.forall
//  CHECK-NEXT:     %[[ALLOC2:.*]] = memref.alloc() : memref<8xf32>
//  CHECK-NEXT:     scf.for
//       CHECK:     scf.yield %[[ALLOC0]]

func.func @nested_forall_with_dynamic_shape(%arg0: index) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.forall (%i) in (%c16) {
      %alloc = memref.alloc(%arg0) : memref<?xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc) : (memref<?xf32>) -> ()
      }
    }
    return
}
// CHECK-LABEL: func @nested_forall_with_dynamic_shape
//   CHECK-NOT: memref.alloc
//       CHECK: scf.forall
//  CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc(%arg0) : memref<?xf32>
//  CHECK-NEXT: scf.forall

func.func @nested_forall_with_dynamic_range(%arg0: index) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.forall (%i) in (%arg0) {
      %alloc = memref.alloc() : memref<4xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc) : (memref<4xf32>) -> ()
      }
    }
    return
}
// CHECK-LABEL: func @nested_forall_with_dynamic_range
//   CHECK-NOT: memref.alloc
//       CHECK: scf.forall
//  CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT: scf.forall

func.func @nested_forall_with_multi_blocks(
    %lb: index,
    %ub: index,
    %step: index,
    %arg0: memref<2xf32>) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %0 = memref.alloc() : memref<2xf32>
    scf.forall (%i) in (%c16) {
      scf.forall(%j1) in (%c4) {
        %alloc = memref.alloc() : memref<4xf32>
        scf.forall (%k1) in (%c4) {
          %1 = scf.for %kk = %lb to %ub step %step
            iter_args(%iterBuf = %arg0) -> (memref<2xf32>) {
            %alloc_0 = memref.alloc() : memref<2xf32>
            "test.source"(%alloc) : (memref<4xf32>) -> ()
            "test.source"(%alloc_0) : (memref<2xf32>) -> ()
            scf.yield %0 : memref<2xf32>
          }
        }
      }

      scf.forall (%j2) in (%c8) {
        %alloc_2 = memref.alloc() : memref<8xf32>
        scf.forall (%k2) in (%c4) {
          "test.source"(%alloc_2) : (memref<8xf32>) -> ()
        }
      }
    }
    return
}
// CHECK-LABEL: func @nested_forall_with_multi_blocks
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<2xf32>
//  CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc() : memref<256xf32>
//  CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc() : memref<1024xf32>
//  CHECK-NEXT: scf.forall
//  CHECK-NEXT:   scf.forall
//   CHECK-NOT:     memref.alloc
//       CHECK:     %[[SUBVIEW:.*]] = memref.subview %[[ALLOC1]][%{{.*}}] [{{.*}}] [{{.*}}]
//  CHECK-NEXT:     scf.forall
//  CHECK-NEXT:       %[[ALLOC3:.*]] = memref.alloc() : memref<2xf32>
//  CHECK-NEXT:       scf.for
//       CHECK:   scf.forall
//       CHECK:     %[[SUBVIEW:.*]] = memref.subview %[[ALLOC2]][%{{.*}}] [{{.*}}] [{{.*}}]
//  CHECK-NEXT:     scf.forall

func.func @nested_forall_nd() {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.forall (%i) in (%c16) {
      %alloc_0 = memref.alloc() : memref<2x4xf32>
      %alloc_1 = memref.alloc() : memref<4x8xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc_0) : (memref<2x4xf32>) -> ()
        %alloc_2 = memref.alloc() : memref<2x16xf32>
        "test.source"(%alloc_2) : (memref<2x16xf32>) -> ()
      }
      "test.source"(%alloc_1) : (memref<4x8xf32>) -> ()
    }
    return
}
// CHECK-LABEL: func @nested_forall_nd
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<32x4xf32>
//  CHECK-NEXT: scf.forall
//       CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC0]][%{{.*}}, 0] [2, 4] [1, 1]
//  CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc() : memref<4x8xf32>
//  CHECK-NEXT: scf.forall
//       CHECK: %[[ALLOC2:.*]] = memref.alloc() : memref<2x16xf32>