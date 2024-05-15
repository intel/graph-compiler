// RUN: gc-opt -allow-unregistered-dialect -p 'builtin.module(func.func(merge-alloc{check}))'  %s | FileCheck %s

// CHECK-DAG: func.func @basic() -> memref<8x64xf32>  attributes {__mergealloc_scope = [[TOPSCOPE:[0-9]+]]
func.func @basic() -> memref<8x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %ctrue = arith.constant 1 : i1
  // b is used in return, complex lifetime
  // CHECK-DAG: %[[B:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], -2, -2>}
  %b = memref.alloc() : memref<8x64xf32>
  "test.source"(%b)  : (memref<8x64xf32>) -> ()
  // c and d has overlapping lifetime
  // CHECK-DAG: %[[C:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 11, 14>}
  %c = memref.alloc() : memref<8x64xf32>
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // CHECK-DAG: %[[D:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 13, 13>}
  %d = memref.alloc() : memref<8x64xf32>
  "test.source"(%d)  : (memref<8x64xf32>) -> ()
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // e and f have overlapping lifetime due to the loop
  // CHECK-DAG: %[[E:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 17, 22>}
  // CHECK-DAG: %[[F:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 17, 22>}
  %e = memref.alloc() : memref<8x64xf32>
  %f = memref.alloc() : memref<8x64xf32>
  // CHECK: scf.for
  scf.for %i = %c0 to %c5 step %c1 {
    "test.source"(%e)  : (memref<8x64xf32>) -> ()
    "test.source"(%f)  : (memref<8x64xf32>) -> ()
    // CHECK-DAG: %[[G:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 21, 21>}
    %g = memref.alloc() : memref<8x64xf32>
    "test.source"(%g)  : (memref<8x64xf32>) -> ()
  }
  // CHECK-DAG: %[[H:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE]], 24, 39>}
  %h = memref.alloc() : memref<8x64xf32>
  // CHECK: scf.forall
  scf.forall (%iv) in (%c5) {
    // check that the alloc in the forall should switch to another scope id
    // CHECK-NOT: array<i64: [[TOPSCOPE]]
    // CHECK-DAG: %[[L:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE:[0-9]+]], 27, 27>}
    %l = memref.alloc() : memref<8x64xf32>
    "test.source"(%h)  : (memref<8x64xf32>) -> ()
    "test.source"(%l)  : (memref<8x64xf32>) -> ()
    scf.for %i = %c0 to %c5 step %c1 {
      // CHECK-DAG: %[[G:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE]], 30, 30>}
      %g = memref.alloc() : memref<8x64xf32>
      "test.source"(%g)  : (memref<8x64xf32>) -> ()
    }
    // CHECK-DAG: %[[K:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE]], 33, 38>}
    %k = memref.alloc() : memref<8x64xf32>
    scf.if %ctrue {
      // CHECK-DAG: %[[J:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[FORSCOPE]], 35, 35>}
      %j = memref.alloc() : memref<8x64xf32>
      "test.source"(%j)  : (memref<8x64xf32>) -> ()
    } else {
      "test.source"(%k)  : (memref<8x64xf32>) -> ()
    }
    // CHECK-DAG: {__mergealloc_scope = [[FORSCOPE]] : i64}
  }
  return %b : memref<8x64xf32>
}

// CHECK-DAG: func.func @basic2() attributes {__mergealloc_scope = [[TOPSCOPE2:[0-9]+]]
func.func @basic2() {
  // CHECK-DAG: %[[B:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE2]], 4, 6>}
  %b = memref.alloc() : memref<8x64xi8>
  %cur = memref.subview %b[1,0][1,64][1,1] : memref<8x64xi8> to memref<1x64xi8, strided<[64, 1], offset: 64>>
  "test.source"(%cur)  : (memref<1x64xi8, strided<[64, 1], offset: 64>>) -> ()
  %cur2 = memref.subview %cur[0,0][1,16][1,1] : memref<1x64xi8, strided<[64, 1], offset: 64>> to memref<1x16xi8, strided<[64, 1], offset: 64>>
  "test.source"(%cur2)  : (memref<1x16xi8, strided<[64, 1], offset: 64>>) -> ()
  // CHECK-DAG: %[[C:.*]] = memref.alloc() {__mergealloc_lifetime = array<i64: [[TOPSCOPE2]], 8, 8>}
  %c = memref.alloc() : memref<8x64xi8>
  "test.source"(%c)  : (memref<8x64xi8>) -> ()
  return
}