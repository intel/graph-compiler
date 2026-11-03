// RUN: gc-opt %s --peeled-for-to-if --split-input-file | FileCheck %s

// Every spelling of the peeling split bound that scf.for loop peeling has
// produced must be recognized as an at-most-one-iteration loop.

// Pre-canonicalization affine.apply form (LLVM < 3dcd66d01103).
// CHECK-LABEL: @split_bound_affine_apply
// CHECK-NOT: scf.for
// CHECK: scf.if
#map = affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>
func.func @split_bound_affine_apply(%ub: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %split = affine.apply #map()[%c0, %ub, %c64]
  %r = scf.for %i = %split to %ub step %c64 iter_args(%a = %init) -> f32 {
    %n = arith.addf %a, %a : f32
    scf.yield %n : f32
  }
  return %r : f32
}

// -----

// Clamped affine.max form (LLVM >= 3dcd66d01103).
// CHECK-LABEL: @split_bound_affine_max
// CHECK-NOT: scf.for
// CHECK: scf.if
#map = affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2, s0)>
func.func @split_bound_affine_max(%ub: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %split = affine.max #map()[%c0, %ub, %c64]
  %r = scf.for %i = %split to %ub step %c64 iter_args(%a = %init) -> f32 {
    %n = arith.addf %a, %a : f32
    scf.yield %n : f32
  }
  return %r : f32
}

// -----

// Canonicalized constant-step form.
// CHECK-LABEL: @split_bound_floordiv
// CHECK-NOT: scf.for
// CHECK: scf.if
#map = affine_map<()[s0] -> ((s0 floordiv 64) * 64)>
func.func @split_bound_floordiv(%ub: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %split = affine.apply #map()[%ub]
  %r = scf.for %i = %split to %ub step %c64 iter_args(%a = %init) -> f32 {
    %n = arith.addf %a, %a : f32
    scf.yield %n : f32
  }
  return %r : f32
}

// -----

// arith form (LLVM >= 1b6c091ba8a7).
// CHECK-LABEL: @split_bound_arith
// CHECK-NOT: scf.for
// CHECK: scf.if
func.func @split_bound_arith(%ub: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %min = arith.minsi %ub, %c0 : index
  %diff = arith.subi %ub, %min : index
  %rem = arith.remui %diff, %c64 : index
  %split = arith.subi %ub, %rem : index
  %r = scf.for %i = %split to %ub step %c64 iter_args(%a = %init) -> f32 {
    %n = arith.addf %a, %a : f32
    scf.yield %n : f32
  }
  return %r : f32
}

// -----

// All-constant bounds with a single iteration.
// CHECK-LABEL: @constant_single_iteration
// CHECK-NOT: scf.for
// CHECK: scf.if
func.func @constant_single_iteration(%init: f32) -> f32 {
  %c64 = arith.constant 64 : index
  %c100 = arith.constant 100 : index
  %r = scf.for %i = %c64 to %c100 step %c64 iter_args(%a = %init) -> f32 {
    %n = arith.addf %a, %a : f32
    scf.yield %n : f32
  }
  return %r : f32
}

// -----

// A multi-iteration loop must stay an scf.for.
// CHECK-LABEL: @not_single_iteration
// CHECK: scf.for
// CHECK-NOT: scf.if
func.func @not_single_iteration(%ub: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %r = scf.for %i = %c0 to %ub step %c64 iter_args(%a = %init) -> f32 {
    %n = arith.addf %a, %a : f32
    scf.yield %n : f32
  }
  return %r : f32
}
