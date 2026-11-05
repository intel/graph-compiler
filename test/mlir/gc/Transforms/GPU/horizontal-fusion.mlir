// RUN: gc-opt %s --horizontal-fusion --split-input-file | FileCheck %s

// Fuse three sibling matmuls sharing A. Later initialization producers must be
// hoisted before the fused op, while an earlier result consumer stays valid.
// CHECK-LABEL: func.func @matmul_shared_lhs
// CHECK:         %[[INIT1:.+]] = linalg.fill
// CHECK:         %[[INIT2:.+]] = linalg.fill
// CHECK-NEXT:    %[[FUSED:.+]]:3 = linalg.generic
// CHECK-SAME:      ins(%arg0, %arg1, %arg2, %arg3 :
// CHECK-SAME:      outs(%arg4, %[[INIT1]], %[[INIT2]] :
// CHECK-SAME:      gc.fused_matmul
// CHECK:           linalg.yield %{{.*}}, %{{.*}}, %{{.*}} : f32, f32, f32
// CHECK-NEXT:    } -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
// CHECK-NEXT:    %[[VIEW:.+]] = tensor.cast %[[FUSED]]#0
// CHECK-NOT:     linalg.matmul
// CHECK:         return %[[VIEW]], %[[FUSED]]#1, %[[FUSED]]#2
func.func @matmul_shared_lhs(%a: tensor<64x256xf16>, %b0: tensor<256x64xf16>,
                            %b1: tensor<256x64xf16>, %b2: tensor<256x64xf16>,
                            %init0: tensor<64x64xf32>)
    -> (tensor<?x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a, %b0 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %view = tensor.cast %r0 : tensor<64x64xf32> to tensor<?x64xf32>
  %zero = arith.constant 0.0 : f32
  %empty1 = tensor.empty() : tensor<64x64xf32>
  %init1 = linalg.fill ins(%zero : f32) outs(%empty1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %empty2 = tensor.empty() : tensor<64x64xf32>
  %one = arith.constant 1.0 : f32
  %init2 = linalg.fill ins(%one : f32) outs(%empty2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r2 = linalg.matmul ins(%a, %b2 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %view, %r1, %r2 : tensor<?x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Shared RHS is deduplicated even though it is not the first generic input.
// CHECK-LABEL: func.func @matmul_shared_rhs
// CHECK-NEXT:    %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:      ins(%arg0, %arg2, %arg1 :
// CHECK-SAME:      outs(%arg3, %arg4 :
// CHECK-SAME:      gc.fused_matmul
// CHECK:           linalg.yield %{{.*}}, %{{.*}} : f32, f32
// CHECK-NEXT:    } -> (tensor<64x64xf32>, tensor<64x64xf32>)
// CHECK-NEXT:    return %[[FUSED]]#0, %[[FUSED]]#1
func.func @matmul_shared_rhs(%a0: tensor<64x256xf16>, %a1: tensor<64x256xf16>,
                            %b: tensor<256x64xf16>, %init0: tensor<64x64xf32>,
                            %init1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a0, %b : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a1, %b : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Identical matrix shapes alone are not sharing.
// CHECK-LABEL: func.func @no_shared_input
// CHECK-NEXT:    %[[R0:.+]] = linalg.matmul
// CHECK-NEXT:    %[[R1:.+]] = linalg.matmul
// CHECK-NEXT:    return %[[R0]], %[[R1]]
func.func @no_shared_input(%a0: tensor<64x256xf16>, %a1: tensor<64x256xf16>,
                          %b0: tensor<256x64xf16>, %b1: tensor<256x64xf16>,
                          %init0: tensor<64x64xf32>, %init1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a0, %b0 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a1, %b1 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Dependence through a pure cast must not be mistaken for horizontal fusion.
// CHECK-LABEL: func.func @dependent_matmuls
// CHECK-NEXT:    %[[R0:.+]] = linalg.matmul
// CHECK:         %[[R1:.+]] = linalg.matmul
// CHECK-NEXT:    return %[[R0]], %[[R1]]
func.func @dependent_matmuls(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
                            %init0: tensor<64x64xf32>, %init1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>) outs(%init0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %view = tensor.cast %r0 : tensor<64x64xf32> to tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %view : tensor<64x64xf32>, tensor<64x64xf32>) outs(%init1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// A dynamic initialization defined between siblings through tensor.dim is
// hoisted before the fused op.
// CHECK-LABEL: func.func @dynamic_init_between_siblings
// CHECK:         %[[DIM:.+]] = tensor.dim %arg0
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty(%[[DIM]])
// CHECK-NEXT:    %[[INIT1:.+]] = linalg.fill {{.*}} outs(%[[EMPTY]] :
// CHECK-NEXT:    %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:      ins(%arg0, %arg1, %arg2 :
// CHECK-SAME:      outs(%arg3, %[[INIT1]] :
// CHECK-SAME:      gc.fused_matmul
// CHECK-NOT:     linalg.matmul
// CHECK:         return %[[FUSED]]#0, %[[FUSED]]#1
func.func @dynamic_init_between_siblings(%a: tensor<?x256xf16>, %b0: tensor<256x64xf16>,
                                        %b1: tensor<256x64xf16>, %init0: tensor<?x64xf32>)
    -> (tensor<?x64xf32>, tensor<?x64xf32>) {
  %r0 = linalg.matmul ins(%a, %b0 : tensor<?x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<?x64xf32>) -> tensor<?x64xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %a, %c0 : tensor<?x256xf16>
  %empty = tensor.empty(%dim) : tensor<?x64xf32>
  %zero = arith.constant 0.0 : f32
  %init1 = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<?x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %r0, %r1 : tensor<?x64xf32>, tensor<?x64xf32>
}

// -----

// A memory effect between siblings is a barrier.
// CHECK-LABEL: func.func @side_effect_between_siblings
// CHECK:         linalg.matmul
// CHECK-NEXT:    memref.store
// CHECK-NEXT:    linalg.matmul
// CHECK-NOT:     gc.fused_matmul
func.func @side_effect_between_siblings(%a: tensor<64x256xf16>, %b0: tensor<256x64xf16>,
                                       %b1: tensor<256x64xf16>, %init0: tensor<64x64xf32>,
                                       %init1: tensor<64x64xf32>, %m: memref<f32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a, %b0 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %one = arith.constant 1.0 : f32
  memref.store %one, %m[] : memref<f32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// An impure operand definition between siblings cannot be hoisted.
// CHECK-LABEL: func.func @impure_init_between_siblings
// CHECK:         linalg.matmul
// CHECK-NEXT:    bufferization.to_tensor
// CHECK-NEXT:    linalg.matmul
// CHECK-NOT:     gc.fused_matmul
func.func @impure_init_between_siblings(%a: tensor<64x256xf16>, %b0: tensor<256x64xf16>,
                                       %b1: tensor<256x64xf16>, %init0: tensor<64x64xf32>,
                                       %m: memref<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a, %b0 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %init1 = bufferization.to_tensor %m : memref<64x64xf32> to tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Unrelated dynamic row extents cannot be proven equal.
// CHECK-LABEL: func.func @unprovable_dynamic_rows
// CHECK-NEXT:    linalg.matmul
// CHECK-NEXT:    linalg.matmul
// CHECK-NEXT:    return
func.func @unprovable_dynamic_rows(%a0: tensor<?x256xf16>, %a1: tensor<?x256xf16>,
                                  %b: tensor<256x64xf16>, %init0: tensor<?x64xf32>,
                                  %init1: tensor<?x64xf32>)
    -> (tensor<?x64xf32>, tensor<?x64xf32>) {
  %r0 = linalg.matmul ins(%a0, %b : tensor<?x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<?x64xf32>) -> tensor<?x64xf32>
  %r1 = linalg.matmul ins(%a1, %b : tensor<?x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %r0, %r1 : tensor<?x64xf32>, tensor<?x64xf32>
}

// -----

// Dynamic row extents derived from the same tensor.dim are provably equal.
// CHECK-LABEL: func.func @provable_dynamic_rows
// CHECK:         tensor.empty
// CHECK-NEXT:    linalg.generic
// CHECK-SAME:      gc.fused_matmul
// CHECK-NOT:     linalg.matmul
func.func @provable_dynamic_rows(%a0: tensor<?x256xf16>, %b: tensor<256x64xf16>,
                                %init0: tensor<?x64xf32>, %init1: tensor<?x64xf32>)
    -> (tensor<?x64xf32>, tensor<?x64xf32>) {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %a0, %c0 : tensor<?x256xf16>
  %a1 = tensor.empty(%dim) : tensor<?x256xf16>
  %r0 = linalg.matmul ins(%a0, %b : tensor<?x256xf16>, tensor<256x64xf16>) outs(%init0 : tensor<?x64xf32>) -> tensor<?x64xf32>
  %r1 = linalg.matmul ins(%a1, %b : tensor<?x256xf16>, tensor<256x64xf16>) outs(%init1 : tensor<?x64xf32>) -> tensor<?x64xf32>
  return %r0, %r1 : tensor<?x64xf32>, tensor<?x64xf32>
}

// -----

// The group's common inputs narrow to A after the second matmul, so a later
// matmul that only shares B with the seed is not added.
// CHECK-LABEL: func.func @three_way_narrowing
// CHECK-NEXT:    %[[FUSED:.+]]:2 = linalg.generic
// CHECK-SAME:      ins(%arg0, %arg1, %arg2 :
// CHECK-SAME:      gc.fused_matmul
// CHECK:         %[[R2:.+]] = linalg.matmul ins(%arg3, %arg1
// CHECK-NEXT:    return %[[FUSED]]#0, %[[FUSED]]#1, %[[R2]]
func.func @three_way_narrowing(%a: tensor<64x256xf16>, %b: tensor<256x64xf16>,
                              %c: tensor<256x64xf16>, %d: tensor<64x256xf16>,
                              %i0: tensor<64x64xf32>, %i1: tensor<64x64xf32>,
                              %i2: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul ins(%a, %b : tensor<64x256xf16>, tensor<256x64xf16>) outs(%i0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %c : tensor<64x256xf16>, tensor<256x64xf16>) outs(%i1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r2 = linalg.matmul ins(%d, %b : tensor<64x256xf16>, tensor<256x64xf16>) outs(%i2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1, %r2 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Transposed B maps are preserved; mixed layouts are not fused.
#ma = affine_map<(m, n, k) -> (m, k)>
#mbt = affine_map<(m, n, k) -> (n, k)>
#mc = affine_map<(m, n, k) -> (m, n)>
// CHECK-DAG:   #[[MAP_BT:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-LABEL: func.func @transposed_b
// CHECK-NEXT:    linalg.generic {indexing_maps = [#{{.*}}, #[[MAP_BT]], #[[MAP_BT]],
// CHECK-SAME:      gc.fused_matmul
// CHECK-LABEL: func.func @mixed_b_layouts
// CHECK-NEXT:    linalg.matmul
// CHECK-NEXT:    linalg.matmul
func.func @transposed_b(%a: tensor<64x256xf16>, %b0: tensor<64x256xf16>,
                       %b1: tensor<64x256xf16>, %i0: tensor<64x64xf32>,
                       %i1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul indexing_maps = [#ma, #mbt, #mc] ins(%a, %b0 : tensor<64x256xf16>, tensor<64x256xf16>) outs(%i0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul indexing_maps = [#ma, #mbt, #mc] ins(%a, %b1 : tensor<64x256xf16>, tensor<64x256xf16>) outs(%i1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}
func.func @mixed_b_layouts(%a: tensor<64x256xf16>, %b0: tensor<64x256xf16>,
                          %b1: tensor<256x64xf16>, %i0: tensor<64x64xf32>,
                          %i1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul indexing_maps = [#ma, #mbt, #mc] ins(%a, %b0 : tensor<64x256xf16>, tensor<64x256xf16>) outs(%i0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%i1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// Different discardable attributes keep the matmuls apart.
// CHECK-LABEL: func.func @attribute_mismatch
// CHECK-NEXT:    linalg.matmul {foo}
// CHECK-NEXT:    linalg.matmul
// CHECK-NEXT:    return
func.func @attribute_mismatch(%a: tensor<64x256xf16>, %b0: tensor<256x64xf16>,
                             %b1: tensor<256x64xf16>, %i0: tensor<64x64xf32>,
                             %i1: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %r0 = linalg.matmul {foo} ins(%a, %b0 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%i0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r1 = linalg.matmul ins(%a, %b1 : tensor<64x256xf16>, tensor<256x64xf16>) outs(%i1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r0, %r1 : tensor<64x64xf32>, tensor<64x64xf32>
}
