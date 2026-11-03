// RUN: gc-opt %s --transform-interpreter | FileCheck %s

// Test the conditional memory-effect and speculation interfaces through dead
// code elimination and LICM. Both forms use fused branches and a scalar bound.
#lhs = affine_map<(e, m, n, k) -> (m, k)>
#weights = affine_map<(e, m, n, k) -> (e, n, k)>
#offsets = affine_map<(e, m, n, k) -> (e)>
#out = affine_map<(e, m, n, k) -> (m, n)>

module attributes {transform.with_named_sequence} {
  // Tensor destinations are values, not memory writes. Unused results may be
  // erased even when the grouped op has multiple output accumulators.
  // CHECK-LABEL: func.func @unused_tensor
  // CHECK-NEXT:    return
  // CHECK-NEXT:  }
  func.func @unused_tensor(%a: tensor<128x256xf16>, %w0: tensor<2x64x256xf16>,
                           %w1: tensor<2x64x256xf16>, %offs: tensor<2xi32>,
                           %bound: index, %init0: tensor<128x64xf32>,
                           %init1: tensor<128x64xf32>) {
    %r:2 = linalgx.grouped_matmul
        <{indexing_maps = [#lhs, #lhs, #weights, #weights, #offsets, #out, #out]}>
        ins(%a, %w0, %a, %w1, %offs : tensor<128x256xf16>, tensor<2x64x256xf16>,
            tensor<128x256xf16>, tensor<2x64x256xf16>, tensor<2xi32>)
        max_group_size(%bound)
        outs(%init0, %init1 : tensor<128x64xf32>, tensor<128x64xf32>)
        -> (tensor<128x64xf32>, tensor<128x64xf32>)
    return
  }

  // Buffer destinations are mutated, so the result-less op must survive DCE.
  // CHECK-LABEL: func.func @unused_buffer
  // CHECK-NEXT:    linalgx.grouped_matmul
  // CHECK-SAME:      max_group_size(%arg4)
  // CHECK-SAME:      outs(%arg5, %arg6 : memref<128x64xf32>, memref<128x64xf32>)
  // CHECK-NEXT:    return
  // CHECK-NEXT:  }
  func.func @unused_buffer(%a: memref<128x256xf16>, %w0: memref<2x64x256xf16>,
                           %w1: memref<2x64x256xf16>, %offs: memref<2xi32>,
                           %bound: index, %init0: memref<128x64xf32>,
                           %init1: memref<128x64xf32>) {
    linalgx.grouped_matmul
        <{indexing_maps = [#lhs, #lhs, #weights, #weights, #offsets, #out, #out]}>
        ins(%a, %w0, %a, %w1, %offs : memref<128x256xf16>, memref<2x64x256xf16>,
            memref<128x256xf16>, memref<2x64x256xf16>, memref<2xi32>)
        max_group_size(%bound)
        outs(%init0, %init1 : memref<128x64xf32>, memref<128x64xf32>)
    return
  }

  // The tensor op is pure and speculatable: hoist it even though the loop may
  // execute zero times. Keep both results live through a side-effecting call.
  // CHECK-LABEL: func.func @tensor_licm
  // CHECK:         %[[RESULT:.+]]:2 = linalgx.grouped_matmul
  // CHECK-NEXT:    scf.for
  // CHECK-NEXT:      func.call @consume(%[[RESULT]]#0, %[[RESULT]]#1)
  // CHECK-NEXT:    }
  // CHECK-NEXT:    return
  // CHECK-NEXT:  }
  func.func @tensor_licm(%a: tensor<128x256xf16>, %w0: tensor<2x64x256xf16>,
                         %w1: tensor<2x64x256xf16>, %offs: tensor<2xi32>,
                         %bound: index, %init0: tensor<128x64xf32>,
                         %init1: tensor<128x64xf32>, %upper: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %iv = %zero to %upper step %one {
      %r:2 = linalgx.grouped_matmul
          <{indexing_maps = [#lhs, #lhs, #weights, #weights, #offsets, #out, #out]}>
          ins(%a, %w0, %a, %w1, %offs : tensor<128x256xf16>, tensor<2x64x256xf16>,
              tensor<128x256xf16>, tensor<2x64x256xf16>, tensor<2xi32>)
          max_group_size(%bound)
          outs(%init0, %init1 : tensor<128x64xf32>, tensor<128x64xf32>)
          -> (tensor<128x64xf32>, tensor<128x64xf32>)
      func.call @consume(%r#0, %r#1) : (tensor<128x64xf32>, tensor<128x64xf32>) -> ()
    }
    return
  }

  // Buffer accumulators are read and written on each iteration; hoisting would
  // change their results and would introduce writes for an empty loop.
  // CHECK-LABEL: func.func @buffer_licm
  // CHECK:         scf.for
  // CHECK-NEXT:      linalgx.grouped_matmul
  // CHECK-SAME:        outs(%arg5, %arg6 : memref<128x64xf32>, memref<128x64xf32>)
  // CHECK-NEXT:    }
  // CHECK-NEXT:    return
  // CHECK-NEXT:  }
  func.func @buffer_licm(%a: memref<128x256xf16>, %w0: memref<2x64x256xf16>,
                         %w1: memref<2x64x256xf16>, %offs: memref<2xi32>,
                         %bound: index, %init0: memref<128x64xf32>,
                         %init1: memref<128x64xf32>, %upper: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %iv = %zero to %upper step %one {
      linalgx.grouped_matmul
          <{indexing_maps = [#lhs, #lhs, #weights, #weights, #offsets, #out, #out]}>
          ins(%a, %w0, %a, %w1, %offs : memref<128x256xf16>, memref<2x64x256xf16>,
              memref<128x256xf16>, memref<2x64x256xf16>, memref<2xi32>)
          max_group_size(%bound)
          outs(%init0, %init1 : memref<128x64xf32>, memref<128x64xf32>)
    }
    return
  }

  func.func private @consume(tensor<128x64xf32>, tensor<128x64xf32>)

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    %functions = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %functions {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
