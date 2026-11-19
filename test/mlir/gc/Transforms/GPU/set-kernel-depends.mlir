// RUN: gc-opt %s --set-kernel-depends --split-input-file --verify-diagnostics | FileCheck %s

// Both consumers use the same producer result. consumer0 uses it twice, but
// each kernel must get exactly one dependency on producer.
// CHECK:       module attributes {gc.module = {kernels = {
// CHECK-SAME:     consumer0 = {depends = ["producer"]},
// CHECK-SAME:     consumer1 = {depends = ["producer"]}
// CHECK-SAME:   }}}
func.func @fanout(%input: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
  %empty = tensor.empty() : tensor<8xf32>
  %produced = scf.forall (%i) = (0) to (8) step (4)
      shared_outs(%out = %empty) -> (tensor<8xf32>) {
    %tile = tensor.extract_slice %input[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
    }
  } {gc.kernel_name = "producer"}

  %empty0 = tensor.empty() : tensor<8xf32>
  %empty1 = tensor.empty() : tensor<8xf32>
  %consumer0:2 = scf.forall (%i) = (0) to (8) step (4)
      shared_outs(%out0 = %empty0, %out1 = %empty1)
      -> (tensor<8xf32>, tensor<8xf32>) {
    %tile0 = tensor.extract_slice %produced[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %tile1 = tensor.extract_slice %produced[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile0 into %out0[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
      tensor.parallel_insert_slice %tile1 into %out1[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
    }
  } {gc.kernel_name = "consumer0"}

  %empty2 = tensor.empty() : tensor<8xf32>
  %consumer1 = scf.forall (%i) = (0) to (8) step (4)
      shared_outs(%out = %empty2) -> (tensor<8xf32>) {
    %tile = tensor.extract_slice %produced[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
    }
  } {gc.kernel_name = "consumer1"}
  return %consumer0#0, %consumer0#1, %consumer1
    : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// -----

// Reject an expand_shape outside any tiled kernel.
func.func @untiled_expand(%input: tensor<8xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{untiled reshape outside a GPU kernel; tile or fuse it before collecting kernel dependencies}}
  %expanded = tensor.expand_shape %input [[0, 1]] output_shape [2, 4]
    : tensor<8xf32> into tensor<2x4xf32>
  return %expanded : tensor<2x4xf32>
}

// -----

// Reject the split-kernel chain producer -> collapse_shape -> consumer.
func.func @untiled_collapse(%input: tensor<2x4xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<2x4xf32>
  %produced = scf.forall (%i) = (0) to (2) step (1)
      shared_outs(%out = %empty) -> (tensor<2x4xf32>) {
    %tile = tensor.extract_slice %input[%i, 0] [1, 4] [1, 1]
      : tensor<2x4xf32> to tensor<1x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[%i, 0] [1, 4] [1, 1]
        : tensor<1x4xf32> into tensor<2x4xf32>
    }
  } {gc.kernel_name = "producer"}
  // expected-error @+1 {{untiled reshape outside a GPU kernel; tile or fuse it before collecting kernel dependencies}}
  %collapsed = tensor.collapse_shape %produced [[0, 1]]
    : tensor<2x4xf32> into tensor<8xf32>
  %empty_out = tensor.empty() : tensor<8xf32>
  %result = scf.forall (%i) = (0) to (8) step (4)
      shared_outs(%out = %empty_out) -> (tensor<8xf32>) {
    %tile = tensor.extract_slice %collapsed[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
    }
  } {gc.kernel_name = "consumer"}
  return %result : tensor<8xf32>
}

// -----

// A forall without gc.kernel_name does not establish a GPU kernel boundary.
// Validation stops at the first untiled reshape.
func.func @reshape_in_unmarked_loop(%input: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %result = scf.forall (%i) = (0) to (8) step (4)
      shared_outs(%out = %empty) -> (tensor<8xf32>) {
    %tile = tensor.extract_slice %input[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    // expected-error @+1 {{untiled reshape outside a GPU kernel; tile or fuse it before collecting kernel dependencies}}
    %expanded = tensor.expand_shape %tile [[0, 1]] output_shape [2, 2]
      : tensor<4xf32> into tensor<2x2xf32>
    %collapsed = tensor.collapse_shape %expanded [[0, 1]]
      : tensor<2x2xf32> into tensor<4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %out[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
    }
  }
  return %result : tensor<8xf32>
}

// -----

// Reshapes nested inside a named kernel are local and remain supported.
func.func @reshapes_in_kernel(%input: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<8xf32>
  %result = scf.forall (%i) = (0) to (8) step (4)
      shared_outs(%out = %empty) -> (tensor<8xf32>) {
    %tile = tensor.extract_slice %input[%i] [4] [1]
      : tensor<8xf32> to tensor<4xf32>
    %reshaped = scf.for %j = %c0 to %c1 step %c1
        iter_args(%arg = %tile) -> (tensor<4xf32>) {
      %expanded = tensor.expand_shape %arg [[0, 1]] output_shape [2, 2]
        : tensor<4xf32> into tensor<2x2xf32>
      %collapsed = tensor.collapse_shape %expanded [[0, 1]]
        : tensor<2x2xf32> into tensor<4xf32>
      scf.yield %collapsed : tensor<4xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %reshaped into %out[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
    }
  } {gc.kernel_name = "local"}
  return %result : tensor<8xf32>
}
