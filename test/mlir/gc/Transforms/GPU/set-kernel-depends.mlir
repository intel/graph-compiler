// RUN: gc-opt %s --set-kernel-depends | FileCheck %s

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
