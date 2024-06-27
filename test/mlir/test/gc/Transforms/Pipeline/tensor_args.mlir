// RUN: gc-opt %s --gc-cpu-pipeline | FileCheck %s

module {
// CHECK: aaa
// check that the func returns void
// CHECK-NOT: ) -> !llvm.struct<
func.func @aaa(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
    %out = tensor.empty() : tensor<128xf32>
    %2 = linalg.add ins(%a, %b : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
    // CHECK-NOT: memcpy
    return %2 : tensor<128xf32>
}
}