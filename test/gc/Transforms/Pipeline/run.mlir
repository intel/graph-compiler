// RUN: gc-opt %s --gc-cpu-pipeline | gc-cpu-runner -e main -entry-point-result=void | FileCheck %s

module {
func.func @aaa() -> tensor<128xf32> {
    %c2 = arith.constant 2.0 : f32
    %a = tensor.empty() : tensor<128xf32>
    %2 = linalg.fill ins(%c2 : f32) outs(%a : tensor<128xf32>) -> tensor<128xf32>
    return %2 : tensor<128xf32>
}

func.func @main() {
    %result = call @aaa() : ()-> tensor<128xf32>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %c128 step %c1 {
        %4 = tensor.extract %result[%iv] : tensor<128xf32>
        cpuruntime.printf "%f\n" %4 : f32
    }
    return
}
// CHECK-COUNT-128: 2.000000
}