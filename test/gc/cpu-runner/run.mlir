// RUN: gc-opt %s --convert-func-to-llvm --convert-cf-to-llvm | gc-cpu-runner -e main -entry-point-result=void | FileCheck --allow-empty %s
func.func @main() {
    return
}
// CHECK-NOT: any